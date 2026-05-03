"""
End-to-end Causal-JEPA training on the AoU cardiometabolic cohort.

Run from the repo root:

    python -m backend.ml.train_jepa
    # or with overrides:
    python -m backend.ml.train_jepa --epochs 40 --d-model 128 --n-layers 3

Outputs (under ``backend/ml/results/`` and ``backend/ml/checkpoints/``):
- ``training_log.jsonl``           — per-epoch train/val metrics
- ``training_curves.png``          — loss + cosine + embedding-std plots
- ``report.json``                  — headline metrics (all splits) + config
- ``report.md``                    — human-readable summary
- ``jepa_best.pt``                 — best-val-loss checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

# Force UTF-8 stdout on Windows so rich/table arrows render correctly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

# Make ``backend`` importable as a top-level package when run from repo root.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "backend"))

from ml.data import DataConfig, build_dataloaders                    # noqa: E402
from ml.evaluate import (evaluate_loader, intervention_linear_probe)  # noqa: E402
from ml.jepa_model import CausalJEPA                                  # noqa: E402
from ml.regularizers import sigreg, action_orthogonality_loss         # noqa: E402

console = Console()


# ======================================================================
# Config
# ======================================================================

@dataclass
class TrainConfig:
    # Data
    npz_path: str = "backend/training_data/patient_tensors.npz"
    manifest_path: str = "backend/training_data/manifest.json"
    intervention_map_path: str = "backend/training_data/intervention_map.json"
    max_seq_len: int = 64
    batch_size: int = 64
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    #: Used only for stratified train/val/test indices (fixed across multi-seed runs).
    split_seed: int = 42
    #: When True, append ``_seed{seed}`` to checkpoint / report / log filenames.
    append_seed_to_artifacts: bool = False
    num_workers: int = 0

    # Model (deliberately modest for ~4k samples × 25 features to avoid collapse/overfit)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    z_dim: int = 64              # bumped 16 -> 64 for AdaLN (needs capacity
                                 # to produce distinct (gamma, beta, alpha) per drug)
    predictor_hidden: int = 256
    predictor_layers: int = 3    # 2 -> 3 to let z-conditioning stack
    predictor_style: str = "adaln"  # "adaln" | "concat" (ablation)
    dropout: float = 0.1
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 0.9999

    # Optim
    epochs: int = 80
    lr: float = 2e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    early_stop_patience: int = 15
    loss_type: str = "cosine"       # "cosine" (2-2*cos) or "mse"

    # LeWorldModel-inspired regularizers
    sigreg_lambda: float = 0.1      # weight of SIGReg (Gaussian-prior on s_x)
    sigreg_projections: int = 1024  # M in SIGReg (paper default)
    sigreg_knots: int = 17          # trapezoidal quadrature nodes in [0.2, 4.0]
    sigreg_on_predictor: bool = True  # also regularize predictor output
    orth_lambda: float = 0.5        # weight of action-embedding orthogonality

    # Output
    out_dir: str = "backend/ml/results"
    ckpt_dir: str = "backend/ml/checkpoints"


# ======================================================================
# Helpers
# ======================================================================

def set_seed(seed: int) -> None:
    """Global RNGs for PyTorch, NumPy, and ``random`` (stdlib)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(epoch: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup, returns LR multiplier in [0, 1]."""
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / max(cfg.warmup_epochs, 1)
    progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def ema_schedule(epoch: int, cfg: TrainConfig) -> float:
    """Linearly ramp EMA momentum from start → end over training."""
    t = min(epoch / max(cfg.epochs - 1, 1), 1.0)
    return cfg.ema_momentum_start + (cfg.ema_momentum_end - cfg.ema_momentum_start) * t


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def jepa_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """JEPA prediction loss.

    ``cosine``: 2 - 2 * cos(pred, target).mean()  —  equivalent to
      || l2_norm(pred) - l2_norm(target) ||^2, immune to embedding magnitude drift.
      This is the loss used by BYOL / SimSiam / I-JEPA.
    ``mse``: F.mse_loss(pred, target)  —  raw latent MSE (magnitude-sensitive).
    """
    if loss_type == "cosine":
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(target, dim=-1)
        return 2.0 - 2.0 * (pred_n * tgt_n).sum(dim=-1).mean()
    return F.mse_loss(pred, target)


# ======================================================================
# Epoch loops
# ======================================================================

def train_one_epoch(
    model: CausalJEPA,
    loader,
    optimizer,
    device: torch.device,
    ema_momentum: float,
    grad_clip: float,
    loss_type: str,
    cfg: "TrainConfig",
    progress: Progress,
    task_id,
) -> dict:
    model.train()
    total_loss = total_pred = total_sig = total_orth = total_cos = 0.0
    total_n = 0

    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad()

        # Encode context with gradients
        _, s_x = model.context_encoder(
            batch["context_x"], batch["context_mask"], batch["context_timestamps"],
            src_key_padding_mask=batch["context_padding_mask"],
        )
        # Target encoder is EMA-only (stop-gradient)
        with torch.no_grad():
            _, s_y = model.target_encoder(
                batch["target_y"], batch["target_mask"], batch["target_timestamps"],
                src_key_padding_mask=batch["target_padding_mask"],
            )
        s_y = s_y.detach()

        embedded_z = model.action_embedding(batch["intervention_z"])
        predicted_s_y = model.predictor(s_x, embedded_z)

        L_pred = jepa_loss(predicted_s_y, s_y, loss_type)

        # ---- LeWorldModel-style SIGReg on the learned embeddings ----
        L_sig = s_x.new_zeros(())
        if cfg.sigreg_lambda > 0.0:
            L_sig = sigreg(
                s_x,
                num_projections=cfg.sigreg_projections,
                num_knots=cfg.sigreg_knots,
            )
            if cfg.sigreg_on_predictor:
                L_sig = L_sig + sigreg(
                    predicted_s_y,
                    num_projections=cfg.sigreg_projections,
                    num_knots=cfg.sigreg_knots,
                )

        # ---- Action-embedding orthogonality (fix lisinopril collapse) ----
        L_orth = s_x.new_zeros(())
        if cfg.orth_lambda > 0.0:
            L_orth = action_orthogonality_loss(model.action_embedding)

        loss = L_pred + cfg.sigreg_lambda * L_sig + cfg.orth_lambda * L_orth
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()
        model.update_target_encoder(momentum=ema_momentum)

        with torch.no_grad():
            cos = F.cosine_similarity(predicted_s_y, s_y, dim=-1).mean().item()
        bs = batch["context_x"].shape[0]
        total_loss += loss.item() * bs
        total_pred += L_pred.item() * bs
        total_sig += L_sig.item() * bs
        total_orth += L_orth.item() * bs
        total_cos += cos * bs
        total_n += bs

        progress.update(
            task_id,
            advance=1,
            description=(
                f"[cyan]train  pred={L_pred.item():.4f}  "
                f"sig={L_sig.item():.4f}  orth={L_orth.item():.4f}  cos={cos:+.3f}"
            ),
        )

    n = max(total_n, 1)
    return {
        "loss": total_loss / n,
        "pred_loss": total_pred / n,
        "sigreg": total_sig / n,
        "orth": total_orth / n,
        "cos": total_cos / n,
    }


@torch.no_grad()
def validate_one_epoch(
    model: CausalJEPA, loader, device: torch.device, loss_type: str,
) -> dict:
    model.eval()
    total_loss, total_cos, total_n = 0.0, 0.0, 0
    sy_all = []
    for batch in loader:
        batch = move_to_device(batch, device)
        predicted_s_y, s_y = model(
            context_x=batch["context_x"],
            target_y=batch["target_y"],
            intervention_z=batch["intervention_z"],
            context_mask=batch["context_mask"],
            target_mask=batch["target_mask"],
            context_timestamps=batch["context_timestamps"],
            target_timestamps=batch["target_timestamps"],
            context_padding_mask=batch["context_padding_mask"],
            target_padding_mask=batch["target_padding_mask"],
        )
        loss = jepa_loss(predicted_s_y, s_y, loss_type)
        cos = F.cosine_similarity(predicted_s_y, s_y, dim=-1).mean().item()
        bs = batch["context_x"].shape[0]
        total_loss += loss.item() * bs
        total_cos += cos * bs
        total_n += bs
        sy_all.append(s_y.cpu())

    sy = torch.cat(sy_all, dim=0)
    sy_std = sy.std(dim=0, unbiased=False).mean().item()
    return {
        "loss": total_loss / max(total_n, 1),
        "cos": total_cos / max(total_n, 1),
        "target_std": sy_std,
    }


# ======================================================================
# Reporting
# ======================================================================

def fmt_pct(x: float) -> str:
    return f"{x*100:5.2f}%"


def write_training_curves(log: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in log]
    tr_pred = [r.get("train_pred_loss", r["train_loss"]) for r in log]
    va_loss = [r["val_loss"] for r in log]
    tr_cos = [r["train_cos"] for r in log]
    va_cos = [r["val_cos"] for r in log]
    va_std = [r["val_target_std"] for r in log]
    tr_sig = [r.get("train_sigreg", 0.0) for r in log]
    tr_orth = [r.get("train_orth", 0.0) for r in log]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))

    axes[0].plot(epochs, tr_pred, label="train (pred only)", color="#377eb8", lw=2)
    axes[0].plot(epochs, va_loss, label="val (pred only)",   color="#e41a1c", lw=2)
    axes[0].set_title("Prediction loss (lower = better)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_cos, label="train", color="#377eb8", lw=2)
    axes[1].plot(epochs, va_cos, label="val",   color="#e41a1c", lw=2)
    axes[1].set_title("Predictor <-> Target cosine (higher = better)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("cosine sim")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, va_std, color="#4daf4a", lw=2)
    axes[2].axhline(0.01, color="red", ls="--", lw=1, label="collapse threshold")
    axes[2].set_title("Target-embedding std (collapse watch)")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("mean per-dim std")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, tr_sig,  label="SIGReg",  color="#984ea3", lw=2)
    axes[3].plot(epochs, tr_orth, label="Action-orth", color="#ff7f00", lw=2)
    axes[3].set_title("Regularizer components (train)")
    axes[3].set_xlabel("epoch"); axes[3].set_ylabel("loss")
    axes[3].legend(); axes[3].grid(alpha=0.3)
    axes[3].set_yscale("log")

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_final_tables(
    cfg: TrainConfig,
    meta: dict,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    probe: dict,
    best_epoch: int,
    best_val_loss: float,
    wallclock: float,
    device: str,
) -> tuple[str, list[dict]]:
    """Print rich tables and return (verdict_text, verdict_rows) for Markdown."""

    # ---------- Config & splits ----------
    t = Table(title="Run Configuration", show_lines=False, title_style="bold")
    t.add_column("Field", style="cyan"); t.add_column("Value", style="white")
    t.add_row("Device", device)
    t.add_row("Epochs", str(cfg.epochs))
    t.add_row("Batch size", str(cfg.batch_size))
    t.add_row("Max seq len (weeks)", str(cfg.max_seq_len))
    t.add_row("Model dims", f"d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, z={cfg.z_dim}")
    t.add_row("Loss", cfg.loss_type)
    t.add_row("LR / weight_decay", f"{cfg.lr:g} / {cfg.weight_decay:g}")
    t.add_row("EMA momentum", f"{cfg.ema_momentum_start} -> {cfg.ema_momentum_end}")
    t.add_row(
        "SIGReg (LeWM)",
        f"lambda={cfg.sigreg_lambda:g}  M={cfg.sigreg_projections}  "
        f"knots={cfg.sigreg_knots}  on_pred={cfg.sigreg_on_predictor}",
    )
    t.add_row("Action orthogonality", f"lambda={cfg.orth_lambda:g}")
    t.add_row("Wallclock", f"{wallclock:.1f} s")
    console.print(t)

    s = Table(title="Dataset Splits", show_lines=False, title_style="bold")
    s.add_column("Split", style="cyan")
    s.add_column("N", justify="right")
    for lbl in meta["intervention_labels"]:
        s.add_column(lbl, justify="right")
    for split, counts_key in [("train", "train_intervention_counts"),
                              ("val",   "val_intervention_counts"),
                              ("test",  "test_intervention_counts")]:
        s.add_row(
            split,
            str(meta["split_sizes"][split]),
            *(str(x) for x in meta[counts_key]),
        )
    console.print(s)

    # ---------- Headline metrics ----------
    m = Table(title="Headline Metrics — Causal-JEPA", show_lines=False, title_style="bold")
    m.add_column("Metric", style="cyan")
    m.add_column("Train", justify="right")
    m.add_column("Val", justify="right")
    m.add_column("Test", justify="right")
    m.add_column("Interpretation", style="dim")

    def row(name, key, lower_is_better, interp):
        t_ = train_metrics[key]; v_ = val_metrics[key]; e_ = test_metrics[key]
        m.add_row(name, f"{t_:.4f}", f"{v_:.4f}", f"{e_:.4f}", interp)

    row("Latent MSE", "mse", True, "predicted s_y_hat vs target s_y (lower is better)")
    row("Cosine sim", "cosine_sim", False, "direction match, higher is better (0..1)")
    row("Explained var", "explained_var", False, "1 - Var(resid)/Var(s_y); negative => worse than mean")
    row("Target emb std", "target_embedding_std", False, "avg per-dim std of s_y; ~0 means collapse")
    row("Ctx emb std", "context_embedding_std", False, "avg per-dim std of s_x")
    row("z-sensitivity", "intervention_sensitivity", False, "pairwise dist across z; higher = predictor uses z")
    row("z-sens / std(s_y)", "intervention_sensitivity_relative", False, "relative to natural scale")
    console.print(m)

    # ---------- Linear probe ----------
    p = Table(title="Linear Probe — intervention ← s_x (test split)",
              show_lines=False, title_style="bold")
    p.add_column("Metric", style="cyan"); p.add_column("Value", justify="right")
    p.add_row("Train accuracy", fmt_pct(probe["train_acc"]))
    p.add_row("Test accuracy",  fmt_pct(probe["eval_acc"]))
    p.add_row("Majority baseline", fmt_pct(probe["majority_baseline"]))
    for cls_idx, acc in probe["per_class_eval_acc"].items():
        lbl = meta["intervention_labels"][cls_idx]
        p.add_row(f"  · {lbl} (z={cls_idx})", fmt_pct(acc))
    console.print(p)

    # ---------- Verdict ----------
    verdict_rows = []

    def verdict(name, ok, detail):
        tag = "[bold green]GOOD[/]" if ok else "[bold red]CONCERN[/]"
        verdict_rows.append({"check": name, "ok": bool(ok), "detail": detail})
        return f"{tag}  {name} — {detail}"

    v = []
    v.append(verdict(
        "No representation collapse",
        test_metrics["target_embedding_std"] > 0.05,
        f"target std = {test_metrics['target_embedding_std']:.3f} (>{0.05:.2f})",
    ))
    v.append(verdict(
        "Predictor tracks target direction",
        test_metrics["cosine_sim"] > 0.5,
        f"test cosine = {test_metrics['cosine_sim']:+.3f} (want > 0.5)",
    ))
    v.append(verdict(
        "Positive explained variance",
        test_metrics["explained_var"] > 0.0,
        f"test EV = {test_metrics['explained_var']:+.3f}",
    ))
    v.append(verdict(
        "Predictor conditions on intervention",
        test_metrics["intervention_sensitivity_relative"] > 0.10,
        f"Δ(ŝ_y across z)/std(s_y) = {test_metrics['intervention_sensitivity_relative']:.3f} (want > 0.10)",
    ))
    v.append(verdict(
        "Encoder separates treatment cohorts",
        probe["eval_acc"] > probe["majority_baseline"] + 0.03,
        f"probe acc {probe['eval_acc']*100:.1f}% vs majority {probe['majority_baseline']*100:.1f}%",
    ))
    v.append(verdict(
        "No pathological train/val gap",
        (val_metrics["mse"] - train_metrics["mse"]) < 0.5 * max(train_metrics["mse"], 1e-6),
        f"val−train MSE = {val_metrics['mse']-train_metrics['mse']:+.4f}",
    ))

    panel = Panel(
        "\n".join(v)
        + f"\n\n[bold]Best val loss:[/] {best_val_loss:.4f} at epoch {best_epoch}",
        title="Quality verdicts",
        border_style="bright_blue",
    )
    console.print(panel)

    return "\n".join(line.replace("[bold green]GOOD[/]", "GOOD")
                         .replace("[bold red]CONCERN[/]", "CONCERN")
                         .replace("[bold]", "").replace("[/]", "")
                      for line in v), verdict_rows


def write_markdown_report(path: Path, report: dict) -> None:
    m = report["metrics"]
    cfg = report["config"]
    meta = report["data_meta"]

    lines = [
        f"# Causal-JEPA Training Report",
        f"",
        f"- Cohort: **{report.get('cohort_id','?')}**",
        f"- Device: `{report['device']}`",
        f"- Wallclock: {report['wallclock_sec']:.1f}s",
        f"- Best val loss: **{report['best_val_loss']:.4f}** at epoch {report['best_epoch']}",
        f"",
        f"## Headline metrics",
        f"",
        f"| Metric | Train | Val | Test |",
        f"|---|---:|---:|---:|",
    ]
    for name, key in [
        ("Latent MSE", "mse"),
        ("Cosine sim", "cosine_sim"),
        ("Explained var", "explained_var"),
        ("Target emb std", "target_embedding_std"),
        ("Ctx emb std", "context_embedding_std"),
        ("z-sensitivity", "intervention_sensitivity"),
        ("z-sens / std(s_y)", "intervention_sensitivity_relative"),
    ]:
        lines.append(
            f"| {name} | {m['train'][key]:.4f} | {m['val'][key]:.4f} | {m['test'][key]:.4f} |"
        )

    probe = report["linear_probe"]
    lines += [
        "",
        "## Linear probe (intervention ← s_x)",
        "",
        f"- Train accuracy: **{probe['train_acc']*100:.2f}%**",
        f"- Test accuracy:  **{probe['eval_acc']*100:.2f}%**",
        f"- Majority baseline: {probe['majority_baseline']*100:.2f}%",
    ]
    for cls, acc in probe["per_class_eval_acc"].items():
        lbl = meta["intervention_labels"][cls]
        lines.append(f"  - {lbl}: {acc*100:.2f}%")

    lines += ["", "## Quality verdicts", ""]
    for row in report["verdicts"]:
        badge = "GOOD" if row["ok"] else "CONCERN"
        lines.append(f"- **{badge}** — {row['check']}: {row['detail']}")

    lines += [
        "", "## Splits", "",
        f"- train: {meta['split_sizes']['train']} · val: {meta['split_sizes']['val']}"
        f" · test: {meta['split_sizes']['test']}",
        f"- intervention labels: {meta['intervention_labels']}",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================
# Main
# ======================================================================

def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    tag = getattr(cfg, "_tag", "") or ""
    suffix = f"_{tag}" if tag else ""
    if getattr(cfg, "append_seed_to_artifacts", False):
        suffix = f"{suffix}_seed{cfg.seed}"

    # ---- Data ----
    console.rule("[bold]1. Loading AoU cohort")
    data_cfg = DataConfig(
        npz_path=cfg.npz_path,
        manifest_path=cfg.manifest_path,
        intervention_map_path=cfg.intervention_map_path,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.split_seed,
        num_workers=cfg.num_workers,
    )
    train_loader, val_loader, test_loader, meta = build_dataloaders(data_cfg)
    console.print(
        f"patients={meta['num_patients']}  features={meta['num_features']}"
        f"  interventions={meta['intervention_labels']}"
        f"  splits={meta['split_sizes']}"
    )

    # ---- Model ----
    console.rule("[bold]2. Building Causal-JEPA")
    model = CausalJEPA(
        num_features=meta["num_features"],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        num_interventions=meta["num_interventions"],
        z_dim=cfg.z_dim,
        predictor_hidden=cfg.predictor_hidden,
        predictor_layers=cfg.predictor_layers,
        predictor_style=cfg.predictor_style,
        ema_momentum=cfg.ema_momentum_start,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    console.print(f"parameters: trainable={n_trainable:,}  total={n_total:,}  device={device}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ---- Train ----
    console.rule("[bold]3. Training")
    steps_per_epoch = len(train_loader)
    training_log: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    patience = 0

    t0 = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        outer = progress.add_task("[magenta]epochs", total=cfg.epochs)
        for epoch in range(cfg.epochs):
            # LR schedule
            lr_mult = cosine_warmup_lr(epoch, cfg)
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * lr_mult
            m_ema = ema_schedule(epoch, cfg)

            inner = progress.add_task(f"[cyan]epoch {epoch+1}", total=steps_per_epoch)
            tr = train_one_epoch(
                model, train_loader, optimizer, device,
                m_ema, cfg.grad_clip, cfg.loss_type, cfg, progress, inner,
            )
            progress.remove_task(inner)
            va = validate_one_epoch(model, val_loader, device, cfg.loss_type)

            row = {
                "epoch": epoch + 1,
                "lr": cfg.lr * lr_mult,
                "ema_momentum": m_ema,
                "train_loss": tr["loss"],
                "train_pred_loss": tr["pred_loss"],
                "train_sigreg": tr["sigreg"],
                "train_orth": tr["orth"],
                "train_cos": tr["cos"],
                "val_loss": va["loss"],
                "val_cos": va["cos"],
                "val_target_std": va["target_std"],
            }
            training_log.append(row)
            console.print(
                f"epoch {epoch+1:03d} "
                f"| train pred {tr['pred_loss']:.4f} sig {tr['sigreg']:.4f} "
                f"orth {tr['orth']:.4f} cos {tr['cos']:+.3f} "
                f"| val loss {va['loss']:.4f} cos {va['cos']:+.3f} std {va['target_std']:.3f} "
                f"| lr {cfg.lr*lr_mult:.2e} ema {m_ema:.4f}"
            )

            # Checkpoint on improved val loss
            if va["loss"] < best_val - 1e-6:
                best_val = va["loss"]
                best_epoch = epoch + 1
                patience = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": best_epoch,
                        "val_loss": best_val,
                        "config": asdict(cfg),
                        "data_meta": {
                            k: v for k, v in meta.items()
                            if k != "feature_means" and k != "feature_stds"
                        },
                        "feature_means": meta["feature_means"],
                        "feature_stds": meta["feature_stds"],
                    },
                    ckpt_dir / f"jepa_best{suffix}.pt",
                )
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    console.print(
                        f"[yellow]early stop triggered at epoch {epoch+1} "
                        f"(no improvement for {patience} epochs)[/]"
                    )
                    progress.update(outer, advance=1)
                    break

            progress.update(outer, advance=1)

    wallclock = time.time() - t0

    # Persist per-epoch log
    with (out_dir / f"training_log{suffix}.jsonl").open("w", encoding="utf-8") as fh:
        for row in training_log:
            fh.write(json.dumps(row) + "\n")

    # Curves
    write_training_curves(training_log, out_dir / f"training_curves{suffix}.png")

    # ---- Reload best and evaluate ----
    console.rule("[bold]4. Loading best checkpoint and evaluating")
    ckpt = torch.load(ckpt_dir / f"jepa_best{suffix}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    train_metrics = evaluate_loader(model, train_loader, device, meta["num_interventions"])
    val_metrics = evaluate_loader(model, val_loader, device, meta["num_interventions"])
    test_metrics = evaluate_loader(model, test_loader, device, meta["num_interventions"])

    probe = intervention_linear_probe(
        train_metrics["_sx"], train_metrics["_z"],
        test_metrics["_sx"], test_metrics["_z"],
    )

    # ---- Report ----
    console.rule("[bold]5. Final report")

    def _public(m: dict) -> dict:
        return {k: v for k, v in m.items() if not k.startswith("_")}

    manifest = json.loads(Path(cfg.manifest_path).read_text())
    _, verdict_rows = render_final_tables(
        cfg,
        meta,
        _public(train_metrics),
        _public(val_metrics),
        _public(test_metrics),
        probe,
        best_epoch,
        best_val,
        wallclock,
        str(device),
    )

    report = {
        "cohort_id": manifest.get("cohort_id"),
        "device": str(device),
        "wallclock_sec": wallclock,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "config": asdict(cfg),
        "data_meta": {k: v for k, v in meta.items()
                       if k not in ("feature_means", "feature_stds")},
        "metrics": {
            "train": _public(train_metrics),
            "val":   _public(val_metrics),
            "test":  _public(test_metrics),
        },
        "linear_probe": probe,
        "verdicts": verdict_rows,
    }
    (out_dir / f"report{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(out_dir / f"report{suffix}.md", report)

    console.print(Panel(
        f"Checkpoint : [green]{ckpt_dir/f'jepa_best{suffix}.pt'}[/]\n"
        f"Curves     : [green]{out_dir/f'training_curves{suffix}.png'}[/]\n"
        f"Log        : [green]{out_dir/f'training_log{suffix}.jsonl'}[/]\n"
        f"Report JSON: [green]{out_dir/f'report{suffix}.json'}[/]\n"
        f"Report MD  : [green]{out_dir/f'report{suffix}.md'}[/]",
        title="Artifacts", border_style="green",
    ))


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Train Causal-JEPA on AoU cohort")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--d-model", type=int, default=cfg.d_model)
    p.add_argument("--n-layers", type=int, default=cfg.n_layers)
    p.add_argument("--n-heads", type=int, default=cfg.n_heads)
    p.add_argument("--d-ff", type=int, default=cfg.d_ff)
    p.add_argument("--z-dim", type=int, default=cfg.z_dim)
    p.add_argument("--predictor-hidden", type=int, default=cfg.predictor_hidden)
    p.add_argument("--predictor-layers", type=int, default=cfg.predictor_layers)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--max-seq-len", type=int, default=cfg.max_seq_len)
    p.add_argument("--warmup-epochs", type=int, default=cfg.warmup_epochs)
    p.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    p.add_argument("--early-stop-patience", type=int, default=cfg.early_stop_patience)
    p.add_argument("--seed", type=int, default=cfg.seed,
                   help="Global torch/numpy/random seed (init, shuffle, dropout).")
    p.add_argument("--split-seed", type=int, default=cfg.split_seed,
                   help="Stratified split RNG only; keep fixed when sweeping --seed.")
    p.add_argument("--append-seed-to-artifacts", action="store_true",
                   help="Suffix artifacts with _seed{seed} (multi-seed sweeps).")
    p.add_argument("--npz-path", type=str, default=cfg.npz_path)
    p.add_argument("--out-dir", type=str, default=cfg.out_dir)
    p.add_argument("--ckpt-dir", type=str, default=cfg.ckpt_dir)
    p.add_argument("--loss-type", type=str, default=cfg.loss_type, choices=["cosine", "mse"])
    p.add_argument("--ema-momentum-start", type=float, default=cfg.ema_momentum_start)
    p.add_argument("--ema-momentum-end", type=float, default=cfg.ema_momentum_end)
    p.add_argument("--sigreg-lambda", type=float, default=cfg.sigreg_lambda,
                   help="Weight of SIGReg (Gaussian prior). 0 disables.")
    p.add_argument("--sigreg-projections", type=int, default=cfg.sigreg_projections)
    p.add_argument("--sigreg-knots", type=int, default=cfg.sigreg_knots)
    p.add_argument("--sigreg-on-predictor", action="store_true",
                   default=cfg.sigreg_on_predictor)
    p.add_argument("--no-sigreg-on-predictor", dest="sigreg_on_predictor",
                   action="store_false")
    p.add_argument("--orth-lambda", type=float, default=cfg.orth_lambda,
                   help="Weight of action-embedding orthogonality penalty. 0 disables.")
    p.add_argument("--predictor-style", type=str, default=cfg.predictor_style,
                   choices=["adaln", "concat"],
                   help="Predictor conditioning style: AdaLN-Zero (default) or vanilla concat MLP.")
    p.add_argument("--tag", type=str, default="",
                   help="Optional tag suffix for output files (e.g. jepa_best_<tag>.pt).")
    a = p.parse_args()

    cfg.epochs = a.epochs
    cfg.batch_size = a.batch_size
    cfg.lr = a.lr
    cfg.weight_decay = a.weight_decay
    cfg.d_model = a.d_model
    cfg.n_layers = a.n_layers
    cfg.n_heads = a.n_heads
    cfg.d_ff = a.d_ff
    cfg.z_dim = a.z_dim
    cfg.predictor_hidden = a.predictor_hidden
    cfg.predictor_layers = a.predictor_layers
    cfg.dropout = a.dropout
    cfg.max_seq_len = a.max_seq_len
    cfg.warmup_epochs = a.warmup_epochs
    cfg.grad_clip = a.grad_clip
    cfg.early_stop_patience = a.early_stop_patience
    cfg.seed = a.seed
    cfg.split_seed = a.split_seed
    cfg.append_seed_to_artifacts = a.append_seed_to_artifacts
    cfg.npz_path = a.npz_path
    cfg.out_dir = a.out_dir
    cfg.ckpt_dir = a.ckpt_dir
    cfg.loss_type = a.loss_type
    cfg.ema_momentum_start = a.ema_momentum_start
    cfg.ema_momentum_end = a.ema_momentum_end
    cfg.sigreg_lambda = a.sigreg_lambda
    cfg.sigreg_projections = a.sigreg_projections
    cfg.sigreg_knots = a.sigreg_knots
    cfg.sigreg_on_predictor = a.sigreg_on_predictor
    cfg.orth_lambda = a.orth_lambda
    cfg.predictor_style = a.predictor_style
    cfg._tag = a.tag  # dynamic attribute; used for artifact filenames
    return cfg


if __name__ == "__main__":
    main(parse_args())
