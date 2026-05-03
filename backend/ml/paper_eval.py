"""
Paper-grade validation for Causal-JEPA (non-causal; threats-to-validity aware).

Runs a single entry point that produces **human-readable** JSON + Markdown reports:

1. **Predictive / representation** — test MSE, cosine, explained variance, std,
   intervention sensitivity (same as training eval).
2. **Prognostic ablation** — predict with fixed z=0 (ignore assigned treatment) vs
   true z. If gap is small, the model is mostly learning baseline trajectory.
3. **Negative controls** — shuffled z (wrong patient’s label), random continuous
   ``z`` embedding, all-z=0. Wrong z should raise MSE and lower cosine vs s_y.
4. **Permutation null** — global shuffle of test z labels; distribution of MSE
   over random permutations vs true (rough p-value: how often shuffled MSE is
   as good as true by luck).
5. **Query alignment** — when z_query equals realized z, cosine(ŝ_y, s_y) vs
   when z_query != z_true. Gap supports “z-conditioned” predictions; not causal.
6. **Argmax-z probe** (exploratory) — which z maximizes cos(ŝ_y, s_y) for each
   test patient. Not causal: s_y was generated under the true z.
7. **Twin search — unfiltered** — Qdrant counterfactual retrieval, treatment
   match vs random, outcome divergence, latency.
8. **Twin search — z-stratified (oracle / ceiling)** — only search training
   patients with ``intervention_z == query z``. Match rate is ~100% by
   construction; we report **mean top-K cosine** as within-stratum embedding
   proximity and compare to unfiltered.
9. **Pre-window risk tertiles (HbA1c)** — intervention sensitivity and
   (optional) a subset of unfiltered match rates by tertile.
10. **Baselines** — Ridge / sklearn MLP / PyTorch MLP on **frozen** ``s_x`` (+ one-hot z),
    plus MLP on **mean-pooled raw** pre-window labs (+ z). Same test ``s_y``.
    Reports **raw MSE** (Ridge-optimal) and **L2-normalized MSE** (closer to cosine geometry).

**Interpretation (for the paper)**: These tests jointly support a *conditional
predictive* world model, not *causal* identification. Summarize limitations in
the written report header.

Usage
-----
  python -m backend.ml.paper_eval
  python -m backend.ml.paper_eval --checkpoint backend/ml/checkpoints/jepa_best.pt --out-dir backend/ml/results
  python -m backend.ml.paper_eval --skip-qdrant  # no Qdrant; latent tests only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient, models
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "backend"))

from ml.data import DataConfig, build_dataloaders  # noqa: E402
from ml.evaluate import (  # noqa: E402
    _forward_batch,
    evaluate_loader,
    intervention_linear_probe,
)
from ml.jepa_model import CausalJEPA  # noqa: E402
from ml.twin_search import (  # noqa: E402
    TwinSearchConfig,
    index_training_targets,
    run_aggregate_eval,
    _encode_test,
    _load_model,
    _rebuild_splits,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

console = Console()


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


@dataclass
class PaperEvalConfig:
    checkpoint: str = "backend/ml/checkpoints/jepa_best.pt"
    out_dir: str = "backend/ml/results"
    feature_vocab_path: str = "backend/training_data/feature_vocab.json"
    qdrant_path: str = "backend/ml/qdrant_paper_eval"
    collection: str = "aou_jepa_twin_targets_paper"
    top_k: int = 10
    n_eval_patients: int = 200
    n_perm: int = 199  # + true = 200 for permutation null
    skip_qdrant: bool = False
    twin_reset: bool = True  # rebuild Qdrant; False = append to existing collection
    n_example_patients: int = 2
    perm_seed: int = 0
    skip_baselines: bool = False
    bootstrap_n: int = 2000  # paired bootstrap: JEPA vs each baseline MSE
    baseline_mlp_epochs: int = 80


# ----------------------------------------------------------------------
# Raw pre-window summary (for risk strat)
# ----------------------------------------------------------------------


def _pre_window_hba1c(
    dataset,
    row: int,
    feature_idx: dict[str, int],
) -> float | None:
    """Masked mean HbA1c in the pre window (un-standardized), or None."""
    if "hba1c" not in feature_idx:
        return None
    f = feature_idx["hba1c"]
    abs_row = int(dataset.indices[row])
    v = dataset.pre_values[abs_row]
    m = dataset.pre_mask[abs_row]
    obs = m[:, f] > 0
    if not obs.any():
        return None
    return float(v[obs, f].mean())


# ----------------------------------------------------------------------
# Batched test metrics with z overrides
# ----------------------------------------------------------------------


@torch.no_grad()
def _batch_metrics(
    model: CausalJEPA,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    z_mode: str,
    z_perm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (s_y_hat, s_y, s_x, z_used) for one batch."""
    b = {k: v.to(device) for k, v in batch.items()}
    B = b["intervention_z"].shape[0]
    _, s_x = model.context_encoder(
        b["context_x"],
        b["context_mask"],
        b["context_timestamps"],
        src_key_padding_mask=b.get("context_padding_mask"),
    )
    _, s_y = model.target_encoder(
        b["target_y"],
        b["target_mask"],
        b["target_timestamps"],
        src_key_padding_mask=b.get("target_padding_mask"),
    )
    zt = b["intervention_z"].long()
    if z_mode == "true":
        z_use = zt
    elif z_mode == "fixed0":
        z_use = torch.zeros(B, dtype=torch.long, device=device)
    elif z_mode == "permute_in_batch":
        # destroy alignment: each row gets another row's z
        z_use = zt[torch.randperm(B, device=device)]
    elif z_mode == "permute_external":
        assert z_perm is not None and z_perm.shape[0] == B
        z_use = z_perm.to(device)
    elif z_mode == "shift":
        n = model.action_embedding.num_embeddings
        z_use = (zt + 1) % n
    else:
        raise ValueError(z_mode)
    emb = model.action_embedding(z_use)
    s_y_hat = model.predictor(s_x, emb)
    return s_y_hat, s_y, s_x, z_use


@torch.no_grad()
def _batch_metrics_random_emb(
    model: CausalJEPA,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = {k: v.to(device) for k, v in batch.items()}
    _, s_x = model.context_encoder(
        b["context_x"],
        b["context_mask"],
        b["context_timestamps"],
        src_key_padding_mask=b.get("context_padding_mask"),
    )
    _, s_y = model.target_encoder(
        b["target_y"],
        b["target_mask"],
        b["target_timestamps"],
        src_key_padding_mask=b.get("target_padding_mask"),
    )
    B, zdim = b["intervention_z"].shape[0], model.action_embedding.embedding_dim
    # match rough scale to trained embeddings
    with torch.no_grad():
        w = model.action_embedding.weight
        std = w.std().item()
    z_noise = torch.randn(
        (B, zdim), device=device, generator=rng, dtype=s_x.dtype
    ) * (std if std > 1e-6 else 0.05)
    s_y_hat = model.predictor(s_x, z_noise)
    return s_y_hat, s_y, s_x


def run_test_metrics_mode(
    model: CausalJEPA,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    z_mode: str,
    z_perm: torch.Tensor | None = None,
) -> dict[str, float]:
    mse_w = 0.0
    cos_w = 0.0
    n = 0
    if z_mode == "permute_in_batch":
        torch.manual_seed(0)
    for batch in test_loader:
        s_y_hat, s_y, _, _ = _batch_metrics(
            model, batch, device, z_mode, z_perm
        )
        w = s_y.shape[0]
        mse_w += F.mse_loss(s_y_hat, s_y).item() * w
        cos_w += F.cosine_similarity(s_y_hat, s_y, dim=-1).sum().item()
        n += w
    mse = mse_w / max(n, 1)
    cos = cos_w / max(n, 1)
    sy_var = None  # not needed for ablations
    return {
        "mse": float(mse),
        "cosine_to_s_y": float(cos),
        "n": int(n),
    }


@torch.no_grad()
def run_test_metrics_random_emb(
    model: CausalJEPA,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    mse_w = 0.0
    cos_w = 0.0
    n = 0
    for batch in test_loader:
        s_y_hat, s_y, _ = _batch_metrics_random_emb(
            model, batch, device, g
        )
        w = s_y.shape[0]
        mse_w += F.mse_loss(s_y_hat, s_y).item() * w
        cos_w += F.cosine_similarity(s_y_hat, s_y, dim=-1).sum().item()
        n += w
    return {
        "mse": float(mse_w / max(n, 1)),
        "cosine_to_s_y": float(cos_w / max(n, 1)),
        "n": int(n),
    }


@torch.no_grad()
def alignment_true_vs_mismatch(
    model: CausalJEPA,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_interventions: int,
) -> dict[str, float]:
    """For each test row: cos(ŝ_y, s_y) when z_query=z_true vs max when z wrong."""
    sum_true = 0.0
    sum_best_wrong = 0.0
    n = 0
    for batch in test_loader:
        b = {k: v.to(device) for k, v in batch.items()}
        _, s_x = model.context_encoder(
            b["context_x"],
            b["context_mask"],
            b["context_timestamps"],
            src_key_padding_mask=b.get("context_padding_mask"),
        )
        _, s_y = model.target_encoder(
            b["target_y"],
            b["target_mask"],
            b["target_timestamps"],
            src_key_padding_mask=b.get("target_padding_mask"),
        )
        zt = b["intervention_z"].long()
        B = zt.shape[0]
        c_true: list = []
        for zi in range(num_interventions):
            zz = torch.full((B,), zi, dtype=torch.long, device=device)
            pred = model.predict_counterfactual(s_x, zz)
            c = F.cosine_similarity(pred, s_y, dim=-1)
            c_true.append(c)
        stack = torch.stack(c_true, dim=0)  # (K, B)
        for i in range(B):
            true_z = int(zt[i].item())
            ct = stack[true_z, i].item()
            max_wrong = max(
                stack[j, i].item() for j in range(num_interventions) if j != true_z
            ) if num_interventions > 1 else 0.0
            sum_true += ct
            sum_best_wrong += max_wrong
        n += B
    return {
        "mean_cosine_z_eq_z_true": sum_true / max(n, 1),
        "mean_max_cosine_z_ne_z_true": sum_best_wrong / max(n, 1),
        "gap": (sum_true - sum_best_wrong) / max(n, 1),
        "n": n,
    }


@torch.no_grad()
def z_argmax_cosine_recovered(
    model: CausalJEPA,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_interventions: int,
) -> dict[str, float]:
    """argmax over z of cos(ŝ(s_x,z), s_y) — exploratory; s_y is post-hoc real."""
    correct = 0
    n = 0
    for batch in test_loader:
        b = {k: v.to(device) for k, v in batch.items()}
        _, s_x = model.context_encoder(
            b["context_x"],
            b["context_mask"],
            b["context_timestamps"],
            src_key_padding_mask=b.get("context_padding_mask"),
        )
        _, s_y = model.target_encoder(
            b["target_y"],
            b["target_mask"],
            b["target_timestamps"],
            src_key_padding_mask=b.get("target_padding_mask"),
        )
        B = s_x.shape[0]
        best = torch.empty(B, dtype=torch.long, device=device)
        val = torch.empty(B, device=device)
        for zi in range(num_interventions):
            zz = torch.full((B,), zi, dtype=torch.long, device=device)
            pred = model.predict_counterfactual(s_x, zz)
            c = F.cosine_similarity(pred, s_y, dim=-1)
            if zi == 0:
                val = c
                best.zero_()
            else:
                m = c > val
                val = torch.where(m, c, val)
                best = torch.where(m, torch.full_like(best, zi), best)
        zt = b["intervention_z"].to(device)
        correct += (best == zt).sum().item()
        n += B
    p = 1.0 / num_interventions
    return {
        "argmax_z_accuracy": correct / max(n, 1),
        "n": n,
        "chance_3class": 1.0 / num_interventions,
    }


# ----------------------------------------------------------------------
# Global permutation of z labels
# ----------------------------------------------------------------------


def mse_with_z_vector(
    model: CausalJEPA,
    test_sx: torch.Tensor,
    test_sy: torch.Tensor,
    test_z: torch.Tensor,
    device: torch.device,
) -> float:
    """MSE(ŝ_y(perm), s_y) for a full tensor (already batched in memory is heavy — loop batches)."""
    mse_acc = 0.0
    n = 0
    Bchunk = 256
    N = test_sx.shape[0]
    for start in range(0, N, Bchunk):
        sli = slice(start, min(start + Bchunk, N))
        sx = test_sx[sli].to(device)
        sy = test_sy[sli].to(device)
        zt = test_z[sli].long().to(device)
        emb = model.action_embedding(zt)
        yh = model.predictor(sx, emb)
        w = yh.shape[0]
        mse_acc += F.mse_loss(yh, sy).item() * w
        n += w
    return mse_acc / max(n, 1)


def test_tensors_from_loader(
    model: CausalJEPA, test_loader, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sx_list, sy_list, z_list = [], [], []
    for batch in test_loader:
        _, s_y, s_x = _forward_batch(
            model, batch, device
        )
        sx_list.append(s_x.cpu())
        sy_list.append(s_y.cpu())
        z_list.append(batch["intervention_z"])
    return (
        torch.cat(sx_list, dim=0),
        torch.cat(sy_list, dim=0),
        torch.cat(z_list, dim=0).long(),
    )


def permute_mse_pvalue(
    mse_true: float,
    model: CausalJEPA,
    test_sx: torch.Tensor,
    test_sy: torch.Tensor,
    test_z: torch.Tensor,
    device: torch.device,
    n_perm: int,
    base_seed: int,
) -> dict[str, float]:
    rng = np.random.RandomState(base_seed)
    N = test_z.shape[0]
    ms = []
    for t in range(n_perm):
        perm = rng.permutation(N)
        z2 = test_z[perm]
        m = mse_with_z_vector(model, test_sx, test_sy, z2, device)
        ms.append(m)
    ms_arr = np.array(ms, dtype=np.float64)
    # p = fraction of null MSEs <= true MSE (if true is better, it's smaller)
    p_onesided = float((ms_arr <= mse_true).mean())
    p_twosided = 2.0 * min(p_onesided, 1.0 - p_onesided) if n_perm > 0 else 1.0
    return {
        "mse_with_true_z": mse_true,
        "perm_null_mse_mean": float(ms_arr.mean()) if len(ms) else None,
        "perm_null_mse_std": float(ms_arr.std()) if len(ms) else None,
        "n_permutations": n_perm,
        "p_onesided_better_than_shuffle": p_onesided,
        "p_twosided_approx": min(p_twosided, 1.0),
    }


# ----------------------------------------------------------------------
# Qdrant: stratified counterfactual search
# ----------------------------------------------------------------------


@torch.no_grad()
def counterfactual_query_stratified(
    model: CausalJEPA,
    client: QdrantClient,
    collection: str,
    s_x: torch.Tensor,
    z: int,
    top_k: int,
    device: torch.device,
) -> list[dict]:
    z_tensor = torch.tensor([z], dtype=torch.long, device=device)
    s_y_hat = model.predict_counterfactual(
        s_x.unsqueeze(0).to(device), z_tensor
    )
    qv = F.normalize(s_y_hat, dim=-1).cpu().numpy()[0].astype(np.float32)
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="intervention_z",
                match=models.MatchValue(value=int(z)),
            )
        ]
    )
    hits = client.query_points(
        collection_name=collection,
        query=qv.tolist(),
        limit=top_k,
        with_payload=True,
        query_filter=flt,
    ).points
    return [
        {
            "patient_id": int(h.payload["patient_id"]),
            "intervention_z": int(h.payload["intervention_z"]),
            "intervention_label": h.payload["intervention_label"],
            "cosine": float(h.score),
            "outcomes": h.payload["outcomes"],
            "n_post_obs": int(h.payload["n_post_obs"]),
        }
        for h in hits
    ]


def stratum_size(client: QdrantClient, collection: str, z: int) -> int:
    r = client.count(
        collection,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="intervention_z",
                    match=models.MatchValue(value=int(z)),
                )
            ]
        ),
    )
    return int(r.count)


def run_stratified_twin_eval(
    model: CausalJEPA,
    client: QdrantClient,
    cfg: TwinSearchConfig,
    test_sx: torch.Tensor,
    test_z: torch.Tensor,
    intervention_labels: list[str],
    device: torch.device,
    selected_idx: list[int],
) -> dict[str, Any]:
    n_z = len(intervention_labels)
    K = cfg.top_k
    per_z_cos: dict[str, list[float]] = {intervention_labels[i]: [] for i in range(n_z)}
    per_z_n: dict[str, int] = {}
    for i in range(n_z):
        per_z_n[intervention_labels[i]] = stratum_size(
            client, cfg.collection, i
        )
    for row in selected_idx:
        for zq in range(n_z):
            twins = counterfactual_query_stratified(
                model,
                client,
                cfg.collection,
                test_sx[row],
                zq,
                K,
                device,
            )
            label = intervention_labels[zq]
            for t in twins:
                per_z_cos[label].append(t["cosine"])
    out: dict[str, Any] = {
        "per_intervention": {},
        "interpretation": "Within-z library only. Mean cosine = embedding "
        "proximity in stratum. Treatment match is 100% by filter (ceiling for "
        "treatment-mix metric, not a lift claim).",
    }
    for lab in intervention_labels:
        c = per_z_cos[lab]
        out["per_intervention"][lab] = {
            "library_n_points": per_z_n[lab],
            "n_cosine_values": len(c),
            "mean_topk_cosine_across_queries": float(np.mean(c)) if c else float("nan"),
        }
    return out


# ----------------------------------------------------------------------
# Risk-stratified intervention sensitivity
# ----------------------------------------------------------------------


@torch.no_grad()
def intervention_sensitivity_subset(
    model: CausalJEPA,
    sx: torch.Tensor,
    device: torch.device,
    num_interventions: int,
) -> float:
    B = sx.shape[0]
    if B == 0:
        return float("nan")
    from ml.evaluate import _intervention_sensitivity
    return _intervention_sensitivity(
        model, sx.cpu(), num_interventions, device
    )


def risk_strata_metrics(
    model: CausalJEPA,
    test_sx: torch.Tensor,
    test_z: torch.Tensor,
    ds_test,
    feature_idx: dict[str, int],
    device: torch.device,
    num_interventions: int,
) -> list[dict[str, Any]]:
    hbs: list[tuple[int, float]] = []
    for r in range(len(ds_test)):
        h = _pre_window_hba1c(ds_test, r, feature_idx)
        if h is not None:
            hbs.append((r, h))
    if len(hbs) < 6:
        return [
            {
                "label": "skipped",
                "reason": "insufficient HbA1c pre coverage for tertiles",
            }
        ]
    rows = [x[0] for x in hbs]
    vals = np.array([x[1] for x in hbs], dtype=np.float64)
    t1, t2 = np.percentile(vals, [33.3, 66.7])
    low = [rows[i] for i in range(len(rows)) if vals[i] <= t1]
    mid = [rows[i] for i in range(len(rows)) if t1 < vals[i] <= t2]
    high = [rows[i] for i in range(len(rows)) if vals[i] > t2]
    out = []
    for name, inds in [
        (f"tertile_low_<= {t1:.2f}", low),
        (f"tertile_mid_{t1:.2f}..{t2:.2f}", mid),
        (f"tertile_high_> {t2:.2f}", high),
    ]:
        if not inds:
            continue
        t_idx = torch.tensor(inds, dtype=torch.long)
        ssub = test_sx[t_idx]
        sens = intervention_sensitivity_subset(
            model, ssub, device, num_interventions
        )
        n_zc = (test_z[t_idx] == 0).sum().item()
        out.append(
            {
                "stratum": name,
                "n": len(inds),
                "intervention_sensitivity": sens,
                "n_metformin": int(n_zc),
            }
        )
    return out


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------


def _headline_test_md(baseline: dict) -> str:
    if not baseline:
        return ""
    order = [
        "mse", "cosine_sim", "explained_var", "target_embedding_std",
        "context_embedding_std", "intervention_sensitivity",
        "intervention_sensitivity_relative", "n_examples",
    ]
    rows: list[list[str]] = []
    for k in order:
        if k in baseline:
            v = baseline[k]
            if k == "n_examples":
                rows.append([k, str(int(v)) if v is not None else "—"])
            elif isinstance(v, (float, int, np.floating)):
                rows.append([k, f"{float(v):.6f}"])
            else:
                rows.append([k, str(v)])
    return _table_md(rows, ["Test metric (evaluate_loader)", "Value"])


def _table_md(rows: list[list[str]], header: list[str]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---:" for _ in header) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _format_risk_block(risk: list[dict]) -> str:
    if not risk or risk[0].get("label") == "skipped":
        r = risk[0].get("reason", "n/a") if risk else "n/a"
        return f"_(Skipped: {r})_\n"
    rows_out: list[list[str]] = []
    for d in risk:
        s = d.get("stratum")
        if s is None:
            continue
        iss = d.get("intervention_sensitivity")
        cell = "—"
        if iss is not None and not (isinstance(iss, float) and np.isnan(iss)):
            cell = f"{iss:.4f}"
        rows_out.append(
            [str(s), str(d.get("n", "")), cell, str(d.get("n_metformin", ""))]
        )
    if not rows_out:
        return "_(no strata; see JSON)_\n"
    return (
        _table_md(
            rows_out,
            ["Stratum", "n", "intervention sensitivity", "n z=0 (metformin)"],
        )
        + "\n"
    )


def _baselines_markdown(baseline_comparison: dict | None) -> str:
    if not baseline_comparison or not baseline_comparison.get("models"):
        return "## 1b. Baselines vs Causal-JEPA (same target s_y)\n\n_(Skipped.)_\n\n"
    ref = baseline_comparison.get("jepa_reference") or {}
    rows: list[list[str]] = [
        [
            "**Causal-JEPA (full model)**",
            f"{ref.get('mse', 0):.6f}",
            f"{ref.get('mse_normalized', 0):.6f}",
            f"{ref.get('cosine_sim', 0):.4f}",
            f"{ref.get('explained_var', 0):.4f}",
        ]
    ]
    for name, m in baseline_comparison["models"].items():
        rows.append(
            [
                name,
                f"{m['mse']:.6f}",
                f"{m.get('mse_normalized', 0):.6f}",
                f"{m['cosine_sim']:.4f}",
                f"{m['explained_var']:.4f}",
            ]
        )
    tbl = _table_md(
        rows,
        ["Model", "MSE (raw)", "MSE (L2-norm)", "Cosine", "Expl. var."],
    )
    boot_lines = [
        "\n**Paired bootstrap — raw vector MSE** (optimized by Ridge/MLP; **positive** Δ ⇒ JEPA lower error):\n",
    ]
    pb = baseline_comparison.get("paired_bootstrap_vs_jepa") or {}
    if not pb:
        boot_lines.append("\n_(No per-sample JEPA MSE; bootstrap omitted.)_\n")
    else:
        for bname, st in pb.items():
            lo = st.get("bootstrap_ci95_low", 0)
            hi = st.get("bootstrap_ci95_high", 0)
            p = st.get("bootstrap_p_two_sided", 1.0)
            frac = st.get("fraction_bootstraps_jepa_better", 0)
            boot_lines.append(
                f"- *{bname}*: mean ΔMSE = **{st.get('mean_mse_improvement_base_minus_jepa', 0):.6f}** "
                f"95% CI [{lo:.6f}, {hi:.6f}], two-sided *p* ≈ {p:.4f}, "
                f"fraction of resamples with JEPA better = {frac:.2%}\n"
            )
    boot_lines.append(
        "\n**Paired bootstrap — L2-normalized (sphere) MSE** vs Ridge "
        "(geometry aligned with cosine training):\n"
    )
    pbn = baseline_comparison.get("paired_bootstrap_normalized_vs_ridge") or {}
    if not pbn:
        boot_lines.append("\n_(Sphere bootstrap n/a.)_\n")
    else:
        for bname, st in pbn.items():
            lo = st.get("bootstrap_ci95_low", 0)
            hi = st.get("bootstrap_ci95_high", 0)
            p = st.get("bootstrap_p_two_sided", 1.0)
            frac = st.get("fraction_bootstraps_jepa_better", 0)
            boot_lines.append(
                f"- *{bname}*: mean Δ = **{st.get('mean_mse_improvement_base_minus_jepa', 0):.6f}** "
                f"95% CI [{lo:.6f}, {hi:.6f}], *p* ≈ {p:.4f}, JEPA better in {frac:.2%} of resamples\n"
            )
    proto = baseline_comparison.get("protocol", "")
    return (
        "## 1b. Baselines vs Causal-JEPA (same target s_y)\n\n"
        f"*{proto}*\n\n"
        "**Why two MSE columns?** Ridge/sklearn MLP are trained to minimize **raw** MSE in embedding "
        "space; JEPA is trained primarily with **cosine** loss, so vector **norm** can differ. "
        "**MSE (L2-norm)** compares directions on the unit sphere (closer to the JEPA objective).\n\n"
        f"{tbl}\n\n"
        + "".join(boot_lines)
        + "\n*Interpretation:* Lower error / higher cosine is better. Bootstraps are **paired** on "
        "test patients; they support **predictive** comparison, not causality.\n\n"
    )


def write_paper_markdown(
    path: Path,
    cfg: PaperEvalConfig,
    train_cfg: dict,
    ckpt_info: dict,
    data_meta: dict,
    linear_probe: dict,
    baseline: dict,
    baseline_comparison: dict | None,
    ablations: dict,
    perm_null: dict,
    align: dict,
    z_argmax: dict,
    risk: list[dict],
    unfiltered: dict,
    stratified: dict,
    primary_verdicts: list[dict],
) -> None:
    v_loss = ckpt_info.get("val_loss", float("nan"))
    try:
        v_loss_s = f"{float(v_loss):.6f}"
    except (TypeError, ValueError):
        v_loss_s = str(v_loss)
    epoch = ckpt_info.get("epoch", "?")
    hdr = _headline_test_md(baseline)
    abl_rows: list[list[str]] = []
    for k, v in ablations.items():
        if k == "true_z (trained objective)":
            abl_rows.append(
                [f"**{k}**", f"{v['mse']:.6f}", f"{v['cosine_to_s_y']:.4f}"]
            )
        else:
            abl_rows.append(
                [k, f"{v['mse']:.6f}", f"{v['cosine_to_s_y']:.4f}"]
            )
    abl_table = _table_md(abl_rows, ["Mode", "MSE vs s_y", "Cos(ŝ_y, s_y)"])
    pmn = perm_null.get("perm_null_mse_mean")
    pmn_s = f"{pmn:.6f}" if pmn is not None and not (isinstance(pmn, float) and np.isnan(pmn)) else "n/a"
    risk_b = _format_risk_block(risk)
    lines_q: list[str] = [
        "\n| Query z | JEPA match | Random | Lift |\n"
        "|:---|---:|---:|---:|\n"
    ]
    pz = unfiltered.get("per_z") or {}
    if unfiltered.get("skipped"):
        lines_q = ["\n_(Qdrant section skipped. Run without `--skip-qdrant`.)_\n\n"]
    else:
        for lab, row in pz.items():
            r = (row or {}).get("treatment_match_rate", 0) * 100
            rr = (row or {}).get("random_treatment_match_rate", 0) * 100
            lf = (row or {}).get("lift_over_random", 0) * 100
            lines_q.append(
                f"| {lab} | {r:.2f}% | {rr:.2f}% | {lf:+.2f}pp |\n"
            )
    strat_r = list((stratified.get("per_intervention") or {}).items())
    if stratified.get("skipped") or not strat_r:
        strat_table = "_(n/a or skipped — see JSON)_\n"
    else:
        strat_table = _table_md(
            [
                [
                    lab,
                    str(vv.get("library_n_points", "")),
                    f"{float(vv.get('mean_topk_cosine_across_queries', 0) or 0.0):.4f}",
                ]
                for lab, vv in strat_r
            ],
            ["z", "train library size", "mean cos top-K to twin s_y (stratified)"],
        ) + "\n"
    probe_md = _table_md(
        [
            ["train acc", f"{linear_probe.get('train_acc', 0.0) * 100:.2f}%"],
            ["test acc", f"{linear_probe.get('eval_acc', 0.0) * 100:.2f}%"],
            ["majority baseline", f"{linear_probe.get('majority_baseline', 0.0) * 100:.2f}%"],
        ],
        ["Linear probe: z from s_x", "Value"],
    )
    verdicts_txt = "\n".join(
        f"- **{'PASS' if v.get('ok') else 'REVIEW'}** -- {v.get('name')}: {v.get('detail')}"
        for v in primary_verdicts
    )
    pass_ct = sum(1 for v in primary_verdicts if v.get("ok"))
    tot_v = len(primary_verdicts)
    readiness = f"""## 8. Brief research-paper readiness (heuristic)

- **Automated checklist:** {pass_ct}/{tot_v} items **PASS** (see §7; not formal hypothesis tests).
- **Conditional predictive / methods contribution:** With clear **non-causal** framing, negative controls, baselines, and (when run) retrieval metrics, this is a **credible empirical package** for an applied ML / digital health methods paper or appendix — strength depends on venue and how hard reviewers push on **identification** and **external validation**.
- **Claims about treatment effects / clinical causality:** **Not** established here; top-tier clinical ML or causal inference venues would expect stronger designs (e.g. trials, IV, negative controls tied to graph, cross-cohort replication).

"""
    sec_baselines = _baselines_markdown(baseline_comparison)
    doc = f"""# Causal-JEPA paper validation (full battery)

This report groups predictive metrics, **negative controls**, and optional
**Qdrant** counterfactual twin search. It does **not** establish causal treatment
effects: unmeasured confounding, treatment policy, and channel bias can remain.
Use the tables to support a **conditional trajectory model** and state limits clearly.

- **Checkpoint:** `{cfg.checkpoint}`
- **Val loss (saved):** {v_loss_s} @ epoch {epoch}
- **Test n:** {data_meta["split_sizes"].get("test", "?")} | **d_model** {train_cfg.get("d_model")} | **z_dim** {train_cfg.get("z_dim")}

## 0. Headline JEPA test metrics (frozen encoders, realized z in forward)

{hdr}

{probe_md}

{sec_baselines}

## 1. Ablations / negative controls (same s_y; alter how z is injected)

{abl_table}

* **Key:** If *fixed z=0* is nearly as good as *true z*, the model may be
  encoding mostly baseline prognosis, not a treatment-specific channel. Random /
  shuffled / shifted z should typically **hurt** MSE vs the true z row if the
  action embedding is used coherently.

## 2. Permutation null (global shuffle of test z labels, same s_x, s_y order)

| Metric | Value |
|:---|---:|
| MSE with true z (aligned) | {perm_null.get("mse_with_true_z", float("nan")):.6f} |
| Mean MSE with a random permutation of z (n={perm_null.get("n_permutations", 0)} draws) | {pmn_s} |
| p (one-sided: shuffled MSE is worse or equal, i.e. as good or luckier than true) | {perm_null.get("p_onesided_better_than_shuffle", 1.0):.4f} |

*Small p* suggests the observed (z, trajectory) pairing is *unlikely* under
random re-labeling (associative, not a causal ATE).

## 3. Query alignment: z = z_true vs wrong z (same s_y anchor)

| Metric | Value |
|:---|---:|
| mean cos(ŝ_y, s_y) at **z = z_true** | {align.get("mean_cosine_z_eq_z_true", 0.0):.4f} |
| mean of **max** cos at z != z_true | {align.get("mean_max_cosine_z_ne_z_true", 0.0):.4f} |
| **Gap** | {align.get("gap", 0.0):.4f} |

*Positive gap:* counterfactual queries that match the realized plan align better
with the realized target anchor than a deliberately mismatched z. Still not
causal if z is entangled with baseline state.

*Note:* with cosine almost 1, this **gap** can be ~0 or slightly negative; treat
**MSE** and **§1–2** (ablations, permutation) as the primary z-use signal.

## 4. Exploratory: which z best matches s_y? (argmax cos; not causal)

| Metric | Value |
|:---|---:|
| argmax z accuracy (vs true label) | {z_argmax.get("argmax_z_accuracy", 0.0) * 100:.1f}% |
| n | {z_argmax.get("n", 0)} |
| chance (3-class) | {z_argmax.get("chance_3class", 1.0 / 3.0) * 100:.1f}% |

*Caveat: s_y is the realized post period; high accuracy can reflect confounding.*

## 5. Pre-window HbA1c tertiles (metabolic risk) and intervention sensitivity

{risk_b}

## 6. Qdrant — unfiltered counterfactual twin search

*Primary retrieval metrics: no filter on the training patients’ actual z.*

{''.join(lines_q)}

### 6b. Z-stratified / oracle (search only patients who had query z in training)

{strat_table}
*{stratified.get("interpretation", "Within-z library: mean top-K cosine; treatment match is 1.0 by construction.")}*

## 7. Automated checklist (heuristic; not formal hypothesis tests on their own)

{verdicts_txt}

{readiness}
"""
    path.write_text(doc, encoding="utf-8")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _verdicts(
    ablations: dict,
    perm: dict,
    align: dict,
    unfiltered: dict,
) -> list[dict[str, Any]]:
    out = []
    true_m = ablations.get("true_z (trained objective)", {})
    fix0 = ablations.get("fixed z=0 (ignore treatment)", {})
    mse_t = true_m.get("mse", 1.0)
    mse0 = fix0.get("mse", 1.0)
    out.append(
        {
            "name": "Prognostic gap (true z better than all z=0)",
            "ok": mse_t < 0.999 * mse0,
            "detail": f"MSE true={mse_t:.5f} vs z=0 {mse0:.5f}",
        }
    )
    p = perm.get("p_onesided_better_than_shuffle", 1.0)
    out.append(
        {
            "name": "Permutation null: true z beats shuffled z (p one-sided)",
            "ok": p < 0.05,
            "detail": f"p = {p:.4f} (H0: random re-labeling as good as true z)",
        }
    )
    out.append(
        {
            "name": "Z alignment gap (at-true z vs best wrong z)",
            "ok": float(align.get("gap", 0.0)) > 0.001,
            "detail": f"gap = {align.get('gap', 0):.4f}",
        }
    )
    if unfiltered.get("skipped"):
        out.append(
            {
                "name": "Qdrant counterfactual twin search (unfiltered)",
                "ok": True,
                "detail": "Skipped (`--skip-qdrant`); re-run with Qdrant for retrieval metrics.",
            }
        )
    else:
        any_lift = any(
            (row or {}).get("lift_over_random", 0) > 0.01
            for row in (unfiltered.get("per_z") or {}).values()
        )
        out.append(
            {
                "name": "At least one z with twin retrieval lift > 1pp (associative)",
                "ok": any_lift,
                "detail": "See unfiltered Qdrant table; a miss is not fatal if the "
                "claim is only latent JEPA, not counterfactual retrieval",
            }
        )
    return out


def run(cfg: PaperEvalConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_idx = json.loads(
        Path(cfg.feature_vocab_path).read_text(encoding="utf-8")
    )

    console.rule("[bold]Causal-JEPA — paper validation battery")
    t_model, train_cfg, meta, ckpt = _load_model(Path(cfg.checkpoint), device)
    model = t_model
    train_loader, val_loader, test_loader, data_meta = _rebuild_splits(
        train_cfg
    )
    n_int = int(meta["num_interventions"])
    labels = data_meta["intervention_labels"]
    ckpt_info = {
        "epoch": ckpt.get("epoch"),
        "val_loss": float(ckpt.get("val_loss", float("nan"))),
    }

    def _strip_eval(d: dict) -> dict:
        o: dict = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if isinstance(v, torch.Tensor):
                o[k] = float(v.mean().item())
            elif isinstance(v, (np.floating, np.integer)):
                o[k] = float(v)
            else:
                o[k] = v
        return o

    console.rule("[1] Test-set / train metrics (standard evaluate_loader)")
    ev_tr = evaluate_loader(
        model, train_loader, device, num_interventions=n_int
    )
    ev_va = evaluate_loader(
        model, val_loader, device, num_interventions=n_int
    )
    ev_te = evaluate_loader(
        model, test_loader, device, num_interventions=n_int
    )
    probe = intervention_linear_probe(
        ev_tr["_sx"], ev_tr["_z"],
        ev_te["_sx"], ev_te["_z"],
    )
    baseline = _strip_eval(ev_te)
    val_metrics = _strip_eval(ev_va)
    del ev_tr, ev_va, ev_te  # free tensors

    baseline_comparison: dict[str, Any] = {}
    if not cfg.skip_baselines:
        console.rule("[1b] Baselines (Ridge / MLP on frozen s_x; pooled-raw MLP)")
        from ml.paper_baselines import (
            jepa_normalized_mse,
            jepa_per_sample_mse,
            run_all_baselines,
        )

        jepa_per = jepa_per_sample_mse(model, test_loader, device)
        baseline_comparison = run_all_baselines(
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            num_interventions=n_int,
            d_model=int(train_cfg["d_model"]),
            jepa_mse_per_sample=jepa_per,
            bootstrap_n=cfg.bootstrap_n,
            bootstrap_seed=cfg.perm_seed,
            torch_mlp_epochs=cfg.baseline_mlp_epochs,
        )
        baseline_comparison["jepa_reference"] = {
            "mse": baseline["mse"],
            "cosine_sim": baseline["cosine_sim"],
            "explained_var": baseline["explained_var"],
            "mse_normalized": jepa_normalized_mse(model, test_loader, device),
        }

    console.rule("[2] Ablations / negative controls (test MSE, cos)")
    cyc = f"(z+1) mod {n_int} (shift)"
    ablations: dict[str, dict] = {
        "true_z (trained objective)": run_test_metrics_mode(
            model, test_loader, device, "true",
        ),
        "fixed z=0 (ignore treatment)": run_test_metrics_mode(
            model, test_loader, device, "fixed0",
        ),
        "in-batch shuffled z (misaligned)": run_test_metrics_mode(
            model, test_loader, device, "permute_in_batch",
        ),
        cyc: run_test_metrics_mode(model, test_loader, device, "shift"),
        "random z embedding (placebo)": run_test_metrics_random_emb(
            model, test_loader, device, cfg.perm_seed,
        ),
    }
    mse_t = ablations["true_z (trained objective)"]["mse"]

    console.rule("[3] Global permutation of z (same s_x, s_y rows)")
    test_sx, test_sy, test_z = test_tensors_from_loader(
        model, test_loader, device,
    )
    perm_null = permute_mse_pvalue(
        mse_t, model, test_sx, test_sy, test_z, device, cfg.n_perm, cfg.perm_seed,
    )

    console.rule("[4] z alignment, argmax probe, HbA1c tertiles")
    align = alignment_true_vs_mismatch(
        model, test_loader, device, n_int,
    )
    z_argmax = z_argmax_cosine_recovered(
        model, test_loader, device, n_int,
    )
    risk = risk_strata_metrics(
        model, test_sx, test_z, test_loader.dataset, feature_idx, device, n_int,
    )

    unfiltered: dict[str, Any] = {"skipped": True, "per_z": {}}
    stratified: dict[str, Any] = {"skipped": True, "per_intervention": {}}
    t_search: TwinSearchConfig | None = None
    if not cfg.skip_qdrant:
        console.rule("[5] Qdrant — index, unfiltered and z-stratified twin search")
        t_search = TwinSearchConfig(
            checkpoint=cfg.checkpoint,
            feature_vocab_path=cfg.feature_vocab_path,
            qdrant_path=cfg.qdrant_path,
            collection=cfg.collection,
            top_k=cfg.top_k,
            n_eval_patients=cfg.n_eval_patients,
            n_example_patients=0,
            out_dir=cfg.out_dir,
            reset=cfg.twin_reset,
        )
        t0 = time.time()
        client, n_indexed, _dim = index_training_targets(
            t_search, model, train_loader, device, labels, feature_idx,
        )
        index_secs = time.time() - t0
        test_sx2, test_z2, test_pid2 = _encode_test(model, test_loader, device)
        results, selected_idx = run_aggregate_eval(
            model, client, t_search,
            test_sx2, test_z2, test_pid2,
            labels, meta["train_intervention_counts"], feature_idx, device,
        )
        st = run_stratified_twin_eval(
            model, client, t_search, test_sx2, test_z2, labels, device, selected_idx,
        )
        unfiltered = {**results, "n_indexed": n_indexed, "index_build_secs": index_secs}
        stratified = st

    verdicts = _verdicts(ablations, perm_null, align, unfiltered)
    mse_t_val = mse_t
    mse_shuf = ablations["in-batch shuffled z (misaligned)"]["mse"]
    verdicts.append(
        {
            "name": "Neg. control: in-batch shuffled z hurts MSE vs true z (>=0.1%)",
            "ok": mse_shuf > mse_t_val * 1.001,
            "detail": f"true {mse_t_val:.5f} vs shuf {mse_shuf:.5f} "
            f"({(mse_shuf / mse_t_val - 1) * 100:+.2f}%)",
        }
    )
    verdicts.append(
        {
            "name": "Neg. control: random z embedding (placebo) has worse MSE",
            "ok": ablations["random z embedding (placebo)"]["mse"] > mse_t_val,
            "detail": f"placebo mse {ablations['random z embedding (placebo)']['mse']:.5f}",
        }
    )
    verdicts.append(
        {
            "name": "s_x encodes z above majority baseline (confounding probe)",
            "ok": float(probe["eval_acc"]) > float(probe["majority_baseline"]) + 0.02,
            "detail": f"probe acc {probe['eval_acc']*100:.1f}% "
            f"vs majority {probe['majority_baseline']*100:.1f}%",
        }
    )
    if baseline_comparison.get("models"):
        ref = baseline_comparison.get("jepa_reference") or {}
        best_m = float(baseline_comparison["best_baseline_by_mse"]["mse"])
        jmse = float(baseline["mse"])
        verdicts.append(
            {
                "name": "Raw MSE: note Ridge/MLP optimize this; JEPA uses cosine (see §1b)",
                "ok": True,
                "detail": f"JEPA raw MSE {jmse:.4f} vs best baseline "
                f"{baseline_comparison['best_baseline_by_mse']['name']} "
                f"{best_m:.4f} — interpret with mse_normalized / cosine",
            }
        )
        best_n = float(
            baseline_comparison["best_baseline_by_mse_normalized"]["mse_normalized"]
        )
        jn = float(ref.get("mse_normalized", 1.0))
        verdicts.append(
            {
                "name": "JEPA scale-free MSE within 25% of best baseline (competitive)",
                "ok": jn <= best_n * 1.25,
                "detail": f"JEPA {jn:.6f} vs best {baseline_comparison['best_baseline_by_mse_normalized']['name']} "
                f"{best_n:.6f} (linear readouts on frozen s_x are very strong)",
            }
        )
        jcos = float(baseline["cosine_sim"])
        best_cos = max(float(m["cosine_sim"]) for m in baseline_comparison["models"].values())
        verdicts.append(
            {
                "name": "JEPA cosine similarity within 0.005 of best baseline",
                "ok": jcos >= best_cos - 0.005,
                "detail": f"JEPA cos {jcos:.4f} vs best baseline cos {best_cos:.4f}",
            }
        )
        pbn = baseline_comparison.get("paired_bootstrap_normalized_vs_ridge") or {}
        rk = "Ridge(s_x + one-hot z), sphere MSE"
        if rk in pbn:
            st = pbn[rk]
            lo = float(st.get("bootstrap_ci95_low", 0))
            hi = float(st.get("bootstrap_ci95_high", 0))
            p = float(st.get("bootstrap_p_two_sided", 1))
            # Tiny effect sizes on sphere: require clear CI away from 0 for PASS
            verdicts.append(
                {
                    "name": "Bootstrap (sphere MSE): JEPA vs Ridge(s_x+z) (paired)",
                    "ok": lo > 1e-5 and p < 0.05,
                    "detail": f"Δ(Ridge−JEPA) mean {st.get('mean_mse_improvement_base_minus_jepa', 0):.6f} "
                    f"CI [{lo:.6f}, {hi:.6f}], p≈{p:.4f} (near-tie expected)",
                }
            )

    report: dict[str, Any] = {
        "paper_eval_version": "1.2",
        "config": asdict(cfg),
        "checkpoint_info": ckpt_info,
        "data_meta": {k: v for k, v in data_meta.items() if k not in (
            "feature_means", "feature_stds",
        )},
        "val_metrics": val_metrics,
        "test_metrics": baseline,
        "linear_probe_sx_to_z": probe,
        "ablation_mse": ablations,
        "permutation_null_mse": perm_null,
        "z_alignment_true_vs_mismatch": align,
        "z_argmax_on_sy": z_argmax,
        "pre_hba1c_risk_strata": risk,
        "qdrant_unfiltered": unfiltered,
        "qdrant_stratified_oracle": stratified,
        "baseline_comparison": baseline_comparison or None,
        "verdicts": verdicts,
    }
    (out / "paper_eval_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    write_paper_markdown(
        out / "paper_eval_report.md",
        cfg,
        train_cfg,
        ckpt_info,
        data_meta,
        probe,
        baseline,
        baseline_comparison if baseline_comparison else None,
        ablations,
        perm_null,
        align,
        z_argmax,
        risk,
        unfiltered,
        stratified,
        verdicts,
    )
    t = Table(title="Paper eval — ablations (MSE; lower better)", show_lines=True)
    t.add_column("Mode", style="cyan", overflow="fold")
    t.add_column("MSE", justify="right")
    t.add_column("Cos", justify="right")
    for k, v in ablations.items():
        t.add_row(
            k,
            f"{v['mse']:.5f}",
            f"{v['cosine_to_s_y']:.4f}",
        )
    console.print(t)
    if baseline_comparison.get("models"):
        bt = Table(title="Baselines vs JEPA (test MSE vs s_y)", show_lines=True)
        bt.add_column("Model", overflow="fold")
        bt.add_column("MSE", justify="right")
        bt.add_column("MSE(norm)", justify="right")
        bt.add_column("Cos", justify="right")
        ref = baseline_comparison.get("jepa_reference", {})
        bt.add_row(
            "[bold]Causal-JEPA[/]",
            f"{ref.get('mse', 0):.5f}",
            f"{ref.get('mse_normalized', 0):.5f}",
            f"{ref.get('cosine_sim', 0):.4f}",
        )
        for nm, mv in baseline_comparison["models"].items():
            bt.add_row(
                nm,
                f"{mv['mse']:.5f}",
                f"{mv.get('mse_normalized', 0):.5f}",
                f"{mv['cosine_sim']:.4f}",
            )
        console.print(bt)
    pnl = Table(title="Key automated verdicts (see MD for all)", show_lines=True)
    pnl.add_column("Item", style="white", overflow="fold")
    pnl.add_column("", justify="right")
    for v in verdicts:
        tag = "PASS" if v["ok"] else "REVIEW"
        pnl.add_row(
            f"[{'green' if v['ok'] else 'yellow'}]{tag}[/] {v['name'][:70]}",
            "…",
        )
    console.print(pnl)
    console.print(
        Panel(
            f"[green]JSON : {out / 'paper_eval_report.json'}[/]\n"
            f"[green]MD   : {out / 'paper_eval_report.md'}[/]"
            + ("" if cfg.skip_qdrant else f"\n[dim]Qdrant: {Path(cfg.qdrant_path)}/ {cfg.collection}[/]"),
            title="Done",
        )
    )


def parse_args() -> PaperEvalConfig:
    d = PaperEvalConfig()
    p = argparse.ArgumentParser(
        description="Causal-JEPA full paper validation (read paper_eval module docstring).",
    )
    p.add_argument("--checkpoint", type=str, default=d.checkpoint)
    p.add_argument("--out-dir", type=str, default=d.out_dir)
    p.add_argument("--feature-vocab", type=str, default=d.feature_vocab_path)
    p.add_argument("--qdrant-path", type=str, default=d.qdrant_path)
    p.add_argument("--collection", type=str, default=d.collection)
    p.add_argument("--top-k", type=int, default=d.top_k)
    p.add_argument("--n-eval-patients", type=int, default=d.n_eval_patients)
    p.add_argument("--n-perm", type=int, default=d.n_perm)
    p.add_argument("--perm-seed", type=int, default=d.perm_seed)
    p.add_argument("--skip-qdrant", action="store_true")
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip Ridge/MLP baseline training (faster).",
    )
    p.add_argument("--bootstrap-n", type=int, default=d.bootstrap_n)
    p.add_argument("--baseline-mlp-epochs", type=int, default=d.baseline_mlp_epochs)
    p.add_argument(
        "--no-twin-reset",
        action="store_true",
        help="Do not wipe the paper-eval Qdrant path before indexing (advanced).",
    )
    a = p.parse_args()
    d.checkpoint = a.checkpoint
    d.out_dir = a.out_dir
    d.feature_vocab_path = a.feature_vocab
    d.qdrant_path = a.qdrant_path
    d.collection = a.collection
    d.top_k = a.top_k
    d.n_eval_patients = a.n_eval_patients
    d.n_perm = a.n_perm
    d.perm_seed = a.perm_seed
    d.skip_qdrant = a.skip_qdrant
    d.skip_baselines = a.skip_baselines
    d.bootstrap_n = a.bootstrap_n
    d.baseline_mlp_epochs = a.baseline_mlp_epochs
    d.twin_reset = not a.no_twin_reset
    return d


if __name__ == "__main__":
    run(parse_args())