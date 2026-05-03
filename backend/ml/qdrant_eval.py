"""
Qdrant-backed patient-embedding retrieval test.

Pipeline
--------
1. Load the best Causal-JEPA checkpoint (produced by ``train_jepa.py``).
2. Re-materialize the same train/val/test splits (deterministic seed).
3. For every patient, compute the pre-intervention context embedding s_x.
4. Upsert all embeddings into a **local** Qdrant collection (on-disk storage,
   no Docker / no network — uses ``QdrantClient(path=...)``). Payload includes
   patient_id, intervention label, split.
5. Run retrieval evaluation on the test split:
     - For each test patient, fetch top-k neighbors restricted to the TRAIN split
     - Measure:
         * top-1 / top-5 / top-10 intervention concordance (same-drug rate)
         * majority-class baseline comparison
         * mean / median / p95 query latency
         * mean top-k cosine similarity
6. Render a rich console report, save JSON + markdown report.

Why this is a meaningful test
-----------------------------
- It exercises the exact production path: encoder -> vector DB -> nearest-
  neighbor lookup. Any bug (dim mismatch, device, normalization) surfaces here.
- Same-intervention retrieval rate is a direct quality signal for the
  downstream "patient twin" use-case: if the embeddings encode the clinical
  state that led to a given prescription, similar patients should tend to
  have received similar prescriptions.
- Top-1 concordance > majority baseline demonstrates the embedding carries
  intervention-discriminative information beyond class imbalance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# ``backend`` importable when run from repo root
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "backend"))

from ml.data import DataConfig, build_dataloaders                    # noqa: E402
from ml.jepa_model import CausalJEPA                                  # noqa: E402

# UTF-8 stdout on Windows for rich
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

console = Console()


# ======================================================================
# Config
# ======================================================================

@dataclass
class QdrantEvalConfig:
    checkpoint: str = "backend/ml/checkpoints/jepa_best.pt"
    qdrant_path: str = "backend/ml/qdrant_local"
    collection: str = "aou_jepa_patients"
    reset: bool = True           # wipe local qdrant dir before running
    top_ks: tuple = (1, 5, 10, 25)
    upsert_batch: int = 256
    num_examples_to_show: int = 3
    out_dir: str = "backend/ml/results"


# ======================================================================
# Helpers
# ======================================================================

def _load_checkpoint(cfg: QdrantEvalConfig, device: torch.device):
    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run `python -m backend.ml.train_jepa` first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = ckpt["config"]
    meta = ckpt["data_meta"]

    model = CausalJEPA(
        num_features=meta["num_features"],
        d_model=train_cfg["d_model"],
        n_heads=train_cfg["n_heads"],
        n_layers=train_cfg["n_layers"],
        d_ff=train_cfg["d_ff"],
        dropout=train_cfg["dropout"],
        num_interventions=meta["num_interventions"],
        z_dim=train_cfg["z_dim"],
        predictor_hidden=train_cfg["predictor_hidden"],
        predictor_layers=train_cfg["predictor_layers"],
        ema_momentum=train_cfg["ema_momentum_start"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, train_cfg, meta, ckpt


def _rebuild_splits(train_cfg: dict) -> tuple:
    """Re-materialize the exact same train/val/test dataloaders as training."""
    data_cfg = DataConfig(
        npz_path=train_cfg["npz_path"],
        manifest_path=train_cfg["manifest_path"],
        intervention_map_path=train_cfg["intervention_map_path"],
        max_seq_len=train_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        val_fraction=train_cfg["val_fraction"],
        test_fraction=train_cfg["test_fraction"],
        seed=train_cfg["seed"],
        num_workers=0,
    )
    return build_dataloaders(data_cfg)


@torch.no_grad()
def _encode_split(model: CausalJEPA, loader, device: torch.device):
    """Encode every patient in a loader -> (s_x, z, person_id)."""
    all_sx, all_z, all_pid = [], [], []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        _, s_x = model.context_encoder(
            batch["context_x"], batch["context_mask"], batch["context_timestamps"],
            src_key_padding_mask=batch["context_padding_mask"],
        )
        all_sx.append(s_x.cpu())
        all_z.append(batch["intervention_z"].cpu())
        all_pid.append(batch["person_id"].cpu())
    return (
        torch.cat(all_sx, dim=0),
        torch.cat(all_z, dim=0),
        torch.cat(all_pid, dim=0),
    )


def _deterministic_uuid(seed_str: str) -> str:
    import hashlib, uuid
    return str(uuid.UUID(hashlib.sha256(seed_str.encode()).hexdigest()[:32]))


# ======================================================================
# Qdrant workflow
# ======================================================================

def build_qdrant_collection(
    cfg: QdrantEvalConfig,
    embeddings: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    pids: dict[str, torch.Tensor],
    intervention_labels: list[str],
    embedding_dim: int,
) -> QdrantClient:
    """Build a fresh local Qdrant collection populated with all embeddings."""
    qdrant_dir = Path(cfg.qdrant_path)
    if cfg.reset and qdrant_dir.exists():
        shutil.rmtree(qdrant_dir)
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(qdrant_dir))

    if client.collection_exists(cfg.collection):
        client.delete_collection(cfg.collection)

    client.create_collection(
        collection_name=cfg.collection,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=128,
            full_scan_threshold=10_000,
        ),
    )

    # Upsert every split
    n_total = sum(v.shape[0] for v in embeddings.values())
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]upserting", total=n_total)

        for split_name, emb in embeddings.items():
            z = labels[split_name]
            pid = pids[split_name]
            # L2-normalize: cosine distance in Qdrant works either way but
            # explicit normalization stabilizes scores across payloads.
            emb_np = torch.nn.functional.normalize(emb, dim=-1).numpy().astype(np.float32)

            points: list[models.PointStruct] = []
            for i in range(emb_np.shape[0]):
                points.append(models.PointStruct(
                    id=_deterministic_uuid(f"{split_name}:{int(pid[i])}"),
                    vector=emb_np[i].tolist(),
                    payload={
                        "patient_id": int(pid[i]),
                        "intervention_z": int(z[i]),
                        "intervention_label": intervention_labels[int(z[i])],
                        "split": split_name,
                    },
                ))

                if len(points) >= cfg.upsert_batch:
                    client.upsert(cfg.collection, points=points)
                    progress.update(task, advance=len(points))
                    points = []

            if points:
                client.upsert(cfg.collection, points=points)
                progress.update(task, advance=len(points))

    return client


def run_retrieval_eval(
    client: QdrantClient,
    cfg: QdrantEvalConfig,
    test_emb: torch.Tensor,
    test_z: torch.Tensor,
    test_pid: torch.Tensor,
    intervention_labels: list[str],
    train_intervention_counts: list[int],
) -> dict:
    """For each test patient, retrieve top-k TRAIN neighbors, compute metrics."""
    max_k = max(cfg.top_ks)
    test_emb_n = torch.nn.functional.normalize(test_emb, dim=-1).numpy().astype(np.float32)

    # Per-k accumulators
    topk_hits = {k: 0 for k in cfg.top_ks}               # top-1 concordance at any k (at least one match)
    topk_all_hit = {k: 0 for k in cfg.top_ks}            # all-k same intervention
    topk_majority = {k: 0 for k in cfg.top_ks}           # majority-vote matches true z
    per_class_topk_correct = {k: Counter() for k in cfg.top_ks}
    per_class_total = Counter()
    sum_top_cos = {k: 0.0 for k in cfg.top_ks}
    latencies_ms: list[float] = []

    examples: list[dict] = []

    # Majority baselines
    majority_class = int(np.argmax(train_intervention_counts))

    # Random-retrieval control: for each test patient, sample k train patients
    # uniformly at random and run the same majority-vote metric. This is the
    # honest "retrieval did nothing useful for intervention" null hypothesis,
    # different from the majority-class baseline (which always predicts the
    # most common class).
    train_z_pool = np.concatenate([
        np.full(train_intervention_counts[c], c, dtype=np.int64)
        for c in range(len(train_intervention_counts))
    ])
    rng = np.random.default_rng(0)
    random_topk_majority = {k: 0 for k in cfg.top_ks}

    filter_train_only = models.Filter(
        must=[models.FieldCondition(key="split", match=models.MatchValue(value="train"))]
    )

    n_test = test_emb_n.shape[0]
    for i in range(n_test):
        query = test_emb_n[i].tolist()
        true_z = int(test_z[i])
        per_class_total[true_z] += 1

        t0 = time.perf_counter()
        hits = client.query_points(
            collection_name=cfg.collection,
            query=query,
            limit=max_k,
            query_filter=filter_train_only,
            with_payload=True,
        ).points
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        neighbor_z = [int(h.payload["intervention_z"]) for h in hits]
        neighbor_scores = [float(h.score) for h in hits]

        # Random control sampling (for *same* test query, same k values)
        for k in cfg.top_ks:
            rand_sample = rng.choice(train_z_pool, size=k, replace=False)
            rand_vote = Counter(rand_sample.tolist()).most_common(1)[0][0]
            if rand_vote == true_z:
                random_topk_majority[k] += 1

        for k in cfg.top_ks:
            ks = neighbor_z[:k]
            if not ks:
                continue
            # any match in top-k
            if true_z in ks:
                topk_hits[k] += 1
            # all same intervention in top-k
            if len(set(ks)) == 1 and ks[0] == true_z:
                topk_all_hit[k] += 1
            # majority vote
            vote = Counter(ks).most_common(1)[0][0]
            if vote == true_z:
                topk_majority[k] += 1
                per_class_topk_correct[k][true_z] += 1
            # mean cosine
            sum_top_cos[k] += float(np.mean(neighbor_scores[:k]))

        # Capture a handful of qualitative examples
        if len(examples) < cfg.num_examples_to_show:
            examples.append({
                "query_patient_id": int(test_pid[i]),
                "true_intervention": intervention_labels[true_z],
                "neighbors": [
                    {
                        "patient_id": int(h.payload["patient_id"]),
                        "intervention": h.payload["intervention_label"],
                        "cosine": float(h.score),
                    }
                    for h in hits[:5]
                ],
            })

    # Assemble metrics
    results = {
        "n_query": n_test,
        "embedding_dim": int(test_emb.shape[1]),
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "median": float(np.median(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
        },
        "majority_baseline_acc": float(
            sum(int(z == majority_class) for z in test_z.tolist()) / max(n_test, 1)
        ),
        "per_k": {},
        "examples": examples,
    }

    for k in cfg.top_ks:
        per_class_acc = {}
        for cls, total in per_class_total.items():
            if total == 0:
                continue
            per_class_acc[intervention_labels[cls]] = float(
                per_class_topk_correct[k][cls] / total
            )
        results["per_k"][str(k)] = {
            "any_hit_rate": topk_hits[k] / max(n_test, 1),
            "all_same_rate": topk_all_hit[k] / max(n_test, 1),
            "majority_vote_acc": topk_majority[k] / max(n_test, 1),
            "random_majority_vote_acc": random_topk_majority[k] / max(n_test, 1),
            "lift_over_random": (
                topk_majority[k] / max(n_test, 1)
                - random_topk_majority[k] / max(n_test, 1)
            ),
            "mean_top_cosine": sum_top_cos[k] / max(n_test, 1),
            "per_class_majority_vote_acc": per_class_acc,
        }

    return results


# ======================================================================
# Reporting
# ======================================================================

def render_report(
    cfg: QdrantEvalConfig,
    ckpt_info: dict,
    intervention_labels: list[str],
    split_sizes: dict,
    retrieval: dict,
    encode_secs: float,
    upsert_secs: float,
) -> list[dict]:
    # ---- Setup panel ----
    setup = Table(title="Qdrant Retrieval Test — Setup", show_lines=False, title_style="bold")
    setup.add_column("Field", style="cyan"); setup.add_column("Value", style="white")
    setup.add_row("Checkpoint", str(cfg.checkpoint))
    setup.add_row("Best val loss / epoch", f"{ckpt_info.get('val_loss', float('nan')):.4f} @ {ckpt_info.get('epoch','?')}")
    setup.add_row("Embedding dim", str(retrieval["embedding_dim"]))
    setup.add_row("Distance", "COSINE (HNSW, m=16, ef_construct=128)")
    setup.add_row("Points indexed", f"train={split_sizes['train']}  val={split_sizes['val']}  test={split_sizes['test']}")
    setup.add_row("Encode time", f"{encode_secs:.2f} s")
    setup.add_row("Upsert time", f"{upsert_secs:.2f} s")
    console.print(setup)

    # ---- Latency ----
    lat = Table(title="Query Latency (test queries, filtered to train-only)",
                show_lines=False, title_style="bold")
    lat.add_column("Stat", style="cyan"); lat.add_column("ms", justify="right")
    for k, v in retrieval["latency_ms"].items():
        lat.add_row(k, f"{v:.2f}")
    console.print(lat)

    # ---- Retrieval metrics ----
    m = Table(title="Top-K Intervention Concordance (test vs train neighbors)",
              show_lines=False, title_style="bold")
    m.add_column("k", justify="right", style="bold")
    m.add_column("any-hit", justify="right")
    m.add_column("JEPA maj-vote", justify="right")
    m.add_column("random maj-vote", justify="right", style="dim")
    m.add_column("lift", justify="right")
    m.add_column("all-same", justify="right")
    m.add_column("mean top-k cos", justify="right")
    for k in cfg.top_ks:
        row = retrieval["per_k"][str(k)]
        lift_pp = row["lift_over_random"] * 100
        lift_txt = (
            f"[green]+{lift_pp:.1f}pp[/]" if lift_pp > 0.5
            else f"[red]{lift_pp:+.1f}pp[/]" if lift_pp < -0.5
            else f"{lift_pp:+.1f}pp"
        )
        m.add_row(
            str(k),
            f"{row['any_hit_rate']*100:.2f}%",
            f"{row['majority_vote_acc']*100:.2f}%",
            f"{row['random_majority_vote_acc']*100:.2f}%",
            lift_txt,
            f"{row['all_same_rate']*100:.2f}%",
            f"{row['mean_top_cosine']:+.4f}",
        )
    console.print(m)

    console.print(
        f"[dim]Always-predict-majority baseline: "
        f"{retrieval['majority_baseline_acc']*100:.2f}% "
        f"(test class prior for most common intervention)[/]"
    )

    # ---- Per-class majority-vote acc at k=5 ----
    p = Table(title="Per-Intervention Majority-Vote Accuracy (k=5)",
              show_lines=False, title_style="bold")
    p.add_column("Intervention", style="cyan")
    p.add_column("accuracy", justify="right")
    for label, acc in retrieval["per_k"]["5"]["per_class_majority_vote_acc"].items():
        p.add_row(label, f"{acc*100:.2f}%")
    console.print(p)

    # ---- Qualitative examples ----
    for ex in retrieval["examples"]:
        e = Table(
            title=f"Example query — patient {ex['query_patient_id']} (true: {ex['true_intervention']})",
            show_lines=False, title_style="bold",
        )
        e.add_column("rank", justify="right", style="dim")
        e.add_column("neighbor patient_id")
        e.add_column("intervention")
        e.add_column("cosine", justify="right")
        for rank, nb in enumerate(ex["neighbors"], 1):
            match_tag = "[green]✓[/]" if nb["intervention"] == ex["true_intervention"] else "[red]x[/]"
            e.add_row(
                f"{rank}",
                str(nb["patient_id"]),
                f"{nb['intervention']} {match_tag}",
                f"{nb['cosine']:+.4f}",
            )
        console.print(e)

    # ---- Verdicts ----
    verdicts: list[dict] = []

    def verdict(name: str, ok: bool, detail: str) -> str:
        tag = "[bold green]GOOD[/]" if ok else "[bold red]CONCERN[/]"
        verdicts.append({"check": name, "ok": bool(ok), "detail": detail})
        return f"{tag}  {name} — {detail}"

    k5 = retrieval["per_k"]["5"]
    k10 = retrieval["per_k"]["10"]

    lines = []
    # --- Patient-twin retrieval (the actual production use-case) ---
    lines.append(verdict(
        "[twin] Top-5 cosine high (patients cluster in embedding space)",
        k5["mean_top_cosine"] > 0.80,
        f"mean top-5 cosine = {k5['mean_top_cosine']:+.3f}",
    ))
    lines.append(verdict(
        "[twin] Top-5 surfaces at least one same-intervention twin",
        k5["any_hit_rate"] > 0.75,
        f"any-match in top-5 = {k5['any_hit_rate']*100:.1f}%",
    ))
    # --- Intervention-label retrieval (harder; encoder never sees z) ---
    lines.append(verdict(
        "[intervention] Top-5 majority-vote lifts over random retrieval",
        k5["lift_over_random"] > 0.02,
        f"JEPA {k5['majority_vote_acc']*100:.1f}% vs random {k5['random_majority_vote_acc']*100:.1f}% "
        f"(lift {k5['lift_over_random']*100:+.1f}pp)",
    ))
    lines.append(verdict(
        "[intervention] Top-10 majority-vote lifts over random retrieval",
        k10["lift_over_random"] > 0.02,
        f"JEPA {k10['majority_vote_acc']*100:.1f}% vs random {k10['random_majority_vote_acc']*100:.1f}% "
        f"(lift {k10['lift_over_random']*100:+.1f}pp)",
    ))
    # --- System-level ---
    lines.append(verdict(
        "Query latency is interactive (filtered scan at 3k points)",
        retrieval["latency_ms"]["p95"] < 200.0,
        f"p95 = {retrieval['latency_ms']['p95']:.1f} ms (full_scan_threshold=10k, so filter => scan)",
    ))

    panel = Panel("\n".join(lines), title="Qdrant retrieval verdicts", border_style="bright_blue")
    console.print(panel)

    return verdicts


def write_markdown(
    path: Path,
    cfg: QdrantEvalConfig,
    ckpt_info: dict,
    split_sizes: dict,
    retrieval: dict,
    verdicts: list[dict],
) -> None:
    lines = [
        "# Qdrant Retrieval Test Report",
        "",
        f"- Checkpoint: `{cfg.checkpoint}` (best val loss {ckpt_info.get('val_loss','?'):.4f} at epoch {ckpt_info.get('epoch','?')})",
        f"- Embedding dim: **{retrieval['embedding_dim']}**",
        f"- Points indexed: train={split_sizes['train']}, val={split_sizes['val']}, test={split_sizes['test']}",
        "",
        "## Latency (ms)",
        "",
        "| stat | ms |",
        "|---|---:|",
    ]
    for k, v in retrieval["latency_ms"].items():
        lines.append(f"| {k} | {v:.2f} |")

    lines += [
        "",
        "## Top-k intervention concordance",
        "",
        f"Majority-class baseline: **{retrieval['majority_baseline_acc']*100:.2f}%**",
        "",
        "| k | any-hit | JEPA maj-vote | random maj-vote | lift | all-same | mean cos |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, row in retrieval["per_k"].items():
        lines.append(
            f"| {k} | {row['any_hit_rate']*100:.2f}% | "
            f"{row['majority_vote_acc']*100:.2f}% | "
            f"{row['random_majority_vote_acc']*100:.2f}% | "
            f"{row['lift_over_random']*100:+.2f}pp | "
            f"{row['all_same_rate']*100:.2f}% | "
            f"{row['mean_top_cosine']:+.4f} |"
        )

    lines += ["", "## Per-intervention majority-vote accuracy (k=5)", ""]
    for label, acc in retrieval["per_k"]["5"]["per_class_majority_vote_acc"].items():
        lines.append(f"- {label}: **{acc*100:.2f}%**")

    lines += ["", "## Verdicts", ""]
    for v in verdicts:
        tag = "GOOD" if v["ok"] else "CONCERN"
        lines.append(f"- **{tag}** — {v['check']}: {v['detail']}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================
# Main
# ======================================================================

def main(cfg: QdrantEvalConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]1. Loading checkpoint")
    model, train_cfg, meta, ckpt = _load_checkpoint(cfg, device)
    ckpt_info = {"epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss", float("nan"))}
    console.print(f"epoch={ckpt_info['epoch']}  val_loss={ckpt_info['val_loss']:.4f}  device={device}")

    console.rule("[bold]2. Rebuilding splits and encoding all patients")
    train_loader, val_loader, test_loader, data_meta = _rebuild_splits(train_cfg)
    t0 = time.time()
    emb_train, z_train, pid_train = _encode_split(model, train_loader, device)
    emb_val,   z_val,   pid_val   = _encode_split(model, val_loader,   device)
    emb_test,  z_test,  pid_test  = _encode_split(model, test_loader,  device)
    encode_secs = time.time() - t0

    embeddings = {"train": emb_train, "val": emb_val, "test": emb_test}
    labels     = {"train": z_train,   "val": z_val,   "test": z_test}
    pids       = {"train": pid_train, "val": pid_val, "test": pid_test}
    console.print(
        f"encoded train={emb_train.shape}  val={emb_val.shape}  test={emb_test.shape}  in {encode_secs:.2f}s"
    )

    console.rule("[bold]3. Building local Qdrant collection and upserting")
    t0 = time.time()
    client = build_qdrant_collection(
        cfg=cfg,
        embeddings=embeddings,
        labels=labels,
        pids=pids,
        intervention_labels=data_meta["intervention_labels"],
        embedding_dim=emb_train.shape[1],
    )
    upsert_secs = time.time() - t0
    info = client.get_collection(cfg.collection)
    console.print(
        f"collection '{cfg.collection}' indexed {info.points_count} points "
        f"(dim={emb_train.shape[1]})  upserted in {upsert_secs:.2f}s"
    )

    console.rule("[bold]4. Running retrieval evaluation (test -> train neighbors)")
    retrieval = run_retrieval_eval(
        client=client,
        cfg=cfg,
        test_emb=emb_test,
        test_z=z_test,
        test_pid=pid_test,
        intervention_labels=data_meta["intervention_labels"],
        train_intervention_counts=data_meta["train_intervention_counts"],
    )

    console.rule("[bold]5. Report")
    verdicts = render_report(
        cfg=cfg,
        ckpt_info=ckpt_info,
        intervention_labels=data_meta["intervention_labels"],
        split_sizes=data_meta["split_sizes"],
        retrieval=retrieval,
        encode_secs=encode_secs,
        upsert_secs=upsert_secs,
    )

    # Save artifacts
    report = {
        "config": asdict(cfg),
        "checkpoint_info": ckpt_info,
        "intervention_labels": data_meta["intervention_labels"],
        "split_sizes": data_meta["split_sizes"],
        "encode_secs": encode_secs,
        "upsert_secs": upsert_secs,
        "retrieval": retrieval,
        "verdicts": verdicts,
    }
    (out_dir / "qdrant_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    write_markdown(out_dir / "qdrant_report.md", cfg, ckpt_info,
                   data_meta["split_sizes"], retrieval, verdicts)

    console.print(Panel(
        f"Collection : [green]{cfg.qdrant_path}/ ({cfg.collection})[/]\n"
        f"JSON       : [green]{out_dir/'qdrant_report.json'}[/]\n"
        f"Markdown   : [green]{out_dir/'qdrant_report.md'}[/]",
        title="Artifacts", border_style="green",
    ))


def parse_args() -> QdrantEvalConfig:
    cfg = QdrantEvalConfig()
    p = argparse.ArgumentParser(description="Qdrant retrieval test for Causal-JEPA embeddings")
    p.add_argument("--checkpoint", type=str, default=cfg.checkpoint)
    p.add_argument("--qdrant-path", type=str, default=cfg.qdrant_path)
    p.add_argument("--collection", type=str, default=cfg.collection)
    p.add_argument("--no-reset", action="store_true", help="keep existing qdrant dir")
    p.add_argument("--top-ks", type=int, nargs="+", default=list(cfg.top_ks))
    p.add_argument("--examples", type=int, default=cfg.num_examples_to_show)
    p.add_argument("--out-dir", type=str, default=cfg.out_dir)
    a = p.parse_args()

    cfg.checkpoint = a.checkpoint
    cfg.qdrant_path = a.qdrant_path
    cfg.collection = a.collection
    cfg.reset = not a.no_reset
    cfg.top_ks = tuple(a.top_ks)
    cfg.num_examples_to_show = a.examples
    cfg.out_dir = a.out_dir
    return cfg


if __name__ == "__main__":
    main(parse_args())
