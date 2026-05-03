"""
Counterfactual Twin Search — the Cohort Compass production flow.

Given a test patient x and a hypothetical intervention z we want to answer:

    "If patient x takes z, which real historical patients lived a post-treatment
     trajectory most similar to the one our predictor projects for x?"

We do this by indexing, for every training patient, the TARGET embedding
``s_y = target_encoder(post_window)`` — i.e., the real clinical path the
patient actually lived after index — together with a payload carrying the
intervention they actually received plus summary outcome statistics
(HbA1c, LDL cholesterol, BP, BMI, etc.) in their post-window.

At query time we encode x -> s_x with the context encoder, then for each z
we run the predictor ``s_y_hat = Predictor(s_x, z)`` and ask Qdrant for the
top-K training patients whose *actual* s_y is closest to that predicted s_y
in cosine space.

Residual retrieval
------------------
Naive averaging of retrieved twins' raw outcomes is biased: neighbours in
latent space are systematically higher or lower than the population for any
given outcome, so the point estimate inherits their selection bias.  We fix
this with *residual retrieval*: for each outcome we hold a tabular ridge
baseline ``\hat y_{\text{base}}(x, z)``, retrieve in latent space as before,
aggregate each twin's **residual** ``y_i - \hat y_{\text{base}}(x_i, z_i)``,
and emit ``\hat y_{\text{base}}(x_{\text{test}}, z) + \overline{\text{residual}}``.
This is the formulation reported in the paper and used by the JEPA\_TWIN
head in ``outcome_eval.py``.

Evaluation
----------
1. Treatment-match rate: when the user asks about z, how often do the top-K
   retrieved twins ACTUALLY take z? Compared to random-retrieval baseline.
   This is the sharpest quality signal: if the predictor's output carries
   intervention-conditional information, queries for z=metformin should
   surface metformin-treated twins more often than chance.

2. Outcome divergence: for the SAME test patient, how different are the
   outcome distributions of twins retrieved under z=metformin vs
   z=atorvastatin vs z=lisinopril? Large divergence => the predictor
   meaningfully conditions on z.

3. Clinical outcome means: for each z, what is the mean HbA1c / LDL / BP /
   BMI across retrieved twins' actual post-window values?
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from rich.table import Table

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "backend"))

from ml.data import DataConfig, build_dataloaders                    # noqa: E402
from ml.jepa_model import CausalJEPA                                  # noqa: E402

try:
    from sklearn.linear_model import Ridge                            # noqa: E402
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

console = Console()


# ======================================================================
# Config
# ======================================================================

# Which raw clinical features to surface in outcome summaries. Indices are
# resolved from feature_vocab.json at runtime so changes to the vocab don't
# break this script.
OUTCOME_FEATURES = [
    ("hba1c",       "HbA1c (%)",         "metformin"),
    ("ldl_chol",    "LDL cholesterol",   "atorvastatin"),
    ("systolic_bp", "Systolic BP (mmHg)","lisinopril"),
    ("diastolic_bp","Diastolic BP",      None),
    ("bmi",         "BMI",               None),
    ("glucose",     "Glucose",           None),
    ("hdl_chol",    "HDL cholesterol",   None),
    ("total_chol",  "Total cholesterol", None),
]


@dataclass
class TwinSearchConfig:
    checkpoint: str = "backend/ml/checkpoints/jepa_best.pt"
    feature_vocab_path: str = "backend/training_data/feature_vocab.json"
    qdrant_path: str = "backend/ml/qdrant_twins"
    collection: str = "aou_jepa_twin_targets"
    top_k: int = 10
    reset: bool = True
    n_eval_patients: int = 200         # aggregate metrics sample size
    n_example_patients: int = 3        # per-true-intervention demo patients
    upsert_batch: int = 256
    out_dir: str = "backend/ml/results"


# ======================================================================
# Helpers
# ======================================================================

def _load_model(ckpt_path: Path, device: torch.device):
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


def _rebuild_splits(train_cfg: dict):
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


def _deterministic_uuid(seed_str: str) -> str:
    import hashlib, uuid
    return str(uuid.UUID(hashlib.sha256(seed_str.encode()).hexdigest()[:32]))


# ----------------------------------------------------------------------
# Raw outcome summaries (operate directly on un-standardized npz values)
# ----------------------------------------------------------------------

def compute_raw_outcomes(
    post_values: np.ndarray,     # (T, F) raw un-standardized
    post_mask: np.ndarray,       # (T, F) 1=observed
    feature_idx: dict[str, int],
) -> dict[str, float | None]:
    """Masked mean for each outcome feature. None if feature never observed."""
    out: dict[str, float | None] = {}
    for key, _, _ in OUTCOME_FEATURES:
        if key not in feature_idx:
            continue
        f = feature_idx[key]
        obs = post_mask[:, f] > 0
        if obs.any():
            out[key] = float(post_values[obs, f].mean())
        else:
            out[key] = None
    return out


def aggregate_outcomes(
    outcomes_list: list[dict[str, float | None]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate across a set of retrieved twins: per-feature mean, std, count."""
    agg: dict[str, dict[str, float | int]] = {}
    for key, _, _ in OUTCOME_FEATURES:
        vals = [o[key] for o in outcomes_list if o.get(key) is not None]
        if vals:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": int(len(vals)),
            }
        else:
            agg[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return agg


# ----------------------------------------------------------------------
# Residual retrieval — per-outcome tabular baseline fit on TRAIN patients
# ----------------------------------------------------------------------

def _pool_pre(values, mask, num_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient mean over observed pre cells + fraction-observed per feature."""
    N = len(values)
    pooled = np.zeros((N, num_features), dtype=np.float32)
    frac = np.zeros((N, num_features), dtype=np.float32)
    for i in range(N):
        v = values[i].astype(np.float32)
        m = mask[i].astype(np.float32)
        counts = m.sum(axis=0)
        sums = (v * m).sum(axis=0)
        pooled[i] = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
        frac[i] = counts / max(m.shape[0], 1)
    return pooled, frac


@dataclass
class ResidualBaseline:
    """Per-outcome Ridge baseline fit on TRAIN patients using pooled pre
    features + one-hot intervention.  Used to de-bias twin retrieval.
    """

    outcome_keys: list[str]
    ridges: dict[str, Ridge]                          # one per outcome
    feat_dim: int
    num_interventions: int
    train_outcome_observed: dict[str, float]          # population mean as fallback

    def predict(self, pooled: np.ndarray, frac: np.ndarray, z: int) -> dict[str, float]:
        """Baseline prediction for a single patient under intervention z."""
        z_oh = np.zeros(self.num_interventions, dtype=np.float32)
        z_oh[z] = 1.0
        x = np.concatenate([pooled, frac, z_oh]).reshape(1, -1)
        out: dict[str, float] = {}
        for k in self.outcome_keys:
            reg = self.ridges.get(k)
            if reg is None:
                out[k] = self.train_outcome_observed.get(k, 0.0)
            else:
                out[k] = float(reg.predict(x)[0])
        return out


def fit_residual_baseline(
    ds_train,
    feature_idx: dict[str, int],
    num_features: int,
    num_interventions: int,
    outcome_keys: list[str] | None = None,
    alpha: float = 1.0,
) -> ResidualBaseline | None:
    """Fit one Ridge per outcome on TRAIN patients (pre-pooled feats + z)."""
    if not _HAS_SKLEARN:
        console.print("[yellow]sklearn not available; residual retrieval disabled[/]")
        return None

    outcome_keys = outcome_keys or [k for k, _, _ in OUTCOME_FEATURES if k in feature_idx]
    rows = list(ds_train.indices) if hasattr(ds_train, "indices") else list(range(len(ds_train)))
    rows_abs = np.array([int(r) for r in rows], dtype=np.int64)

    pre_values = ds_train.pre_values[rows_abs]
    pre_mask = ds_train.pre_mask[rows_abs]
    post_values = ds_train.post_values[rows_abs]
    post_mask = ds_train.post_mask[rows_abs]
    z = ds_train.intervention_z[rows_abs].astype(np.int64)

    pooled, frac = _pool_pre(pre_values, pre_mask, num_features)
    z_oh = np.eye(num_interventions, dtype=np.float32)[z]
    X = np.concatenate([pooled, frac, z_oh], axis=1)

    ridges: dict[str, Ridge] = {}
    pop_mean: dict[str, float] = {}
    for k in outcome_keys:
        f = feature_idx[k]
        obs = post_mask[:, :, f].sum(axis=1) > 0
        if obs.sum() < 20:
            continue
        y = np.zeros(len(rows_abs), dtype=np.float32)
        for i in range(len(rows_abs)):
            m = post_mask[i, :, f].astype(np.float32)
            if m.sum() > 0:
                y[i] = float((post_values[i, :, f] * m).sum() / m.sum())
        reg = Ridge(alpha=alpha)
        reg.fit(X[obs], y[obs])
        ridges[k] = reg
        pop_mean[k] = float(y[obs].mean())

    return ResidualBaseline(
        outcome_keys=outcome_keys,
        ridges=ridges,
        feat_dim=X.shape[1],
        num_interventions=num_interventions,
        train_outcome_observed=pop_mean,
    )


def aggregate_outcomes_residual(
    twin_outcomes: list[dict[str, float | None]],      # raw outcomes per twin
    twin_baselines: list[dict[str, float]],            # baseline predicted per twin
    query_baseline: dict[str, float],                  # baseline for the *query* patient under query_z
) -> dict[str, dict[str, float | int]]:
    """Residual aggregation: mean(y_i - ŷ_base_i) + ŷ_base_query."""
    agg: dict[str, dict[str, float | int]] = {}
    for key, _, _ in OUTCOME_FEATURES:
        residuals = []
        for tw_o, tw_b in zip(twin_outcomes, twin_baselines):
            if tw_o.get(key) is None or key not in tw_b:
                continue
            residuals.append(tw_o[key] - tw_b[key])
        if residuals and key in query_baseline:
            mean_r = float(np.mean(residuals))
            std_r = float(np.std(residuals))
            agg[key] = {
                "mean": float(query_baseline[key] + mean_r),
                "std": std_r,
                "n": int(len(residuals)),
            }
        else:
            agg[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return agg


# ----------------------------------------------------------------------
# Indexing (build the twin library)
# ----------------------------------------------------------------------

@torch.no_grad()
def index_training_targets(
    cfg: TwinSearchConfig,
    model: CausalJEPA,
    train_loader,
    device: torch.device,
    intervention_labels: list[str],
    feature_idx: dict[str, int],
    residual_baseline: ResidualBaseline | None = None,
    num_features: int | None = None,
) -> tuple[QdrantClient, int, int]:
    """Encode every TRAIN patient's post-window with the target encoder and
    upsert the resulting vector plus raw outcome summary to Qdrant."""

    qdrant_dir = Path(cfg.qdrant_path)
    if cfg.reset and qdrant_dir.exists():
        shutil.rmtree(qdrant_dir)
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_dir))

    if client.collection_exists(cfg.collection):
        client.delete_collection(cfg.collection)

    # Probe the embedding dimension off the first batch
    first = next(iter(train_loader))
    first = {k: v.to(device) for k, v in first.items()}
    _, s_y_probe = model.target_encoder(
        first["target_y"], first["target_mask"], first["target_timestamps"],
        src_key_padding_mask=first["target_padding_mask"],
    )
    dim = s_y_probe.shape[-1]

    client.create_collection(
        collection_name=cfg.collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10_000),
    )

    # Raw-dataset access for outcome summaries (bypasses standardization)
    ds = train_loader.dataset
    raw_post_values = ds.post_values
    raw_post_mask = ds.post_mask
    raw_intervention_z = ds.intervention_z
    raw_person_ids = ds.person_ids

    buf_points: list[models.PointStruct] = []
    total = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]indexing train post-targets", total=len(ds))

        batch_iter = iter(train_loader)
        # Rewind: include the first (already-encoded) batch without re-encoding.
        batches = [first]
        for b in batch_iter:
            batches.append({k: v.to(device) for k, v in b.items()})

        s_y_chunks = [s_y_probe]
        for b in batches[1:]:
            _, sy = model.target_encoder(
                b["target_y"], b["target_mask"], b["target_timestamps"],
                src_key_padding_mask=b["target_padding_mask"],
            )
            s_y_chunks.append(sy)
        s_y_all = torch.cat(s_y_chunks, dim=0).cpu()
        person_id_all = torch.cat([b["person_id"].cpu() for b in batches], dim=0)

        # Map person_id -> dataset row index (train split only)
        ds_pid_to_row = {int(pid): row for row, pid in enumerate(raw_person_ids[ds.indices])}

        s_y_norm = torch.nn.functional.normalize(s_y_all, dim=-1).numpy().astype(np.float32)

        for i in range(s_y_norm.shape[0]):
            pid = int(person_id_all[i])
            row = ds_pid_to_row[pid]
            row_abs = int(ds.indices[row])
            z = int(raw_intervention_z[row_abs])
            outcomes = compute_raw_outcomes(
                raw_post_values[row_abs], raw_post_mask[row_abs], feature_idx
            )
            payload = {
                "patient_id": pid,
                "intervention_z": z,
                "intervention_label": intervention_labels[z],
                "n_post_obs": int(raw_post_mask[row_abs].sum()),
                "outcomes": outcomes,
            }
            if residual_baseline is not None and num_features is not None:
                pooled_i, frac_i = _pool_pre(
                    ds.pre_values[row_abs:row_abs+1],
                    ds.pre_mask[row_abs:row_abs+1],
                    num_features,
                )
                payload["pre_pooled"] = pooled_i[0].tolist()
                payload["pre_frac"] = frac_i[0].tolist()
                payload["baseline_under_actual_z"] = residual_baseline.predict(
                    pooled_i[0], frac_i[0], z,
                )
            buf_points.append(models.PointStruct(
                id=_deterministic_uuid(f"train:{pid}"),
                vector=s_y_norm[i].tolist(),
                payload=payload,
            ))

            if len(buf_points) >= cfg.upsert_batch:
                client.upsert(cfg.collection, points=buf_points)
                total += len(buf_points)
                progress.update(task, advance=len(buf_points))
                buf_points = []

        if buf_points:
            client.upsert(cfg.collection, points=buf_points)
            total += len(buf_points)
            progress.update(task, advance=len(buf_points))

    return client, total, dim


# ----------------------------------------------------------------------
# Counterfactual querying
# ----------------------------------------------------------------------

@torch.no_grad()
def counterfactual_query(
    model: CausalJEPA,
    client: QdrantClient,
    collection: str,
    s_x: torch.Tensor,           # (D,) single patient context embedding
    z: int,
    top_k: int,
    device: torch.device,
) -> list[dict]:
    """Predict ŝ_y = Predictor(s_x, z) and retrieve top-K real post-states."""
    z_tensor = torch.tensor([z], dtype=torch.long, device=device)
    s_y_hat = model.predict_counterfactual(s_x.unsqueeze(0).to(device), z_tensor)
    query_vec = torch.nn.functional.normalize(s_y_hat, dim=-1).cpu().numpy()[0].astype(np.float32)

    hits = client.query_points(
        collection_name=collection,
        query=query_vec.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    out = []
    for h in hits:
        row = {
            "patient_id": int(h.payload["patient_id"]),
            "intervention_z": int(h.payload["intervention_z"]),
            "intervention_label": h.payload["intervention_label"],
            "cosine": float(h.score),
            "outcomes": h.payload["outcomes"],
            "n_post_obs": int(h.payload["n_post_obs"]),
        }
        if "baseline_under_actual_z" in h.payload:
            row["baseline_under_actual_z"] = h.payload["baseline_under_actual_z"]
        out.append(row)
    return out


def residual_counterfactual_aggregate(
    twins: list[dict],
    query_pre_pooled: np.ndarray,
    query_pre_frac: np.ndarray,
    query_z: int,
    baseline: ResidualBaseline,
) -> dict[str, dict[str, float | int]]:
    """Apply residual retrieval to an already-retrieved list of twins.

    For every twin we already have ``outcomes`` (y_i) and
    ``baseline_under_actual_z`` (ŷ_base_i).  For the *query* patient we
    compute ``ŷ_base(x_query, z=query_z)`` and emit
        ŷ_query = ŷ_base(x_query, z=query_z) + mean_i (y_i - ŷ_base_i).
    """
    query_baseline = baseline.predict(query_pre_pooled, query_pre_frac, query_z)
    twin_outcomes = [t["outcomes"] for t in twins]
    twin_baselines = [t.get("baseline_under_actual_z", {}) for t in twins]
    return aggregate_outcomes_residual(twin_outcomes, twin_baselines, query_baseline)


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def _encode_test(model: CausalJEPA, test_loader, device: torch.device):
    all_sx, all_z, all_pid = [], [], []
    for batch in test_loader:
        b = {k: v.to(device) for k, v in batch.items()}
        _, s_x = model.context_encoder(
            b["context_x"], b["context_mask"], b["context_timestamps"],
            src_key_padding_mask=b["context_padding_mask"],
        )
        all_sx.append(s_x.cpu())
        all_z.append(batch["intervention_z"])
        all_pid.append(batch["person_id"])
    return torch.cat(all_sx), torch.cat(all_z), torch.cat(all_pid)


def run_aggregate_eval(
    model: CausalJEPA,
    client: QdrantClient,
    cfg: TwinSearchConfig,
    test_sx: torch.Tensor,
    test_z: torch.Tensor,
    test_pid: torch.Tensor,
    intervention_labels: list[str],
    train_intervention_counts: list[int],
    feature_idx: dict[str, int],
    device: torch.device,
) -> dict:
    """For each test patient, run counterfactual queries for every z and
    aggregate treatment-match rate, outcome divergence, latency."""

    K = cfg.top_k
    n_z = len(intervention_labels)
    rng = np.random.default_rng(0)

    # Subsample test set for runtime; stratified by true intervention.
    # (Aggregate metrics are not sensitive to which test patients we use.)
    selected_idx: list[int] = []
    per_true_target = cfg.n_eval_patients // n_z
    for true_z in range(n_z):
        rows = np.where(test_z.numpy() == true_z)[0]
        if len(rows) == 0:
            continue
        rng.shuffle(rows)
        selected_idx.extend(rows[:per_true_target].tolist())
    selected_idx = sorted(set(selected_idx))
    n_eval = len(selected_idx)

    # Training label pool for random-retrieval baseline
    train_z_pool = np.concatenate([
        np.full(train_intervention_counts[c], c, dtype=np.int64)
        for c in range(n_z)
    ])

    # Per-(query z) accumulators
    treat_match_counts = {z: 0 for z in range(n_z)}         # # of twin hits where twin.z == query_z
    treat_match_total = {z: 0 for z in range(n_z)}          # total twins evaluated for that query_z
    outcome_sums = {z: {k: [] for k, _, _ in OUTCOME_FEATURES} for z in range(n_z)}
    random_match_counts = {z: 0 for z in range(n_z)}
    random_match_total = {z: 0 for z in range(n_z)}
    latencies_ms: list[float] = []

    # Per-test-patient outcome-divergence accumulator
    divergence_abs: list[dict[str, float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]counterfactual queries ({n_eval} patients x {n_z} z)",
            total=n_eval * n_z,
        )

        for row in selected_idx:
            per_z_outcomes: dict[int, dict[str, float]] = {}
            for z_query in range(n_z):
                t0 = time.perf_counter()
                twins = counterfactual_query(
                    model, client, cfg.collection,
                    test_sx[row], z_query, K, device,
                )
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                # Treatment-match rate
                for t in twins:
                    treat_match_total[z_query] += 1
                    if t["intervention_z"] == z_query:
                        treat_match_counts[z_query] += 1

                # Random baseline (same K draws, same query_z label)
                rand_draws = rng.choice(train_z_pool, size=K, replace=False)
                random_match_total[z_query] += K
                random_match_counts[z_query] += int((rand_draws == z_query).sum())

                # Outcome aggregates
                agg = aggregate_outcomes([t["outcomes"] for t in twins])
                for k in outcome_sums[z_query]:
                    if agg[k]["n"] > 0:
                        outcome_sums[z_query][k].append(agg[k]["mean"])

                # Store mean outcome per-z for divergence
                per_z_outcomes[z_query] = {
                    k: agg[k]["mean"] for k in agg if not np.isnan(agg[k]["mean"])
                }

                progress.update(task, advance=1)

            # Outcome divergence for this test patient: avg absolute pairwise
            # difference across z queries.
            d = {}
            for key, _, _ in OUTCOME_FEATURES:
                vals = [per_z_outcomes[z].get(key) for z in range(n_z)]
                vals = [v for v in vals if v is not None and not np.isnan(v)]
                if len(vals) >= 2:
                    pairs = [abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]]
                    d[key] = float(np.mean(pairs)) if pairs else 0.0
            divergence_abs.append(d)

    # ---- Assemble results ----
    per_z = {}
    for z_query in range(n_z):
        # Treatment match
        tm_rate = (
            treat_match_counts[z_query] / max(treat_match_total[z_query], 1)
        )
        rm_rate = (
            random_match_counts[z_query] / max(random_match_total[z_query], 1)
        )
        # Outcome mean summary
        out = {}
        for k, label, _assoc in OUTCOME_FEATURES:
            vals = outcome_sums[z_query][k]
            if vals:
                out[k] = {
                    "display": label,
                    "mean_of_top_k_means": float(np.mean(vals)),
                    "std_of_top_k_means": float(np.std(vals)),
                    "n_patients": int(len(vals)),
                }
        per_z[intervention_labels[z_query]] = {
            "z": z_query,
            "treatment_match_rate": tm_rate,
            "random_treatment_match_rate": rm_rate,
            "lift_over_random": tm_rate - rm_rate,
            "outcomes": out,
        }

    # Outcome divergence (averaged across test patients)
    mean_divergence = {}
    for key, label, _ in OUTCOME_FEATURES:
        vals = [d[key] for d in divergence_abs if key in d]
        if vals:
            mean_divergence[key] = {
                "display": label,
                "mean_abs_diff_across_z": float(np.mean(vals)),
                "n_patients": int(len(vals)),
            }

    results = {
        "n_eval_patients": n_eval,
        "top_k": K,
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "median": float(np.median(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
        },
        "per_z": per_z,
        "outcome_divergence_across_z": mean_divergence,
    }
    return results, selected_idx


def build_demo_examples(
    model: CausalJEPA,
    client: QdrantClient,
    cfg: TwinSearchConfig,
    test_sx: torch.Tensor,
    test_z: torch.Tensor,
    test_pid: torch.Tensor,
    test_post_outcomes: list[dict],
    intervention_labels: list[str],
    feature_idx: dict[str, int],
    device: torch.device,
    rng: np.random.Generator,
) -> list[dict]:
    """Pick one test patient per true intervention and produce a 3-z demo."""
    demos: list[dict] = []
    for true_z in range(len(intervention_labels)):
        rows = np.where(test_z.numpy() == true_z)[0]
        if len(rows) == 0:
            continue
        # Prefer patients with richer post observations (more outcomes to show)
        rng.shuffle(rows)
        # Use the first row with at least 2 observed outcome fields
        chosen_row = None
        for r in rows:
            o = test_post_outcomes[r]
            present = sum(1 for k, _, _ in OUTCOME_FEATURES if o.get(k) is not None)
            if present >= 3:
                chosen_row = int(r)
                break
        if chosen_row is None:
            chosen_row = int(rows[0])

        patient_id = int(test_pid[chosen_row])
        demo = {
            "patient_id": patient_id,
            "true_intervention": intervention_labels[true_z],
            "actual_post_outcomes": test_post_outcomes[chosen_row],
            "per_z": {},
        }
        for z_query in range(len(intervention_labels)):
            twins = counterfactual_query(
                model, client, cfg.collection,
                test_sx[chosen_row], z_query, cfg.top_k, device,
            )
            demo["per_z"][intervention_labels[z_query]] = {
                "z": z_query,
                "twins": twins,
                "aggregate_outcomes": aggregate_outcomes([t["outcomes"] for t in twins]),
                "twin_treatment_distribution": Counter(
                    t["intervention_label"] for t in twins
                ),
            }
        demos.append(demo)
        if len(demos) >= cfg.n_example_patients:
            break
    return demos


# ======================================================================
# Reporting
# ======================================================================

def render_report(
    cfg: TwinSearchConfig,
    intervention_labels: list[str],
    n_indexed: int,
    dim: int,
    ckpt_info: dict,
    results: dict,
    demos: list[dict],
    index_secs: float,
) -> list[dict]:
    # Setup
    setup = Table(title="Counterfactual Twin Search — Setup",
                  show_lines=False, title_style="bold")
    setup.add_column("Field", style="cyan"); setup.add_column("Value", style="white")
    setup.add_row("Checkpoint", cfg.checkpoint)
    setup.add_row("Best val loss / epoch",
                  f"{ckpt_info.get('val_loss', float('nan')):.4f} @ {ckpt_info.get('epoch','?')}")
    setup.add_row("Twin library", f"{n_indexed:,} train patients indexed (dim={dim})")
    setup.add_row("Top-K", str(cfg.top_k))
    setup.add_row("Test patients evaluated", str(results["n_eval_patients"]))
    setup.add_row("Index build time", f"{index_secs:.1f}s")
    console.print(setup)

    # Latency
    lat = Table(title="Query Latency (ŝ_y -> top-K cosine retrieval)",
                show_lines=False, title_style="bold")
    lat.add_column("Stat", style="cyan"); lat.add_column("ms", justify="right")
    for k, v in results["latency_ms"].items():
        lat.add_row(k, f"{v:.2f}")
    console.print(lat)

    # Treatment-match table (THE core metric)
    tm = Table(
        title=f"Treatment-Match Rate in Top-{cfg.top_k} Twins (does querying for z actually surface z-treated twins?)",
        show_lines=False, title_style="bold",
    )
    tm.add_column("Query z", style="cyan")
    tm.add_column("JEPA match rate", justify="right")
    tm.add_column("Random baseline", justify="right", style="dim")
    tm.add_column("Lift", justify="right")
    for label, row in results["per_z"].items():
        lift_pp = row["lift_over_random"] * 100
        lift_txt = (
            f"[green]+{lift_pp:.1f}pp[/]" if lift_pp > 1.0
            else f"[red]{lift_pp:+.1f}pp[/]" if lift_pp < -1.0
            else f"{lift_pp:+.1f}pp"
        )
        tm.add_row(
            label,
            f"{row['treatment_match_rate']*100:.2f}%",
            f"{row['random_treatment_match_rate']*100:.2f}%",
            lift_txt,
        )
    console.print(tm)

    # Outcome distribution per query z
    for label, row in results["per_z"].items():
        o = Table(
            title=f"Mean clinical outcomes across top-{cfg.top_k} twins when query z = {label}",
            show_lines=False, title_style="bold",
        )
        o.add_column("Outcome", style="cyan")
        o.add_column("mean", justify="right")
        o.add_column("std (across patients)", justify="right")
        o.add_column("n test patients", justify="right")
        for key, _, _ in OUTCOME_FEATURES:
            if key in row["outcomes"]:
                s = row["outcomes"][key]
                o.add_row(
                    s["display"],
                    f"{s['mean_of_top_k_means']:.2f}",
                    f"{s['std_of_top_k_means']:.2f}",
                    str(s["n_patients"]),
                )
        console.print(o)

    # Outcome divergence across z
    d = Table(
        title="Outcome Divergence Across z Queries (mean |Δ| between twin cohorts for the same test patient)",
        show_lines=False, title_style="bold",
    )
    d.add_column("Outcome", style="cyan")
    d.add_column("mean |pairwise Δ|", justify="right")
    d.add_column("n test patients", justify="right")
    for key, _, _ in OUTCOME_FEATURES:
        if key in results["outcome_divergence_across_z"]:
            s = results["outcome_divergence_across_z"][key]
            d.add_row(s["display"], f"{s['mean_abs_diff_across_z']:.3f}", str(s["n_patients"]))
    console.print(d)

    # Demo examples
    for demo in demos:
        console.rule(
            f"[bold magenta]Demo — patient {demo['patient_id']} "
            f"(true intervention: {demo['true_intervention']})"
        )

        # Actual post-window outcomes
        ap = Table(title="Actual post-window outcomes for this patient",
                   show_lines=False, title_style="bold")
        ap.add_column("Outcome", style="cyan"); ap.add_column("value", justify="right")
        for key, label, _ in OUTCOME_FEATURES:
            v = demo["actual_post_outcomes"].get(key)
            ap.add_row(label, f"{v:.2f}" if v is not None else "—")
        console.print(ap)

        for z_label, payload in demo["per_z"].items():
            # Outcome aggregate under this hypothetical z
            tbl = Table(
                title=f"IF patient takes {z_label} -> top-{cfg.top_k} twins' actual outcomes",
                show_lines=False, title_style="bold",
            )
            tbl.add_column("Outcome", style="cyan")
            tbl.add_column("mean", justify="right")
            tbl.add_column("std", justify="right")
            tbl.add_column("n twins with obs", justify="right")
            for key, display, _ in OUTCOME_FEATURES:
                s = payload["aggregate_outcomes"][key]
                if s["n"] > 0:
                    tbl.add_row(display, f"{s['mean']:.2f}", f"{s['std']:.2f}", str(s["n"]))
                else:
                    tbl.add_row(display, "—", "—", "0")
            console.print(tbl)

            # Twin treatment distribution
            dist = payload["twin_treatment_distribution"]
            dist_txt = "  ".join(
                f"{lbl}:{dist.get(lbl,0)}" for lbl in intervention_labels
            )
            console.print(
                f"[dim]Top-{cfg.top_k} twin treatment mix: {dist_txt}  "
                f"(we asked for {z_label}; ideal: most neighbors are {z_label})[/]"
            )

            # Top-5 twin listing
            tw = Table(
                title=f"Top-5 counterfactual twins for z = {z_label}",
                show_lines=False, title_style="bold",
            )
            tw.add_column("rank", style="dim", justify="right")
            tw.add_column("patient_id")
            tw.add_column("actually took")
            tw.add_column("cosine", justify="right")
            tw.add_column("HbA1c", justify="right")
            tw.add_column("LDL", justify="right")
            tw.add_column("sBP", justify="right")
            tw.add_column("BMI", justify="right")
            for rank, t in enumerate(payload["twins"][:5], 1):
                match_tag = ("[green]✓[/]" if t["intervention_label"] == z_label
                             else "[dim]x[/]")
                o = t["outcomes"]
                tw.add_row(
                    str(rank),
                    str(t["patient_id"]),
                    f"{t['intervention_label']} {match_tag}",
                    f"{t['cosine']:+.4f}",
                    f"{o['hba1c']:.2f}"       if o.get("hba1c")       is not None else "—",
                    f"{o['ldl_chol']:.1f}"    if o.get("ldl_chol")    is not None else "—",
                    f"{o['systolic_bp']:.1f}" if o.get("systolic_bp") is not None else "—",
                    f"{o['bmi']:.1f}"         if o.get("bmi")         is not None else "—",
                )
            console.print(tw)

    # ---- Verdicts ----
    verdicts: list[dict] = []
    lines = []

    def verdict(name: str, ok: bool, detail: str) -> str:
        tag = "[bold green]GOOD[/]" if ok else "[bold red]CONCERN[/]"
        verdicts.append({"check": name, "ok": bool(ok), "detail": detail})
        return f"{tag}  {name} — {detail}"

    # Treatment-match lift per z
    any_z_good_lift = False
    for label, row in results["per_z"].items():
        lift = row["lift_over_random"]
        ok = lift > 0.02
        any_z_good_lift = any_z_good_lift or ok
        lines.append(verdict(
            f"Query z={label} surfaces z-treated twins above random",
            ok,
            f"JEPA {row['treatment_match_rate']*100:.1f}% vs random "
            f"{row['random_treatment_match_rate']*100:.1f}% (lift {lift*100:+.1f}pp)",
        ))

    # Outcome divergence on at least one clinically meaningful endpoint
    divergence = results["outcome_divergence_across_z"]
    keys_of_interest = ["hba1c", "ldl_chol", "systolic_bp"]
    any_meaningful_divergence = False
    for k in keys_of_interest:
        if k in divergence:
            d = divergence[k]
            display = d["display"]
            # Clinically interesting: HbA1c >0.05%, LDL >2, sBP >1 mmHg, etc.
            thresh = {"hba1c": 0.05, "ldl_chol": 2.0, "systolic_bp": 1.0}[k]
            ok = d["mean_abs_diff_across_z"] > thresh
            any_meaningful_divergence = any_meaningful_divergence or ok
            lines.append(verdict(
                f"Predictor output for different z yields different {display}",
                ok,
                f"mean |Δ| across z = {d['mean_abs_diff_across_z']:.2f} "
                f"(need > {thresh} to be clinically noticeable)",
            ))

    lines.append(verdict(
        "Query latency interactive",
        results["latency_ms"]["p95"] < 200.0,
        f"p95 = {results['latency_ms']['p95']:.1f} ms",
    ))

    panel = Panel(
        "\n".join(lines),
        title="Counterfactual twin-search verdicts",
        border_style="bright_blue",
    )
    console.print(panel)

    return verdicts


def write_markdown(
    path: Path,
    cfg: TwinSearchConfig,
    ckpt_info: dict,
    n_indexed: int,
    dim: int,
    results: dict,
    verdicts: list[dict],
    demos: list[dict],
    intervention_labels: list[str],
) -> None:
    lines = [
        "# Counterfactual Twin Search Report",
        "",
        f"- Checkpoint: `{cfg.checkpoint}` (val loss {ckpt_info.get('val_loss','?'):.4f} at epoch {ckpt_info.get('epoch','?')})",
        f"- Twin library: **{n_indexed:,} training patients** indexed in Qdrant (dim={dim})",
        f"- Top-K: **{cfg.top_k}**",
        f"- Test patients evaluated: **{results['n_eval_patients']}**",
        "",
        "## Query latency (ms)",
        "",
        "| stat | ms |",
        "|---|---:|",
    ]
    for k, v in results["latency_ms"].items():
        lines.append(f"| {k} | {v:.2f} |")

    lines += [
        "",
        "## Treatment-match rate (top-K twins that actually received query z)",
        "",
        "| Query z | JEPA | random | lift |",
        "|---|---:|---:|---:|",
    ]
    for label, row in results["per_z"].items():
        lines.append(
            f"| {label} | {row['treatment_match_rate']*100:.2f}% | "
            f"{row['random_treatment_match_rate']*100:.2f}% | "
            f"{row['lift_over_random']*100:+.2f}pp |"
        )

    lines += ["", "## Outcome divergence across z queries", ""]
    for key, _, _ in OUTCOME_FEATURES:
        if key in results["outcome_divergence_across_z"]:
            s = results["outcome_divergence_across_z"][key]
            lines.append(
                f"- {s['display']}: mean |Δ across z queries| = "
                f"**{s['mean_abs_diff_across_z']:.3f}** (n={s['n_patients']})"
            )

    lines += ["", "## Verdicts", ""]
    for v in verdicts:
        tag = "GOOD" if v["ok"] else "CONCERN"
        lines.append(f"- **{tag}** — {v['check']}: {v['detail']}")

    # Demos
    for demo in demos:
        lines += [
            "",
            f"## Demo — patient {demo['patient_id']} (true intervention: {demo['true_intervention']})",
            "",
            "Actual post-window outcomes:",
        ]
        for key, label, _ in OUTCOME_FEATURES:
            v = demo["actual_post_outcomes"].get(key)
            lines.append(f"- {label}: {f'{v:.2f}' if v is not None else '—'}")

        for z_label, payload in demo["per_z"].items():
            lines += [
                "",
                f"### Counterfactual: if patient takes {z_label}",
                "",
                f"Twin treatment mix: " + ", ".join(
                    f"{lbl}={payload['twin_treatment_distribution'].get(lbl,0)}"
                    for lbl in intervention_labels
                ),
                "",
                f"Top-5 twins:",
                "",
                "| rank | patient_id | actually took | cos | HbA1c | LDL | sBP | BMI |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
            def fmt(val, prec):
                return f"{val:.{prec}f}" if val is not None else "—"
            for rank, t in enumerate(payload["twins"][:5], 1):
                o = t["outcomes"]
                match = "✓" if t["intervention_label"] == z_label else "x"
                hba1c = fmt(o.get("hba1c"), 2)
                ldl = fmt(o.get("ldl_chol"), 1)
                sbp = fmt(o.get("systolic_bp"), 1)
                bmi = fmt(o.get("bmi"), 1)
                lines.append(
                    f"| {rank} | {t['patient_id']} | {t['intervention_label']} {match} | "
                    f"{t['cosine']:+.4f} | {hba1c} | {ldl} | {sbp} | {bmi} |"
                )

    path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================
# Main
# ======================================================================

def main(cfg: TwinSearchConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Feature index from vocab
    feature_idx = json.loads(Path(cfg.feature_vocab_path).read_text())

    console.rule("[bold]1. Loading JEPA checkpoint")
    model, train_cfg, meta, ckpt = _load_model(Path(cfg.checkpoint), device)
    ckpt_info = {"epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss", float("nan"))}
    console.print(f"epoch={ckpt_info['epoch']}  val_loss={ckpt_info['val_loss']:.4f}  device={device}")

    console.rule("[bold]2. Rebuilding splits")
    train_loader, val_loader, test_loader, data_meta = _rebuild_splits(train_cfg)
    console.print(
        f"train={data_meta['split_sizes']['train']}  "
        f"val={data_meta['split_sizes']['val']}  test={data_meta['split_sizes']['test']}"
    )

    console.rule("[bold]3a. Fitting residual-retrieval baseline (per-outcome Ridge on TRAIN)")
    residual_baseline = fit_residual_baseline(
        train_loader.dataset,
        feature_idx,
        num_features=int(data_meta["num_features"]),
        num_interventions=len(data_meta["intervention_labels"]),
    )
    if residual_baseline is not None:
        console.print(f"residual baseline fitted for {len(residual_baseline.ridges)} outcomes")

    console.rule("[bold]3b. Indexing training post-targets into Qdrant")
    t0 = time.time()
    client, n_indexed, dim = index_training_targets(
        cfg, model, train_loader, device,
        data_meta["intervention_labels"], feature_idx,
        residual_baseline=residual_baseline,
        num_features=int(data_meta["num_features"]),
    )
    index_secs = time.time() - t0
    console.print(f"indexed {n_indexed:,} patients with dim={dim} in {index_secs:.1f}s")

    console.rule("[bold]4. Encoding test patients and precomputing their actual outcomes")
    test_sx, test_z, test_pid = _encode_test(model, test_loader, device)
    # Raw test outcomes aligned to test_loader.dataset.indices
    ds_test = test_loader.dataset
    test_post_outcomes: list[dict] = []
    for row in range(len(ds_test.indices)):
        row_abs = int(ds_test.indices[row])
        test_post_outcomes.append(compute_raw_outcomes(
            ds_test.post_values[row_abs],
            ds_test.post_mask[row_abs],
            feature_idx,
        ))

    console.rule("[bold]5. Aggregate evaluation (all test patients x all z)")
    results, _selected = run_aggregate_eval(
        model, client, cfg,
        test_sx, test_z, test_pid,
        data_meta["intervention_labels"],
        data_meta["train_intervention_counts"],
        feature_idx, device,
    )

    console.rule("[bold]6. Building demo examples")
    demos = build_demo_examples(
        model, client, cfg, test_sx, test_z, test_pid,
        test_post_outcomes, data_meta["intervention_labels"],
        feature_idx, device, np.random.default_rng(42),
    )

    console.rule("[bold]7. Report")
    verdicts = render_report(
        cfg, data_meta["intervention_labels"], n_indexed, dim,
        ckpt_info, results, demos, index_secs,
    )

    # Save
    report = {
        "config": asdict(cfg),
        "checkpoint_info": ckpt_info,
        "intervention_labels": data_meta["intervention_labels"],
        "n_indexed": n_indexed,
        "embedding_dim": dim,
        "index_secs": index_secs,
        "results": results,
        "verdicts": verdicts,
        "demos": [
            {
                "patient_id": d["patient_id"],
                "true_intervention": d["true_intervention"],
                "actual_post_outcomes": d["actual_post_outcomes"],
                "per_z": {
                    zl: {
                        "z": zp["z"],
                        "aggregate_outcomes": zp["aggregate_outcomes"],
                        "twin_treatment_distribution": dict(zp["twin_treatment_distribution"]),
                        "top_twins": zp["twins"][:5],
                    } for zl, zp in d["per_z"].items()
                },
            } for d in demos
        ],
    }
    (out_dir / "twin_search_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    write_markdown(
        out_dir / "twin_search_report.md", cfg, ckpt_info, n_indexed, dim,
        results, verdicts, demos, data_meta["intervention_labels"],
    )
    console.print(Panel(
        f"Collection : [green]{cfg.qdrant_path}/ ({cfg.collection})[/]\n"
        f"JSON       : [green]{out_dir/'twin_search_report.json'}[/]\n"
        f"Markdown   : [green]{out_dir/'twin_search_report.md'}[/]",
        title="Artifacts", border_style="green",
    ))


def parse_args() -> TwinSearchConfig:
    cfg = TwinSearchConfig()
    p = argparse.ArgumentParser(description="Counterfactual twin search using Qdrant + Health-JEPA (with residual retrieval)")
    p.add_argument("--checkpoint", type=str, default=cfg.checkpoint)
    p.add_argument("--qdrant-path", type=str, default=cfg.qdrant_path)
    p.add_argument("--collection", type=str, default=cfg.collection)
    p.add_argument("--top-k", type=int, default=cfg.top_k)
    p.add_argument("--n-eval-patients", type=int, default=cfg.n_eval_patients)
    p.add_argument("--n-example-patients", type=int, default=cfg.n_example_patients)
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--out-dir", type=str, default=cfg.out_dir)
    a = p.parse_args()
    cfg.checkpoint = a.checkpoint
    cfg.qdrant_path = a.qdrant_path
    cfg.collection = a.collection
    cfg.top_k = a.top_k
    cfg.n_eval_patients = a.n_eval_patients
    cfg.n_example_patients = a.n_example_patients
    cfg.reset = not a.no_reset
    cfg.out_dir = a.out_dir
    return cfg


if __name__ == "__main__":
    main(parse_args())
