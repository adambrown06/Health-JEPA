"""
Downstream clinical-outcome prediction in *real physical units*.

This is the paper's headline utility benchmark.  Given a patient's pre-window
clinical time-series and the intervention they received, predict the mean value
of each target biomarker in the post-window (e.g., HbA1c, LDL, systolic BP).

Why this matters
----------------
The Health-JEPA objective is MSE in the learned representation space, which is
not directly interpretable to clinicians.  A latent MSE of 0.7 means nothing on
its own.  What matters in practice is: "If my patient takes atorvastatin, what
will their LDL be in a year?"  This script evaluates exactly that, in the
native units of each biomarker, and compares every sensible baseline head-to-
head on the same held-out patients.

Comparisons (all predict the *same* per-patient outcome scalars):

    Trivial
      (1) POP_MEAN       — predict the training-set mean outcome
      (2) NO_CHANGE      — predict the patient's own pre-window mean

    Tabular baselines on mean-pooled pre features + one-hot z
      (3) RIDGE
      (4) GBT            — sklearn GradientBoostingRegressor
      (5) MLP_RAW        — sklearn MLPRegressor

    Sequence baselines (end-to-end; own parameters; directly optimize MAE)
      (6) GRU            — GRU over (value, mask, time) + z → outcome head
      (7) TSENCODER      — our own TimeSeriesEncoder + z → outcome head
                           (a faithful "same backbone, different objective"
                            competitor, so the comparison is architecture-
                            controlled)

    JEPA-based readouts (no task-specific fine-tuning)
      (8) JEPA+RIDGE     — Ridge(s_x ⊕ one-hot z) → outcome   (encoder-only)
      (9) JEPA+RIDGE_PRED— Ridge(ŝ_y(s_x, z)) → outcome       (encoder+predictor)
      (10) JEPA+TWIN     — *residual retrieval*: retrieve top-K training
                           twins by cos-sim in s_y space of ŝ_y; aggregate
                           each twin's residual (y_i − ŷ_base(x_i, z_i))
                           against a per-outcome Ridge baseline, and emit
                           ŷ_base(x_test, z_test) + mean(residuals).  This
                           removes the systematic selection bias that was
                           present in the previous naive-mean implementation
                           (where the retrieved cohort was not a population
                           sample) and is the formulation reported in the
                           paper.  Setting ``--naive-twins`` recovers the
                           pre-residual aggregation for comparison.

    Propensity Score Matching (PSM)
    -------------------------------
    Observational EHR data suffer from prescriber confounding: patients who
    receive metformin vs. lisinopril differ in measured *and* unmeasured
    baseline characteristics.  To give a confounding-controlled read of
    every model, we additionally fit a multinomial logistic regression on
    the mean-pooled pre-window features to predict the assigned drug
    (the "propensity model"), then construct a *propensity-matched*
    subcohort by 1:1 nearest-neighbour matching on logit(propensity) with a
    caliper of 0.2 σ.  All per-outcome MAE numbers are recomputed on this
    matched subcohort in addition to the full test set, and we report both
    side-by-side so readers can see how much of any effect is explained by
    treatment-selection imbalance.

Metrics
-------
Per-outcome MAE, RMSE, R², Pearson r  (computed only on test patients whose
post-window contains at least one observation of that outcome).  Also a
*macro* summary averaged across outcomes where ≥30 test patients are observed.

Pairwise paired bootstraps (test-patient resampling, 1000 draws) compare each
model to the strongest baseline to assign a confidence-interval to Δ MAE.

Usage
-----
    python -m ml.outcome_eval \
        --checkpoint ml/checkpoints/jepa_best.pt \
        --tag full \
        --outcomes hba1c ldl_chol systolic_bp diastolic_bp bmi

The script writes:
    ml/results/outcome_report_<tag>.md
    ml/results/outcome_report_<tag>.json

so a single driver can call it once per ablation checkpoint and aggregate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.data import DataConfig, build_dataloaders                # noqa: E402
from ml.jepa_model import CausalJEPA, TimeSeriesEncoder           # noqa: E402


# ======================================================================
# Data assembly helpers
# ======================================================================

DEFAULT_OUTCOMES = [
    "hba1c", "ldl_chol", "hdl_chol", "total_chol", "triglycerides",
    "glucose", "creatinine", "systolic_bp", "diastolic_bp", "bmi",
]


@dataclass
class PatientSplit:
    """In-memory tensors for one split."""

    indices: np.ndarray               # patient indices into the npz
    z: np.ndarray                     # (N,) intervention id
    pre_pooled: np.ndarray            # (N, F) mean over observed pre cells
    pre_observed_frac: np.ndarray     # (N, F) fraction of pre slots observed
    outcome_y: np.ndarray             # (N, n_out)  post-window biomarker mean
    outcome_m: np.ndarray             # (N, n_out)  1 if outcome observed
    person_ids: np.ndarray


def _safe_pooled_mean(values_obj, mask_obj, num_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient mean-over-time of observed cells; also fraction-observed."""
    N = len(values_obj)
    pooled = np.zeros((N, num_features), dtype=np.float32)
    frac = np.zeros((N, num_features), dtype=np.float32)
    for i in range(N):
        v = values_obj[i].astype(np.float32)
        m = mask_obj[i].astype(np.float32)
        n_slots = m.shape[0]
        counts = m.sum(axis=0)
        sums = (v * m).sum(axis=0)
        pooled[i] = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
        frac[i] = counts / max(n_slots, 1)
    return pooled, frac


def _outcome_targets(
    post_values_obj, post_mask_obj,
    outcome_feature_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """For each patient, compute per-outcome post-window mean (observed cells)."""
    N = len(post_values_obj)
    K = len(outcome_feature_idx)
    y = np.zeros((N, K), dtype=np.float32)
    m = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        v = post_values_obj[i].astype(np.float32)
        om = post_mask_obj[i].astype(np.float32)
        for k, feat_idx in enumerate(outcome_feature_idx):
            obs = om[:, feat_idx]
            if obs.sum() > 0:
                y[i, k] = float((v[:, feat_idx] * obs).sum() / obs.sum())
                m[i, k] = 1.0
    return y, m


def build_splits_and_outcomes(
    data_cfg: DataConfig,
    outcomes: list[str],
) -> tuple[PatientSplit, PatientSplit, PatientSplit, dict, dict]:
    """Build train/val/test splits aligned with the JEPA training split."""
    # Reuse the JEPA data loader to get meta + indices exactly.
    _, _, _, meta = build_dataloaders(data_cfg)

    # Reload raw npz to compute pooled features & outcome scalars.
    npz = np.load(data_cfg.npz_path, allow_pickle=True)
    pre_v, pre_m = npz["pre_values"], npz["pre_mask"]
    post_v, post_m = npz["post_values"], npz["post_mask"]
    z_all = npz["intervention_z"].astype(np.int64)
    pid_all = npz["person_ids"]

    num_features = int(meta["num_features"])
    feature_vocab = json.loads(Path("training_data/feature_vocab.json").read_text())
    outcome_idx = [feature_vocab[o] for o in outcomes]

    pooled, frac = _safe_pooled_mean(pre_v, pre_m, num_features)
    y_all, ym_all = _outcome_targets(post_v, post_m, outcome_idx)

    # Stratified split (same seeding/scheme as DataConfig via helper).
    from ml.data import _stratified_split
    train_idx, val_idx, test_idx = _stratified_split(
        z_all, data_cfg.val_fraction, data_cfg.test_fraction, data_cfg.seed,
    )

    def pack(idx):
        return PatientSplit(
            indices=idx,
            z=z_all[idx],
            pre_pooled=pooled[idx],
            pre_observed_frac=frac[idx],
            outcome_y=y_all[idx],
            outcome_m=ym_all[idx],
            person_ids=pid_all[idx],
        )

    outcome_meta = {
        "outcomes": outcomes,
        "outcome_idx_in_vocab": outcome_idx,
    }
    return pack(train_idx), pack(val_idx), pack(test_idx), meta, outcome_meta


# ======================================================================
# JEPA embedding helpers (encode the same test patients via frozen model)
# ======================================================================

@torch.no_grad()
def jepa_embeddings_for_indices(
    model: CausalJEPA,
    data_cfg: DataConfig,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """For the patients at `indices`, return (s_x, s_y, ŝ_y_at_true_z, z).

    We build a dataset once and iterate over just those indices.
    """
    from ml.data import build_dataloaders, AoUJepaDataset, _stratified_split, _masked_mean_std

    # Use the same standardization stats the JEPA was trained with
    npz = np.load(data_cfg.npz_path, allow_pickle=True)
    num_features = int(json.loads(Path(data_cfg.manifest_path).read_text())["num_features"])
    z_all = npz["intervention_z"].astype(np.int64)

    train_idx, _, _ = _stratified_split(
        z_all, data_cfg.val_fraction, data_cfg.test_fraction, data_cfg.seed,
    )
    feat_mean, feat_std = _masked_mean_std(
        npz["pre_values"], npz["pre_mask"], train_idx, num_features,
    )

    ds = AoUJepaDataset(npz, indices, feat_mean, feat_std, data_cfg.max_seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    model.eval()
    sx_list, sy_list, syhat_list, z_list = [], [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        _, sx = model.context_encoder(
            b["context_x"], b["context_mask"], b["context_timestamps"],
            src_key_padding_mask=b["context_padding_mask"],
        )
        _, sy = model.target_encoder(
            b["target_y"], b["target_mask"], b["target_timestamps"],
            src_key_padding_mask=b["target_padding_mask"],
        )
        ez = model.action_embedding(b["intervention_z"])
        syhat = model.predictor(sx, ez)
        sx_list.append(sx.cpu()); sy_list.append(sy.cpu())
        syhat_list.append(syhat.cpu()); z_list.append(b["intervention_z"].cpu())
    return (torch.cat(sx_list), torch.cat(sy_list),
            torch.cat(syhat_list), torch.cat(z_list))


# ======================================================================
# Metrics
# ======================================================================

def _observed(y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray):
    obs = mask > 0
    return y_pred[obs], y_true[obs]


def regression_metrics(
    y_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray,
) -> dict:
    """Per-outcome MAE / RMSE / R² / r over observed test patients."""
    y_pred_obs, y_true_obs = _observed(y_pred, y_true, mask)
    n = y_pred_obs.shape[0]
    if n < 2:
        return {"n": int(n), "mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "pearson": float("nan")}
    resid = y_pred_obs - y_true_obs
    mae = float(np.abs(resid).mean())
    rmse = float(np.sqrt((resid ** 2).mean()))
    var_y = float(((y_true_obs - y_true_obs.mean()) ** 2).mean())
    r2 = 1.0 - float((resid ** 2).mean()) / max(var_y, 1e-12)
    pearson = float(np.corrcoef(y_pred_obs, y_true_obs)[0, 1]) if n > 2 else float("nan")
    return {"n": int(n), "mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}


def paired_bootstrap_mae(
    y_a: np.ndarray, y_b: np.ndarray, y_true: np.ndarray, mask: np.ndarray,
    n_draws: int = 1000, seed: int = 0,
) -> dict:
    """Paired bootstrap on |y_a - y_true| - |y_b - y_true|.

    Positive mean Δ ⇒ model A has higher MAE than B (A is worse). Conversely.
    """
    obs = np.where(mask > 0)[0]
    if obs.size < 4:
        return {"n": int(obs.size), "mean_delta": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "p_two_sided": float("nan"),
                "frac_a_better": float("nan")}
    e_a = np.abs(y_a[obs] - y_true[obs])
    e_b = np.abs(y_b[obs] - y_true[obs])
    delta = e_a - e_b
    rng = np.random.default_rng(seed)
    n = obs.size
    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        pick = rng.integers(0, n, n)
        draws[i] = delta[pick].mean()
    mean = float(draws.mean())
    lo, hi = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
    # Two-sided p: fraction of resamples contradicting the observed sign
    obs_mean = float(delta.mean())
    if obs_mean >= 0:
        p = 2 * float((draws <= 0).mean())
    else:
        p = 2 * float((draws >= 0).mean())
    frac_a_better = float((delta < 0).mean())
    return {
        "n": int(obs.size),
        "mean_delta": mean,
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": min(max(p, 0.0), 1.0),
        "frac_a_better_per_patient": frac_a_better,
    }


# ======================================================================
# Sequence baselines (end-to-end)
# ======================================================================

class GRUBaseline(nn.Module):
    """Bi-directional GRU over (value, mask, time) + z → outcome heads.

    Built to be a *strong* competitor: same conditioning, different backbone.
    """

    def __init__(
        self,
        num_features: int,
        num_interventions: int,
        num_outcomes: int,
        hidden_dim: int = 128,
        z_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features * 2 + 1, hidden_dim)  # + time feature
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.action_embedding = nn.Embedding(num_interventions, z_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outcomes),
        )

    def forward(self, values, mask, timestamps, padding_mask, z):
        x = torch.cat([values * mask, mask, timestamps.unsqueeze(-1)], dim=-1)
        h = self.input_proj(x)
        out, _ = self.rnn(h)
        active = (~padding_mask).unsqueeze(-1).float()
        pooled = (out * active).sum(dim=1) / active.sum(dim=1).clamp(min=1.0)
        z_emb = self.action_embedding(z)
        return self.head(torch.cat([pooled, z_emb], dim=-1))


class TransformerBaseline(nn.Module):
    """Same TSEncoder backbone as JEPA but trained end-to-end for outcomes."""

    def __init__(
        self,
        num_features: int,
        num_interventions: int,
        num_outcomes: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        z_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TimeSeriesEncoder(
            num_features=num_features, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        )
        self.action_embedding = nn.Embedding(num_interventions, z_dim)
        self.head = nn.Sequential(
            nn.Linear(d_model + z_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_outcomes),
        )

    def forward(self, values, mask, timestamps, padding_mask, z):
        _, pooled = self.encoder(values, mask, timestamps, src_key_padding_mask=padding_mask)
        z_emb = self.action_embedding(z)
        return self.head(torch.cat([pooled, z_emb], dim=-1))


def train_sequence_baseline(
    model: nn.Module,
    data_cfg: DataConfig,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    outcome_idx: list[int],
    outcome_y_mean: np.ndarray,          # (K,) training-set mean
    outcome_y_std: np.ndarray,           # (K,) training-set std
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    patience: int = 6,
) -> nn.Module:
    """Train an end-to-end sequence baseline with masked MSE on normalized
    outcomes (Huber loss, since some outcomes are heavy-tailed).
    """
    from ml.data import AoUJepaDataset, _masked_mean_std

    npz = np.load(data_cfg.npz_path, allow_pickle=True)
    num_features = int(json.loads(Path(data_cfg.manifest_path).read_text())["num_features"])
    feat_mean, feat_std = _masked_mean_std(
        npz["pre_values"], npz["pre_mask"], train_idx, num_features,
    )

    # Pre-compute outcome tensors per patient idx
    y_all, m_all = _outcome_targets(npz["post_values"], npz["post_mask"], outcome_idx)
    y_norm = (y_all - outcome_y_mean) / np.where(outcome_y_std > 1e-6, outcome_y_std, 1.0)

    class _Wrapped(AoUJepaDataset):
        def __init__(self, npz, indices, means, stds, mxl, y_norm, m_all):
            super().__init__(npz, indices, means, stds, mxl)
            self._y = y_norm
            self._m = m_all

        def __getitem__(self, i):
            idx = int(self.indices[i])
            item = super().__getitem__(i)
            item["outcome_y"] = torch.from_numpy(self._y[idx].astype(np.float32))
            item["outcome_m"] = torch.from_numpy(self._m[idx].astype(np.float32))
            return item

    ds_tr = _Wrapped(npz, train_idx, feat_mean, feat_std, data_cfg.max_seq_len, y_norm, m_all)
    ds_va = _Wrapped(npz, val_idx, feat_mean, feat_std, data_cfg.max_seq_len, y_norm, m_all)
    tr_loader = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    huber = nn.HuberLoss(delta=1.0, reduction="none")
    best_val = float("inf")
    best_state = None
    stall = 0
    model.to(device)

    for ep in range(epochs):
        model.train()
        for b in tr_loader:
            b = {k: v.to(device) for k, v in b.items()}
            pred = model(
                b["context_x"], b["context_mask"], b["context_timestamps"],
                b["context_padding_mask"], b["intervention_z"],
            )
            y, m = b["outcome_y"], b["outcome_m"]
            loss_each = huber(pred, y)
            w = m / m.sum().clamp(min=1.0)
            loss = (loss_each * w).sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # ----- validation -----
        model.eval()
        val_num = 0.0; val_den = 0.0
        with torch.no_grad():
            for b in va_loader:
                b = {k: v.to(device) for k, v in b.items()}
                pred = model(
                    b["context_x"], b["context_mask"], b["context_timestamps"],
                    b["context_padding_mask"], b["intervention_z"],
                )
                y, m = b["outcome_y"], b["outcome_m"]
                resid = (pred - y).pow(2) * m
                val_num += resid.sum().item()
                val_den += m.sum().item()
        val_mse = val_num / max(val_den, 1.0)
        if val_mse < best_val - 1e-5:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def seq_baseline_predict(
    model: nn.Module,
    data_cfg: DataConfig,
    indices: np.ndarray,
    outcome_y_mean: np.ndarray,
    outcome_y_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return (N, K) predictions in *original units* on the provided indices."""
    from ml.data import AoUJepaDataset, _stratified_split, _masked_mean_std

    npz = np.load(data_cfg.npz_path, allow_pickle=True)
    num_features = int(json.loads(Path(data_cfg.manifest_path).read_text())["num_features"])
    z_all = npz["intervention_z"].astype(np.int64)
    train_idx, _, _ = _stratified_split(
        z_all, data_cfg.val_fraction, data_cfg.test_fraction, data_cfg.seed,
    )
    feat_mean, feat_std = _masked_mean_std(
        npz["pre_values"], npz["pre_mask"], train_idx, num_features,
    )

    ds = AoUJepaDataset(npz, indices, feat_mean, feat_std, data_cfg.max_seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

    model.eval()
    preds_n = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        pred = model(
            b["context_x"], b["context_mask"], b["context_timestamps"],
            b["context_padding_mask"], b["intervention_z"],
        )
        preds_n.append(pred.cpu().numpy())
    preds_n = np.concatenate(preds_n, axis=0)
    denom = np.where(outcome_y_std > 1e-6, outcome_y_std, 1.0)
    return preds_n * denom + outcome_y_mean


# ======================================================================
# JEPA-based readouts
# ======================================================================

def ridge_per_outcome(
    X_tr, Y_tr, M_tr, X_te, alpha: float = 1.0,
) -> np.ndarray:
    """Fit one Ridge per outcome column on training patients that observe it.

    Robust to per-outcome missingness in the training set.
    """
    N, K = Y_tr.shape
    preds = np.zeros((X_te.shape[0], K), dtype=np.float32)
    for k in range(K):
        obs = M_tr[:, k] > 0
        if obs.sum() < 20:
            preds[:, k] = np.mean(Y_tr[obs, k]) if obs.sum() > 0 else 0.0
            continue
        reg = Ridge(alpha=alpha)
        reg.fit(X_tr[obs], Y_tr[obs, k])
        preds[:, k] = reg.predict(X_te)
    return preds


def twin_retrieval_predict(
    train_sy: torch.Tensor, train_Y: np.ndarray, train_M: np.ndarray,
    test_syhat: torch.Tensor,
    top_k: int = 10,
    *,
    train_baseline: np.ndarray | None = None,
    test_baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Retrieve top-K training twins by cosine similarity from ŝ_y to train
    s_y and emit a per-outcome prediction.

    If ``train_baseline`` and ``test_baseline`` are provided (shape N_tr x K
    and N_te x K respectively) we run *residual retrieval*:

        ŷ_i = test_baseline[i] + mean_{j ∈ top-K}(train_Y[j] - train_baseline[j])

    restricted to outcomes the twin actually observed.  This removes the
    systematic selection bias that naive twin-mean retrieval inherits from
    the latent-space nearest-neighbour structure.
    """
    a = F.normalize(test_syhat, dim=-1)
    b = F.normalize(train_sy, dim=-1)
    sim = a @ b.T                                      # (N_te, N_tr)
    top = sim.topk(k=top_k, dim=-1).indices.numpy()    # (N_te, K)

    residual = train_baseline is not None and test_baseline is not None

    N_te = test_syhat.shape[0]
    K = train_Y.shape[1]
    preds = np.zeros((N_te, K), dtype=np.float32)
    pop_mean = np.zeros(K, dtype=np.float32)
    for k in range(K):
        obs = train_M[:, k] > 0
        pop_mean[k] = float(train_Y[obs, k].mean()) if obs.sum() > 0 else 0.0

    if residual:
        # y_i - baseline_i over observed outcomes only
        train_resid = train_Y - train_baseline          # invalid where M=0
    for i in range(N_te):
        t = top[i]
        yk = train_Y[t]
        mk = train_M[t]
        denom = mk.sum(axis=0)
        if residual:
            # Aggregate residuals where observed, otherwise zero (no update)
            r = (train_resid[t] * mk).sum(axis=0)
            mean_r = np.where(denom > 0, r / np.maximum(denom, 1.0), 0.0)
            preds[i] = test_baseline[i] + mean_r
        else:
            vals = np.where(
                denom > 0, (yk * mk).sum(axis=0) / np.maximum(denom, 1.0), pop_mean,
            )
            preds[i] = vals
    return preds


# ======================================================================
# Propensity Score Matching (PSM) for prescriber-confounding control
# ======================================================================

def fit_propensity_model(
    X_tr: np.ndarray, z_tr: np.ndarray, num_interventions: int, seed: int = 42,
) -> LogisticRegression:
    """Multinomial LR on pooled pre-window features to predict assigned drug."""
    lr = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=2000, C=1.0, random_state=seed,
    )
    lr.fit(X_tr, z_tr)
    return lr


def propensity_match_subcohort(
    X_te: np.ndarray,
    z_te: np.ndarray,
    propensity_model: LogisticRegression,
    focus_pairs: list[tuple[int, int]],
    caliper_std: float = 0.2,
    rng: np.random.Generator | None = None,
) -> dict[str, dict]:
    """1:1 nearest-neighbour matching on logit(P(z=A | x)) within each
    focus pair (e.g. metformin vs. lisinopril).

    Returns, for each focus pair, a dict with:
      - matched_idx: indices into the test set that survive matching
      - n_matched: count
      - mean_abs_std_diff: pre-match vs post-match standardised-mean-diff
                           on the top-10 absolute-loading features
    """
    rng = rng or np.random.default_rng(0)
    proba = propensity_model.predict_proba(X_te)                 # (N_te, K)
    # Replace zeros before log
    proba = np.clip(proba, 1e-6, 1.0 - 1e-6)

    out: dict[str, dict] = {}
    for zA, zB in focus_pairs:
        # logit of P(zA | x)
        p_a = proba[:, zA] / (proba[:, zA] + proba[:, zB] + 1e-12)
        logit_a = np.log(p_a / (1.0 - p_a))
        idx_a = np.where(z_te == zA)[0]
        idx_b = np.where(z_te == zB)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        # Caliper on SD of logit over the focus-pair population
        mask_pop = np.isin(z_te, [zA, zB])
        sd = float(np.std(logit_a[mask_pop]) + 1e-6)
        caliper = caliper_std * sd

        # Greedy 1:1 match without replacement: iterate treated A, find closest B
        used_b = set()
        matched_a, matched_b = [], []
        order = list(idx_a); rng.shuffle(order)
        for a in order:
            best = None; best_d = caliper + 1.0
            for b in idx_b:
                if b in used_b:
                    continue
                d = abs(logit_a[a] - logit_a[b])
                if d < best_d:
                    best = b; best_d = d
            if best is not None and best_d <= caliper:
                used_b.add(best)
                matched_a.append(a); matched_b.append(best)

        matched = np.array(matched_a + matched_b, dtype=np.int64)
        # Pre vs post balance: standardised mean difference of logit
        if len(matched) > 2:
            logit_pre_a = logit_a[idx_a]
            logit_pre_b = logit_a[idx_b]
            pre_smd = abs(logit_pre_a.mean() - logit_pre_b.mean()) / max(
                np.sqrt(0.5 * (logit_pre_a.var() + logit_pre_b.var())) + 1e-6, 1e-6,
            )
            logit_post_a = logit_a[matched_a]
            logit_post_b = logit_a[matched_b]
            post_smd = abs(logit_post_a.mean() - logit_post_b.mean()) / max(
                np.sqrt(0.5 * (logit_post_a.var() + logit_post_b.var())) + 1e-6, 1e-6,
            )
        else:
            pre_smd = post_smd = float("nan")

        out[f"{zA}_vs_{zB}"] = {
            "matched_idx": matched.tolist(),
            "n_matched": int(len(matched)),
            "n_treated": int(len(idx_a)),
            "n_control": int(len(idx_b)),
            "caliper_logit": float(caliper),
            "smd_logit_pre": float(pre_smd),
            "smd_logit_post": float(post_smd),
        }
    return out


# ======================================================================
# Main driver
# ======================================================================

@dataclass
class OutcomeCfg:
    checkpoint: str = "ml/checkpoints/jepa_best.pt"
    tag: str = "full"
    npz_path: str = "training_data/patient_tensors.npz"
    manifest_path: str = "training_data/manifest.json"
    intervention_map_path: str = "training_data/intervention_map.json"
    max_seq_len: int = 64
    batch_size: int = 64
    seed: int = 42
    outcomes: list[str] = field(default_factory=lambda: [
        "hba1c", "ldl_chol", "hdl_chol", "total_chol",
        "systolic_bp", "diastolic_bp", "bmi", "glucose",
    ])
    top_k: int = 10
    out_dir: str = "ml/results"
    skip_sequence_baselines: bool = False
    run_gru: bool = True
    run_transformer: bool = True
    seq_epochs: int = 30
    # Residual retrieval: subtract a per-outcome tabular RIDGE baseline
    # before averaging retrieved twins (de-biases selection).
    residual_retrieval: bool = True
    # Also report naive-mean twin retrieval under the label JEPA_TWIN_NAIVE
    report_naive_twins: bool = True
    # Propensity-score-matching analysis on focus drug pairs (by class-id).
    # Default: metformin (0) vs. lisinopril (2) if the map is
    # {metformin:0, atorvastatin:1, lisinopril:2}.
    run_psm: bool = True
    psm_focus_pairs: list[tuple[int, int]] = field(
        default_factory=lambda: [(0, 2), (0, 1), (1, 2)]
    )
    psm_caliper_std: float = 0.2


def _fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def run(cfg: OutcomeCfg) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig(
        npz_path=cfg.npz_path, manifest_path=cfg.manifest_path,
        intervention_map_path=cfg.intervention_map_path,
        max_seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed,
    )
    tr_split, va_split, te_split, meta, out_meta = build_splits_and_outcomes(data_cfg, cfg.outcomes)

    num_features = int(meta["num_features"])
    num_interventions = int(meta["num_interventions"])
    num_outcomes = len(cfg.outcomes)

    # Per-outcome normalization (training set)
    obs_tr = tr_split.outcome_m > 0
    y_mean = np.zeros(num_outcomes, dtype=np.float32)
    y_std = np.ones(num_outcomes, dtype=np.float32)
    for k in range(num_outcomes):
        o = obs_tr[:, k]
        if o.sum() > 1:
            y_mean[k] = float(tr_split.outcome_y[o, k].mean())
            y_std[k] = float(tr_split.outcome_y[o, k].std() + 1e-6)

    # -----------------------------------------------------------------
    # Build feature matrices for tabular baselines
    # -----------------------------------------------------------------
    z_onehot_tr = np.eye(num_interventions, dtype=np.float32)[tr_split.z]
    z_onehot_te = np.eye(num_interventions, dtype=np.float32)[te_split.z]

    X_tr_tab = np.concatenate([tr_split.pre_pooled, tr_split.pre_observed_frac, z_onehot_tr], axis=1)
    X_te_tab = np.concatenate([te_split.pre_pooled, te_split.pre_observed_frac, z_onehot_te], axis=1)

    scaler = StandardScaler().fit(X_tr_tab)
    X_tr_tab = scaler.transform(X_tr_tab)
    X_te_tab = scaler.transform(X_te_tab)

    Y_tr = tr_split.outcome_y.copy()
    M_tr = tr_split.outcome_m.copy()
    Y_te = te_split.outcome_y.copy()
    M_te = te_split.outcome_m.copy()

    # -----------------------------------------------------------------
    # Trivial baselines
    # -----------------------------------------------------------------
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    pop = np.broadcast_to(y_mean, Y_te.shape).copy()
    predictions["POP_MEAN"] = pop

    # NO_CHANGE: pre-pooled value of the same outcome feature (in original units!)
    outcome_cols = out_meta["outcome_idx_in_vocab"]
    pre_pooled_outcome = te_split.pre_pooled[:, outcome_cols]
    predictions["NO_CHANGE"] = pre_pooled_outcome.astype(np.float32)

    # -----------------------------------------------------------------
    # Tabular baselines
    # -----------------------------------------------------------------
    print("Fitting RIDGE...")
    predictions["RIDGE"] = ridge_per_outcome(X_tr_tab, Y_tr, M_tr, X_te_tab, alpha=1.0)

    print("Fitting GBT...")
    gbt_preds = np.zeros_like(Y_te, dtype=np.float32)
    for k in range(num_outcomes):
        o = M_tr[:, k] > 0
        if o.sum() < 20:
            gbt_preds[:, k] = y_mean[k]
            continue
        gb = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=cfg.seed,
        )
        gb.fit(X_tr_tab[o], Y_tr[o, k])
        gbt_preds[:, k] = gb.predict(X_te_tab)
    predictions["GBT"] = gbt_preds

    print("Fitting MLP_RAW...")
    mlp_preds = np.zeros_like(Y_te, dtype=np.float32)
    for k in range(num_outcomes):
        o = M_tr[:, k] > 0
        if o.sum() < 20:
            mlp_preds[:, k] = y_mean[k]
            continue
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu",
            solver="adam", alpha=1e-3, max_iter=400, random_state=cfg.seed,
            early_stopping=True, n_iter_no_change=15,
        )
        mlp.fit(X_tr_tab[o], Y_tr[o, k])
        mlp_preds[:, k] = mlp.predict(X_te_tab)
    predictions["MLP_RAW"] = mlp_preds

    # -----------------------------------------------------------------
    # Sequence baselines (end-to-end)
    # -----------------------------------------------------------------
    outcome_idx = out_meta["outcome_idx_in_vocab"]
    if not cfg.skip_sequence_baselines:
        if cfg.run_gru:
            print("Training GRU baseline...")
            t0 = time.time()
            gru = GRUBaseline(num_features, num_interventions, num_outcomes)
            gru = train_sequence_baseline(
                gru, data_cfg, tr_split.indices, va_split.indices,
                outcome_idx, y_mean, y_std, device, epochs=cfg.seq_epochs,
            )
            predictions["GRU_E2E"] = seq_baseline_predict(
                gru, data_cfg, te_split.indices, y_mean, y_std, device,
            )
            print(f"  GRU took {time.time()-t0:.1f}s")

        if cfg.run_transformer:
            print("Training TRANSFORMER baseline...")
            t0 = time.time()
            ts = TransformerBaseline(num_features, num_interventions, num_outcomes)
            ts = train_sequence_baseline(
                ts, data_cfg, tr_split.indices, va_split.indices,
                outcome_idx, y_mean, y_std, device, epochs=cfg.seq_epochs,
            )
            predictions["TSENCODER_E2E"] = seq_baseline_predict(
                ts, data_cfg, te_split.indices, y_mean, y_std, device,
            )
            print(f"  TSENCODER took {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------
    # JEPA-based readouts
    # -----------------------------------------------------------------
    ckpt_path = Path(cfg.checkpoint)
    jepa_results = None
    if ckpt_path.exists():
        print(f"Loading JEPA checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_cfg = ckpt["config"]
        model = CausalJEPA(
            num_features=num_features,
            d_model=ckpt_cfg["d_model"],
            n_heads=ckpt_cfg["n_heads"],
            n_layers=ckpt_cfg["n_layers"],
            d_ff=ckpt_cfg["d_ff"],
            dropout=ckpt_cfg.get("dropout", 0.1),
            num_interventions=num_interventions,
            z_dim=ckpt_cfg["z_dim"],
            predictor_hidden=ckpt_cfg["predictor_hidden"],
            predictor_layers=ckpt_cfg["predictor_layers"],
            predictor_style=ckpt_cfg.get("predictor_style", "adaln"),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        sx_tr, sy_tr, syhat_tr, z_tr = jepa_embeddings_for_indices(
            model, data_cfg, tr_split.indices, device,
        )
        sx_te, sy_te, syhat_te, z_te = jepa_embeddings_for_indices(
            model, data_cfg, te_split.indices, device,
        )

        sx_tr_np = sx_tr.numpy(); sx_te_np = sx_te.numpy()
        syhat_tr_np = syhat_tr.numpy(); syhat_te_np = syhat_te.numpy()

        # Standardize embeddings for Ridge
        sx_scaler = StandardScaler().fit(np.concatenate([sx_tr_np, z_onehot_tr], axis=1))
        X_tr_jepa = sx_scaler.transform(np.concatenate([sx_tr_np, z_onehot_tr], axis=1))
        X_te_jepa = sx_scaler.transform(np.concatenate([sx_te_np, z_onehot_te], axis=1))

        predictions["JEPA_RIDGE"] = ridge_per_outcome(
            X_tr_jepa, Y_tr, M_tr, X_te_jepa, alpha=1.0,
        )

        syhat_scaler = StandardScaler().fit(syhat_tr_np)
        X_tr_pred = syhat_scaler.transform(syhat_tr_np)
        X_te_pred = syhat_scaler.transform(syhat_te_np)
        predictions["JEPA_RIDGE_PRED"] = ridge_per_outcome(
            X_tr_pred, Y_tr, M_tr, X_te_pred, alpha=1.0,
        )

        # Residual retrieval: subtract per-outcome RIDGE baseline before
        # averaging retrieved twins, then add back the baseline for the
        # query patient under the *query* intervention.  This removes
        # selection bias that made naive twin-mean retrieval worse than
        # the population mean on this cohort.
        if cfg.residual_retrieval:
            # Use the *RIDGE* predictions already computed on X_*_tab as
            # the per-outcome baseline (identical recipe: pooled pre
            # features + observation-fraction + one-hot z).
            ridge_tr = ridge_per_outcome(X_tr_tab, Y_tr, M_tr, X_tr_tab, alpha=1.0)
            ridge_te = predictions["RIDGE"]
            predictions["JEPA_TWIN"] = twin_retrieval_predict(
                sy_tr, Y_tr, M_tr, syhat_te, top_k=cfg.top_k,
                train_baseline=ridge_tr, test_baseline=ridge_te,
            )
            if cfg.report_naive_twins:
                predictions["JEPA_TWIN_NAIVE"] = twin_retrieval_predict(
                    sy_tr, Y_tr, M_tr, syhat_te, top_k=cfg.top_k,
                )
        else:
            predictions["JEPA_TWIN"] = twin_retrieval_predict(
                sy_tr, Y_tr, M_tr, syhat_te, top_k=cfg.top_k,
            )

        jepa_results = {
            "checkpoint": str(ckpt_path),
            "val_loss": float(ckpt.get("val_loss", float("nan"))),
            "epoch": int(ckpt.get("epoch", -1)),
            "config": ckpt_cfg,
        }
    else:
        print(f"[warn] JEPA checkpoint not found: {ckpt_path}")

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    for name, preds in predictions.items():
        per_outcome = {}
        for k, o in enumerate(cfg.outcomes):
            per_outcome[o] = regression_metrics(preds[:, k], Y_te[:, k], M_te[:, k])
        obs_counts = np.array([per_outcome[o]["n"] for o in cfg.outcomes])
        maes = np.array([per_outcome[o]["mae"] for o in cfg.outcomes], dtype=np.float64)
        r2s = np.array([per_outcome[o]["r2"] for o in cfg.outcomes], dtype=np.float64)
        good = obs_counts >= 30
        if good.any():
            macro_mae = float(np.nanmean(maes[good]))
            macro_r2 = float(np.nanmean(r2s[good]))
        else:
            macro_mae = float("nan"); macro_r2 = float("nan")
        results[name] = {
            "per_outcome": per_outcome,
            "macro_mae": macro_mae,
            "macro_r2": macro_r2,
            "n_outcomes_used": int(good.sum()),
        }

    # -----------------------------------------------------------------
    # Propensity Score Matching — prescriber-confounding control
    # Fit LR(pre-pooled features + obs-frac) -> intervention on TRAIN; then
    # 1:1 match on logit(P) for focus drug pairs on TEST; re-score all
    # models on the matched subcohort.
    # -----------------------------------------------------------------
    psm_summary: dict = {}
    if cfg.run_psm:
        X_tab_for_ps_tr = np.concatenate(
            [tr_split.pre_pooled, tr_split.pre_observed_frac], axis=1,
        )
        X_tab_for_ps_te = np.concatenate(
            [te_split.pre_pooled, te_split.pre_observed_frac], axis=1,
        )
        ps_scaler = StandardScaler().fit(X_tab_for_ps_tr)
        X_tab_for_ps_tr = ps_scaler.transform(X_tab_for_ps_tr)
        X_tab_for_ps_te = ps_scaler.transform(X_tab_for_ps_te)
        prop_model = fit_propensity_model(
            X_tab_for_ps_tr, tr_split.z, num_interventions, seed=cfg.seed,
        )
        ps_rng = np.random.default_rng(cfg.seed)
        match_info = propensity_match_subcohort(
            X_tab_for_ps_te, te_split.z, prop_model,
            focus_pairs=cfg.psm_focus_pairs,
            caliper_std=cfg.psm_caliper_std,
            rng=ps_rng,
        )
        inv_map = json.loads(Path(cfg.intervention_map_path).read_text())
        ctz = inv_map.get("concept_to_z") or inv_map
        lbls = inv_map.get("labels") or {}
        z_to_label: dict[int, str] = {}
        for cid, zv in ctz.items():
            z_to_label[int(zv)] = str(lbls.get(str(cid), f"z{int(zv)}"))
        for pair_key, info in match_info.items():
            zA_str, _vs, zB_str = pair_key.split("_")
            pair_label = (
                f"{z_to_label.get(int(zA_str), zA_str)}_vs_"
                f"{z_to_label.get(int(zB_str), zB_str)}"
            )
            matched_idx = np.array(info["matched_idx"], dtype=np.int64)
            pair_block: dict = {
                "pair": pair_label,
                "n_matched": info["n_matched"],
                "smd_logit_pre": info["smd_logit_pre"],
                "smd_logit_post": info["smd_logit_post"],
                "per_model": {},
            }
            if matched_idx.size > 4:
                for name, preds in predictions.items():
                    per_outcome = {}
                    for k, o in enumerate(cfg.outcomes):
                        per_outcome[o] = regression_metrics(
                            preds[matched_idx, k],
                            Y_te[matched_idx, k],
                            M_te[matched_idx, k],
                        )
                    obs_counts = np.array(
                        [per_outcome[o]["n"] for o in cfg.outcomes],
                    )
                    maes = np.array(
                        [per_outcome[o]["mae"] for o in cfg.outcomes],
                        dtype=np.float64,
                    )
                    good = obs_counts >= 15
                    r2s = np.array(
                        [per_outcome[o]["r2"] for o in cfg.outcomes],
                        dtype=np.float64,
                    )
                    macro_mae = (
                        float(np.nanmean(maes[good])) if good.any() else float("nan")
                    )
                    macro_r2_psm = (
                        float(np.nanmean(r2s[good])) if good.any() else float("nan")
                    )
                    pair_block["per_model"][name] = {
                        "per_outcome": per_outcome,
                        "macro_mae": macro_mae,
                        "macro_r2": macro_r2_psm,
                        "n_outcomes_used": int(good.sum()),
                    }
            psm_summary[pair_label] = pair_block

    # -----------------------------------------------------------------
    # Paired bootstraps (each model vs best tabular baseline RIDGE/GBT)
    # Compare against whichever has lowest macro MAE among non-JEPA baselines.
    # -----------------------------------------------------------------
    non_jepa = [n for n in predictions if not n.startswith("JEPA_")]
    best_base = min(non_jepa, key=lambda n: results[n]["macro_mae"])
    bootstraps = {}
    for name, preds in predictions.items():
        if name == best_base:
            continue
        per_outcome = {}
        for k, o in enumerate(cfg.outcomes):
            per_outcome[o] = paired_bootstrap_mae(
                preds[:, k], predictions[best_base][:, k], Y_te[:, k], M_te[:, k],
                n_draws=1000, seed=cfg.seed + k,
            )
        bootstraps[name] = {"vs_baseline": best_base, "per_outcome": per_outcome}

    report = {
        "tag": cfg.tag,
        "device": str(device),
        "outcomes": cfg.outcomes,
        "splits": {
            "train": int(len(tr_split.indices)),
            "val": int(len(va_split.indices)),
            "test": int(len(te_split.indices)),
        },
        "observed_test_counts": {
            o: int(M_te[:, k].sum()) for k, o in enumerate(cfg.outcomes)
        },
        "jepa_checkpoint": jepa_results,
        "best_baseline": best_base,
        "results": results,
        "bootstraps": bootstraps,
        "psm": psm_summary,
        "cfg": asdict(cfg),
    }

    # -----------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"outcome_report_{cfg.tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )

    # Markdown report
    md = [
        f"# Outcome-prediction report — tag `{cfg.tag}`",
        "",
        f"- Checkpoint: `{cfg.checkpoint}`",
        f"- Device: `{device}`",
        f"- Splits: train={report['splits']['train']} · val={report['splits']['val']} · test={report['splits']['test']}",
        f"- Best non-JEPA baseline (for bootstraps): **{best_base}**",
        "",
        "## Observed test patient counts per outcome",
        "",
        "| outcome | n |",
        "|:---|---:|",
    ]
    for o, n in report["observed_test_counts"].items():
        md.append(f"| {o} | {n} |")
    md += ["", "## MAE per model and outcome (real units)", ""]
    header = "| model | " + " | ".join(cfg.outcomes) + " | macro MAE | macro R² |"
    sep = "|:---" + "|---:" * (len(cfg.outcomes) + 2) + "|"
    md += [header, sep]
    # Order rows: baselines first, then JEPA variants, then mark the winner per column
    ordered = [m for m in ["POP_MEAN", "NO_CHANGE", "RIDGE", "GBT", "MLP_RAW",
                           "GRU_E2E", "TSENCODER_E2E",
                           "JEPA_RIDGE", "JEPA_RIDGE_PRED",
                           "JEPA_TWIN", "JEPA_TWIN_NAIVE"] if m in results]
    for name in ordered:
        r = results[name]
        cells = []
        for o in cfg.outcomes:
            v = r["per_outcome"][o]["mae"]
            cells.append(_fmt(v))
        cells.append(_fmt(r["macro_mae"]))
        cells.append(_fmt(r["macro_r2"]))
        md.append(f"| {name} | " + " | ".join(cells) + " |")

    md += ["", "## R² per model and outcome", ""]
    md += [header, sep]
    for name in ordered:
        r = results[name]
        cells = [_fmt(r["per_outcome"][o]["r2"]) for o in cfg.outcomes]
        cells.append(_fmt(r["macro_mae"]))
        cells.append(_fmt(r["macro_r2"]))
        md.append(f"| {name} | " + " | ".join(cells) + " |")

    md += ["", f"## Paired bootstrap MAE: model − {best_base} (negative Δ ⇒ model better)", ""]
    md += ["| model | outcome | Δ MAE | 95% CI | p (two-sided) | % test patients model-better |",
           "|:---|:---|---:|:---:|---:|---:|"]
    for name in ordered:
        if name not in bootstraps:
            continue
        b = bootstraps[name]
        for o in cfg.outcomes:
            bs = b["per_outcome"][o]
            md.append(
                f"| {name} | {o} | {_fmt(bs['mean_delta'])} | "
                f"[{_fmt(bs['ci_low'])}, {_fmt(bs['ci_high'])}] | "
                f"{_fmt(bs['p_two_sided'])} | "
                f"{_fmt((bs.get('frac_a_better_per_patient') or 0.0)*100)}% |"
            )

    md += ["", "## Verdicts", ""]
    # Macro-MAE verdicts
    base_macro = results[best_base]["macro_mae"]
    for name in ordered:
        if name == best_base:
            md.append(f"- {name}: baseline (macro MAE {base_macro:.3f})")
            continue
        macro = results[name]["macro_mae"]
        sign = "+" if macro > base_macro else ""
        md.append(
            f"- {name}: macro MAE {macro:.3f} ({sign}{macro - base_macro:+.3f} vs {best_base})"
        )

    # PSM subcohort block
    if psm_summary:
        md += [
            "",
            "## Propensity-matched subcohort (prescriber-confounding control)",
            "",
            "Matched 1:1 on logit(P(z=A|x)) from multinomial LR on pooled pre-window features",
            f"(caliper = {cfg.psm_caliper_std}σ of logit).",
            "",
        ]
        for pair_label, block in psm_summary.items():
            md += [
                f"### {pair_label}",
                "",
                f"- n_matched = **{block['n_matched']}**",
                f"- SMD(logit) pre -> post: {_fmt(block['smd_logit_pre'])} -> {_fmt(block['smd_logit_post'])}",
                "",
                "| model | macro MAE (matched) |",
                "|:---|---:|",
            ]
            for name in ordered:
                pm = block["per_model"].get(name)
                if pm is None:
                    continue
                md.append(f"| {name} | {_fmt(pm['macro_mae'])} |")
            md.append("")

    (out_dir / f"outcome_report_{cfg.tag}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[ok] wrote {out_dir / f'outcome_report_{cfg.tag}.md'}")
    return report


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> OutcomeCfg:
    cfg = OutcomeCfg()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=cfg.checkpoint)
    p.add_argument("--tag", type=str, default=cfg.tag)
    p.add_argument("--outcomes", nargs="+", default=cfg.outcomes)
    p.add_argument("--top-k", type=int, default=cfg.top_k)
    p.add_argument("--out-dir", type=str, default=cfg.out_dir)
    p.add_argument("--skip-sequence-baselines", action="store_true")
    p.add_argument("--no-gru", action="store_true")
    p.add_argument("--no-transformer", action="store_true")
    p.add_argument("--seq-epochs", type=int, default=cfg.seq_epochs)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--naive-twins", action="store_true",
                   help="Disable residual retrieval and use naive twin-mean")
    p.add_argument("--no-psm", action="store_true",
                   help="Skip propensity-score-matching analysis")
    a = p.parse_args()
    cfg.checkpoint = a.checkpoint
    cfg.tag = a.tag
    cfg.outcomes = a.outcomes
    cfg.top_k = a.top_k
    cfg.out_dir = a.out_dir
    cfg.skip_sequence_baselines = a.skip_sequence_baselines
    cfg.run_gru = not a.no_gru
    cfg.run_transformer = not a.no_transformer
    cfg.seq_epochs = a.seq_epochs
    cfg.seed = a.seed
    cfg.residual_retrieval = not a.naive_twins
    cfg.run_psm = not a.no_psm
    return cfg


if __name__ == "__main__":
    run(parse_args())
