"""
Comprehensive evaluation suite for the Causal-JEPA.

Metrics
-------
1. Latent MSE  — F.mse_loss(predicted_s_y, s_y). Headline JEPA objective.
2. Cosine similarity  — mean(cos(predicted_s_y, s_y)). More scale-robust.
3. Explained variance  — 1 - Var(s_y - ŝ_y) / Var(s_y). Directly interpretable.
4. Variance-collapse diagnostic  — mean per-dim std of s_y across the batch.
   Healthy representations have std >> 0; collapse reads std ≈ 0.
5. Intervention sensitivity  — average L2 distance between ŝ_y predictions
   for the *same* context under different z values (relative to the batch std
   of s_y). Measures whether the predictor actually conditions on z.
6. Intervention linear-probe accuracy  — logistic-regression classifier on
   the (frozen) context embeddings predicting the intervention label.
   Indicates whether the encoder learned pre-intervention features that
   differentiate patient populations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ml.jepa_model import CausalJEPA


# ----------------------------------------------------------------------
# Core forward-pass helpers
# ----------------------------------------------------------------------

@torch.no_grad()
def _forward_batch(
    model: CausalJEPA,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model (eval mode) and return (ŝ_y, s_y, s_x)."""
    model.eval()
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    # s_x via context encoder
    _, s_x = model.context_encoder(
        b["context_x"], b["context_mask"], b["context_timestamps"],
        src_key_padding_mask=b.get("context_padding_mask"),
    )
    # s_y via target encoder (already no-grad, but keep eval mode)
    _, s_y = model.target_encoder(
        b["target_y"], b["target_mask"], b["target_timestamps"],
        src_key_padding_mask=b.get("target_padding_mask"),
    )
    embedded_z = model.action_embedding(b["intervention_z"])
    s_y_hat = model.predictor(s_x, embedded_z)
    return s_y_hat, s_y, s_x


# ----------------------------------------------------------------------
# Metric aggregator for one split
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate_loader(
    model: CausalJEPA,
    loader: DataLoader,
    device: torch.device,
    num_interventions: int,
) -> dict:
    """Compute headline JEPA metrics and collect embeddings for probes."""
    all_sy_hat, all_sy, all_sx, all_z = [], [], [], []

    for batch in loader:
        s_y_hat, s_y, s_x = _forward_batch(model, batch, device)
        all_sy_hat.append(s_y_hat.cpu())
        all_sy.append(s_y.cpu())
        all_sx.append(s_x.cpu())
        all_z.append(batch["intervention_z"])

    sy_hat = torch.cat(all_sy_hat, dim=0)
    sy = torch.cat(all_sy, dim=0)
    sx = torch.cat(all_sx, dim=0)
    z = torch.cat(all_z, dim=0)

    mse = F.mse_loss(sy_hat, sy).item()
    cos = F.cosine_similarity(sy_hat, sy, dim=-1).mean().item()

    # Explained variance of target embeddings
    residual_var = (sy - sy_hat).var(dim=0, unbiased=False).mean().item()
    target_var = sy.var(dim=0, unbiased=False).mean().item()
    explained_var = 1.0 - residual_var / max(target_var, 1e-12)

    # Variance-collapse diagnostics
    sy_std = sy.std(dim=0, unbiased=False).mean().item()
    sx_std = sx.std(dim=0, unbiased=False).mean().item()

    # Intervention sensitivity: for each example, predict under all z values,
    # measure average pairwise L2 distance between those predictions — relative
    # to a natural scale (std of s_y).
    intervention_sensitivity = _intervention_sensitivity(model, sx, num_interventions, device)
    rel_sensitivity = intervention_sensitivity / max(sy_std, 1e-12)

    return {
        "mse": mse,
        "cosine_sim": cos,
        "explained_var": explained_var,
        "target_embedding_std": sy_std,
        "context_embedding_std": sx_std,
        "intervention_sensitivity": intervention_sensitivity,
        "intervention_sensitivity_relative": rel_sensitivity,
        "n_examples": int(sy.shape[0]),
        # returned for downstream probes / saving
        "_sx": sx,
        "_sy": sy,
        "_sy_hat": sy_hat,
        "_z": z,
    }


@torch.no_grad()
def _intervention_sensitivity(
    model: CausalJEPA,
    sx: torch.Tensor,
    num_interventions: int,
    device: torch.device,
) -> float:
    """Average pairwise L2 distance between predictor outputs for the same
    context but different intervention ids. 0 means the predictor ignores z.
    """
    B = sx.shape[0]
    sx_dev = sx.to(device)
    preds = []
    for z_val in range(num_interventions):
        z = torch.full((B,), z_val, dtype=torch.long, device=device)
        preds.append(model.predict_counterfactual(sx_dev, z).cpu())
    preds = torch.stack(preds, dim=0)                        # (K, B, D)
    K = preds.shape[0]
    total, pairs = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += (preds[i] - preds[j]).pow(2).sum(dim=-1).sqrt().mean().item()
            pairs += 1
    return total / max(pairs, 1)


# ----------------------------------------------------------------------
# Linear probe: does s_x encode intervention?
# ----------------------------------------------------------------------

def intervention_linear_probe(
    train_sx: torch.Tensor, train_z: torch.Tensor,
    eval_sx: torch.Tensor,  eval_z: torch.Tensor,
) -> dict:
    """Logistic-regression probe for intervention label from s_x.

    Chance-level for 3 balanced classes ≈ 0.33. Above chance indicates
    the encoder separates intervention groups by observable patient state.
    """
    X_tr = StandardScaler().fit_transform(train_sx.numpy())
    X_ev = StandardScaler().fit_transform(eval_sx.numpy())
    y_tr = train_z.numpy()
    y_ev = eval_z.numpy()

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
    )
    clf.fit(X_tr, y_tr)
    acc_tr = float(clf.score(X_tr, y_tr))
    acc_ev = float(clf.score(X_ev, y_ev))

    # Per-class accuracy on eval
    per_class = {}
    for cls in np.unique(y_ev):
        mask = y_ev == cls
        if mask.sum() == 0:
            continue
        per_class[int(cls)] = float((clf.predict(X_ev[mask]) == cls).mean())

    # Majority baseline
    majority_cls = np.bincount(y_tr).argmax()
    majority_acc = float((y_ev == majority_cls).mean())

    return {
        "train_acc": acc_tr,
        "eval_acc": acc_ev,
        "per_class_eval_acc": per_class,
        "majority_baseline": majority_acc,
    }
