"""
Fair baselines for paper comparison: predict target embedding s_y from
(1) frozen context embedding s_x from the trained JEPA (+ optional one-hot z),
(2) pooled raw pre-window labs + one-hot z (no transformer encoder).

All models are trained only on the train split; metrics on the held-out test split
match the JEPA headline task: MSE and cosine vs the **same** target-encoder s_y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluate import _forward_batch

@dataclass
class BaselineMetrics:
    name: str
    mse: float
    cosine_sim: float
    explained_var: float
    mse_per_sample: np.ndarray  # (N,) for bootstrap
    mse_normalized: float = 0.0  # MSE on L2-normalized preds & targets (scale-free)


def _explained_var(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    residual_var = (y_true - y_pred).var(dim=0, unbiased=False).mean().item()
    target_var = y_true.var(dim=0, unbiased=False).mean().item()
    return 1.0 - residual_var / max(target_var, 1e-12)


def _mse_unit_sphere(pt: torch.Tensor, st: torch.Tensor) -> float:
    """MSE after row-wise L2 normalize — closer to what cosine optimizes."""
    pn = F.normalize(pt, dim=-1, eps=1e-8)
    sn = F.normalize(st, dim=-1, eps=1e-8)
    return F.mse_loss(pn, sn).item()


def _metrics_from_numpy(
    pred: np.ndarray, sy: np.ndarray, name: str,
) -> BaselineMetrics:
    pt = torch.from_numpy(pred).float()
    st = torch.from_numpy(sy).float()
    mse = F.mse_loss(pt, st).item()
    cos = F.cosine_similarity(pt, st, dim=-1).mean().item()
    ev = _explained_var(st, pt)
    mse_per = ((pt - st) ** 2).sum(dim=-1).numpy()
    mn = _mse_unit_sphere(pt, st)
    return BaselineMetrics(name, mse, cos, ev, mse_per, mse_normalized=mn)


def _metrics_from_torch(
    pred: torch.Tensor, sy: torch.Tensor, name: str,
) -> BaselineMetrics:
    mse = F.mse_loss(pred, sy).item()
    cos = F.cosine_similarity(pred, sy, dim=-1).mean().item()
    ev = _explained_var(sy, pred)
    mse_per = ((pred - sy) ** 2).sum(dim=-1).detach().cpu().numpy()
    mn = _mse_unit_sphere(pred, sy)
    return BaselineMetrics(name, mse, cos, ev, mse_per, mse_normalized=mn)


def _one_hot(z: torch.Tensor, k: int) -> torch.Tensor:
    return F.one_hot(z.long(), num_classes=k).float()


def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (s_x, s_y, z) on CPU."""
    sx_l, sy_l, z_l = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            _, s_x = model.context_encoder(
                b["context_x"], b["context_mask"], b["context_timestamps"],
                src_key_padding_mask=b.get("context_padding_mask"),
            )
            _, s_y = model.target_encoder(
                b["target_y"], b["target_mask"], b["target_timestamps"],
                src_key_padding_mask=b.get("target_padding_mask"),
            )
            sx_l.append(s_x.cpu())
            sy_l.append(s_y.cpu())
            z_l.append(batch["intervention_z"].long())
    return (
        torch.cat(sx_l, dim=0),
        torch.cat(sy_l, dim=0),
        torch.cat(z_l, dim=0),
    )


def collect_pooled_raw_features(
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool pre-window values over observed steps -> (N, F) and z (N,)."""
    feats, zs = [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        x = b["context_x"]
        m = b["context_mask"]
        pm = b.get("context_padding_mask")
        # mask padded steps: valid where not padding and mask>0
        if pm is not None:
            valid_time = (~pm).float().unsqueeze(-1)
        else:
            valid_time = torch.ones(x.shape[0], x.shape[1], 1, device=device)
        w = m * valid_time
        num = (x * w).sum(dim=1)
        den = w.sum(dim=1).clamp(min=1e-6)
        pooled = num / den
        feats.append(pooled.cpu())
        zs.append(batch["intervention_z"].long())
    return torch.cat(feats, dim=0), torch.cat(zs, dim=0)


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.1)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_torch_mlp(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_va: torch.Tensor,
    y_va: torch.Tensor,
    out_dim: int,
    hidden: int,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> TorchMLP:
    in_dim = X_tr.shape[1]
    m = TorchMLP(in_dim, out_dim, hidden=hidden, n_layers=2).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    tr_ds = TensorDataset(X_tr, y_tr)
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True)
    best_state = None
    best_val = float("inf")
    patience, bad = 8, 0
    for ep in range(epochs):
        m.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.mse_loss(m(xb), yb)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = F.mse_loss(m(X_va.to(device)), y_va.to(device)).item()
        if v < best_val - 1e-6:
            best_val = v
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m.to(device)


def bootstrap_paired_mse(
    mse_jepa_per: np.ndarray,
    mse_base_per: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap mean(MSE_base - MSE_jepa). Positive => JEPA better (lower MSE)."""
    rng = np.random.default_rng(seed)
    n = len(mse_jepa_per)
    diff = mse_base_per - mse_jepa_per
    observed = float(diff.mean())
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(diff[idx].mean()))
    boots_arr = np.array(boots)
    ci_lo, ci_hi = np.percentile(boots_arr, [2.5, 97.5])
    p_two = 2 * min((boots_arr <= 0).mean(), (boots_arr >= 0).mean())
    p_two = min(float(p_two), 1.0)
    return {
        "mean_mse_improvement_base_minus_jepa": observed,
        "bootstrap_ci95_low": float(ci_lo),
        "bootstrap_ci95_high": float(ci_hi),
        "bootstrap_p_two_sided": p_two,
        "n_bootstrap": n_boot,
        "fraction_bootstraps_jepa_better": float((boots_arr > 0).mean()),
    }


def run_all_baselines(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_interventions: int,
    d_model: int,
    jepa_mse_per_sample: np.ndarray | None,
    bootstrap_n: int = 2000,
    bootstrap_seed: int = 0,
    torch_mlp_epochs: int = 80,
) -> dict[str, Any]:
    """
    Train baselines; compare to JEPA using per-sample MSE if provided.

    jepa_mse_per_sample: squared L2 per test row ||ŝ_y - s_y||^2 (same as baselines).
    """
    sx_tr, sy_tr, z_tr = collect_embeddings(model, train_loader, device)
    sx_va, sy_va, z_va = collect_embeddings(model, val_loader, device)
    sx_te, sy_te, z_te = collect_embeddings(model, test_loader, device)

    oh_tr = _one_hot(z_tr, num_interventions)
    oh_va = _one_hot(z_va, num_interventions)
    oh_te = _one_hot(z_te, num_interventions)

    Xz_tr = torch.cat([sx_tr, oh_tr], dim=-1).numpy()
    Xz_va = torch.cat([sx_va, oh_va], dim=-1).numpy()
    Xz_te = torch.cat([sx_te, oh_te], dim=-1).numpy()

    X0_tr = sx_tr.numpy()
    X0_va = sx_va.numpy()
    X0_te = sx_te.numpy()

    sy_tr_np = sy_tr.numpy()
    sy_va_np = sy_va.numpy()
    sy_te_np = sy_te.numpy()

    results: dict[str, Any] = {
        "protocol": (
            "Baselines predict the same target-encoder s_y. "
            "Ridge / sklearn-MLP / torch-MLP use frozen s_x from the trained "
            "JEPA context encoder (+ one-hot z). "
            "Pooled-raw MLP uses mean-pooled standardized pre-window labs (+ z), "
            "no transformer — a classical tabular baseline."
        ),
        "models": {},
    }

    # --- Ridge + z (scaled inputs) ---
    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, max_iter=10_000),
    )
    ridge.fit(Xz_tr, sy_tr_np)
    r = _metrics_from_numpy(ridge.predict(Xz_te), sy_te_np, "Ridge(s_x + one-hot z)")
    results["models"][r.name] = {
        "mse": r.mse,
        "cosine_sim": r.cosine_sim,
        "explained_var": r.explained_var,
        "mse_normalized": r.mse_normalized,
    }
    ridge_z_mse_per = r.mse_per_sample

    # --- Ridge s_x only ---
    ridge0 = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, max_iter=10_000),
    )
    ridge0.fit(X0_tr, sy_tr_np)
    r0 = _metrics_from_numpy(ridge0.predict(X0_te), sy_te_np, "Ridge(s_x only, no z)")
    results["models"][r0.name] = {
        "mse": r0.mse,
        "cosine_sim": r0.cosine_sim,
        "explained_var": r0.explained_var,
        "mse_normalized": r0.mse_normalized,
    }

    # --- sklearn MLP + z (scaled) ---
    skm = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=0,
            alpha=1e-4,
        ),
    )
    skm.fit(Xz_tr, sy_tr_np)
    rs = _metrics_from_numpy(
        skm.predict(Xz_te), sy_te_np, "sklearn MLP(s_x + one-hot z)",
    )
    results["models"][rs.name] = {
        "mse": rs.mse,
        "cosine_sim": rs.cosine_sim,
        "explained_var": rs.explained_var,
        "mse_normalized": rs.mse_normalized,
    }
    sk_mlp_mse_per = rs.mse_per_sample

    # --- Torch MLP + z ---
    mlp = _fit_torch_mlp(
        torch.cat([sx_tr, oh_tr], dim=-1),
        sy_tr,
        torch.cat([sx_va, oh_va], dim=-1),
        sy_va,
        out_dim=d_model,
        hidden=256,
        epochs=torch_mlp_epochs,
        device=device,
    )
    mlp.eval()
    with torch.no_grad():
        pred_te = mlp(torch.cat([sx_te, oh_te], dim=-1).to(device)).cpu()
    rt = _metrics_from_torch(pred_te, sy_te, "PyTorch MLP(s_x + one-hot z)")
    results["models"][rt.name] = {
        "mse": rt.mse,
        "cosine_sim": rt.cosine_sim,
        "explained_var": rt.explained_var,
        "mse_normalized": rt.mse_normalized,
    }
    torch_mlp_mse_per = rt.mse_per_sample

    # --- Pooled raw + z ---
    raw_tr, zr_tr = collect_pooled_raw_features(train_loader, device)
    raw_va, zr_va = collect_pooled_raw_features(val_loader, device)
    raw_te, zr_te = collect_pooled_raw_features(test_loader, device)
    oh_rtr = _one_hot(zr_tr, num_interventions)
    oh_rva = _one_hot(zr_va, num_interventions)
    oh_rte = _one_hot(zr_te, num_interventions)
    Xin_tr = torch.cat([raw_tr, oh_rtr], dim=-1)
    Xin_va = torch.cat([raw_va, oh_rva], dim=-1)
    Xin_te = torch.cat([raw_te, oh_rte], dim=-1)
    mlp_raw = _fit_torch_mlp(
        Xin_tr, sy_tr, Xin_va, sy_va,
        out_dim=d_model, hidden=128, epochs=torch_mlp_epochs, device=device,
    )
    mlp_raw.eval()
    with torch.no_grad():
        pred_r = mlp_raw(Xin_te.to(device)).cpu()
    rr = _metrics_from_torch(pred_r, sy_te, "PyTorch MLP(mean-pooled raw pre + z)")
    results["models"][rr.name] = {
        "mse": rr.mse,
        "cosine_sim": rr.cosine_sim,
        "explained_var": rr.explained_var,
        "mse_normalized": rr.mse_normalized,
    }
    raw_mlp_mse_per = rr.mse_per_sample

    # --- vs JEPA bootstrap ---
    results["paired_bootstrap_vs_jepa"] = {}
    results["paired_bootstrap_normalized_vs_ridge"] = {}
    if jepa_mse_per_sample is not None and len(jepa_mse_per_sample) == len(ridge_z_mse_per):
        for name, per in [
            ("Ridge(s_x + one-hot z)", ridge_z_mse_per),
            ("sklearn MLP(s_x + one-hot z)", sk_mlp_mse_per),
            ("PyTorch MLP(s_x + one-hot z)", torch_mlp_mse_per),
            ("PyTorch MLP(mean-pooled raw pre + z)", raw_mlp_mse_per),
        ]:
            results["paired_bootstrap_vs_jepa"][name] = bootstrap_paired_mse(
                jepa_mse_per_sample, per, n_boot=bootstrap_n, seed=bootstrap_seed,
            )
        rp = ridge.predict(Xz_te)
        pt = torch.from_numpy(rp).float()
        st = torch.from_numpy(sy_te_np).float()
        ridge_norm_per = ((F.normalize(pt, dim=-1) - F.normalize(st, dim=-1)) ** 2).sum(
            dim=-1
        ).numpy()
        jn = jepa_per_sample_mse_normalized(model, test_loader, device)
        if len(jn) == len(ridge_norm_per):
            results["paired_bootstrap_normalized_vs_ridge"] = {
                "Ridge(s_x + one-hot z), sphere MSE": bootstrap_paired_mse(
                    jn, ridge_norm_per, n_boot=bootstrap_n, seed=bootstrap_seed + 1,
                )
            }

    # Best baseline by different criteria (raw MSE favors scale-matched regressors)
    best_name = min(results["models"], key=lambda k: results["models"][k]["mse"])
    best_norm = min(
        results["models"],
        key=lambda k: results["models"][k]["mse_normalized"],
    )
    results["best_baseline_by_mse"] = {
        "name": best_name,
        "mse": results["models"][best_name]["mse"],
    }
    results["best_baseline_by_mse_normalized"] = {
        "name": best_norm,
        "mse_normalized": results["models"][best_norm]["mse_normalized"],
    }
    return results


@torch.no_grad()
def jepa_normalized_mse(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Global mean squared error on L2-normalized ŝ_y and s_y (same as baselines)."""
    tot_sq, tot_el = 0.0, 0
    for batch in test_loader:
        pred, sy, _ = _forward_batch(model, batch, device)
        pn = F.normalize(pred, dim=-1, eps=1e-8)
        sn = F.normalize(sy, dim=-1, eps=1e-8)
        tot_sq += float(((pn - sn) ** 2).sum().item())
        tot_el += int(pn.numel())
    return tot_sq / max(tot_el, 1)


@torch.no_grad()
def jepa_per_sample_mse_normalized(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Per row: ||normalize(ŝ_y) - normalize(s_y)||^2 (sum over dim)."""
    parts = []
    for batch in test_loader:
        pred, sy, _ = _forward_batch(model, batch, device)
        pn = F.normalize(pred, dim=-1, eps=1e-8)
        sn = F.normalize(sy, dim=-1, eps=1e-8)
        parts.append(((pn - sn) ** 2).sum(dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def jepa_per_sample_mse(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """||ŝ_y - s_y||^2 per row (aligned with test_loader order)."""
    parts = []
    for batch in test_loader:
        pred, sy, _ = _forward_batch(model, batch, device)
        parts.append(((pred - sy) ** 2).sum(dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0)
