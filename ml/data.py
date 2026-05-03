"""
AoU Counterfactual Dataset — robust loader for the npz shard produced by
`notebooks/aou_extraction_pipeline.ipynb` (repo root).

Responsibilities
----------------
- Load the ragged ``pre_*`` / ``post_*`` arrays (object dtype of per-patient 2-D tensors)
- Stratified train/val/test split by intervention
- Per-feature standardization using **training-split** masked mean/std
  (with epsilon floor) — critical because feature scales span HbA1c ~7 → glucose ~140
- Pad variable-length sequences to a fixed ``max_seq_len`` and expose a
  PyTorch-style ``padding_mask`` (True = padded position)
- Returns a flat dict of tensors that plugs directly into
  ``CausalJEPA.forward(...)``

Public API
----------
``build_dataloaders(npz_path, manifest_path, cfg) -> (train_loader, val_loader, test_loader, meta)``

``meta`` carries:
    feature_means, feature_stds    : (F,) np arrays (training-split)
    num_features, num_interventions
    intervention_labels            : list[str] in index order (z=0, 1, 2, ...)
    split_sizes                    : dict[str, int]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class DataConfig:
    npz_path: str = "training_data/patient_tensors.npz"
    manifest_path: str = "training_data/manifest.json"
    intervention_map_path: str = "training_data/intervention_map.json"
    max_seq_len: int = 64          # max observed is 53 weeks
    batch_size: int = 64
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 0


# ----------------------------------------------------------------------
# Stratified split
# ----------------------------------------------------------------------

def _stratified_split(
    intervention_z: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 3-way index split preserving intervention balance."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for z in np.unique(intervention_z):
        ids = np.where(intervention_z == z)[0]
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test_idx.extend(ids[:n_test])
        val_idx.extend(ids[n_test : n_test + n_val])
        train_idx.extend(ids[n_test + n_val :])
    return (
        np.array(sorted(train_idx), dtype=np.int64),
        np.array(sorted(val_idx), dtype=np.int64),
        np.array(sorted(test_idx), dtype=np.int64),
    )


# ----------------------------------------------------------------------
# Masked standardization statistics
# ----------------------------------------------------------------------

def _masked_mean_std(
    values_obj: np.ndarray,
    mask_obj: np.ndarray,
    indices: np.ndarray,
    num_features: int,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std using only observed cells
    (mask == 1) across the given sample indices.
    """
    sums = np.zeros(num_features, dtype=np.float64)
    sq = np.zeros(num_features, dtype=np.float64)
    counts = np.zeros(num_features, dtype=np.float64)

    for i in indices:
        v = values_obj[i].astype(np.float64)
        m = mask_obj[i].astype(np.float64)
        vm = v * m
        sums += vm.sum(axis=0)
        sq += (vm * vm).sum(axis=0)
        counts += m.sum(axis=0)

    safe_counts = np.maximum(counts, 1.0)
    mean = sums / safe_counts
    var = np.maximum(sq / safe_counts - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.where(std < eps, 1.0, std)          # features never observed → unit scale
    mean = np.where(counts == 0, 0.0, mean)
    return mean.astype(np.float32), std.astype(np.float32)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class AoUJepaDataset(Dataset):
    """In-memory dataset; pads every sample to ``max_seq_len`` and
    standardizes values in observed positions only (mask==0 positions stay 0).
    """

    def __init__(
        self,
        npz: np.lib.npyio.NpzFile,
        indices: np.ndarray,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        max_seq_len: int,
    ):
        self.indices = indices
        self.max_seq_len = max_seq_len
        self.num_features = feature_means.shape[0]

        self.mean = feature_means.astype(np.float32)         # (F,)
        self.std = feature_stds.astype(np.float32)           # (F,)

        # Time normalization: convert seconds → weeks (matches weekly resolution)
        self.time_scale = 1.0 / (7 * 24 * 3600.0)

        self.pre_values = npz["pre_values"]
        self.pre_mask = npz["pre_mask"]
        self.pre_timestamps = npz["pre_timestamps"]
        self.post_values = npz["post_values"]
        self.post_mask = npz["post_mask"]
        self.post_timestamps = npz["post_timestamps"]
        self.intervention_z = npz["intervention_z"]
        self.person_ids = npz["person_ids"]

    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------

    def _prep_window(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Standardize → pad → produce padding mask.

        Returns (padded_values, padded_mask, padded_timestamps_weeks, padding_mask_bool).
        """
        T, F = values.shape
        T_eff = min(T, self.max_seq_len)

        # Standardize observed positions only; missing cells remain 0.
        v = values[:T_eff].astype(np.float32)
        m = mask[:T_eff].astype(np.float32)
        v_norm = np.where(m > 0, (v - self.mean) / self.std, 0.0).astype(np.float32)

        pv = np.zeros((self.max_seq_len, F), dtype=np.float32)
        pm = np.zeros((self.max_seq_len, F), dtype=np.float32)
        pt = np.zeros(self.max_seq_len, dtype=np.float32)

        pv[:T_eff] = v_norm
        pm[:T_eff] = m

        # Timestamps: seconds → weeks, positioned relative to first step.
        ts_weeks = timestamps[:T_eff].astype(np.float32) * self.time_scale
        if T_eff > 0:
            ts_weeks = ts_weeks - ts_weeks[0]       # start at 0 per window
        pt[:T_eff] = ts_weeks

        padding_mask = np.ones(self.max_seq_len, dtype=bool)
        padding_mask[:T_eff] = False                 # False = real, True = padded
        if T_eff == 0:                               # safety: at least one valid slot
            padding_mask[0] = False

        return pv, pm, pt, padding_mask

    # ------------------------------------------------------------------

    def __getitem__(self, i: int) -> dict:
        idx = int(self.indices[i])

        pv, pm, pt, ppad = self._prep_window(
            self.pre_values[idx], self.pre_mask[idx], self.pre_timestamps[idx]
        )
        tv, tm, tt, tpad = self._prep_window(
            self.post_values[idx], self.post_mask[idx], self.post_timestamps[idx]
        )

        return {
            "context_x": torch.from_numpy(pv),
            "context_mask": torch.from_numpy(pm),
            "context_timestamps": torch.from_numpy(pt),
            "context_padding_mask": torch.from_numpy(ppad),
            "target_y": torch.from_numpy(tv),
            "target_mask": torch.from_numpy(tm),
            "target_timestamps": torch.from_numpy(tt),
            "target_padding_mask": torch.from_numpy(tpad),
            "intervention_z": torch.tensor(int(self.intervention_z[idx]), dtype=torch.long),
            "person_id": torch.tensor(int(self.person_ids[idx]), dtype=torch.long),
        }


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------

def build_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Load the npz, produce train/val/test DataLoaders, return metadata."""
    npz_path = Path(cfg.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Training shard not found: {npz_path}")

    manifest = json.loads(Path(cfg.manifest_path).read_text())
    imap = json.loads(Path(cfg.intervention_map_path).read_text())

    # Deterministic label order by z-index
    z_to_label = {imap["concept_to_z"][k]: imap["labels"][k] for k in imap["concept_to_z"]}
    labels = [z_to_label[z] for z in sorted(z_to_label.keys())]

    num_features = int(manifest["num_features"])
    num_interventions = int(manifest["num_interventions"])

    npz = np.load(npz_path, allow_pickle=True)
    intervention_z = npz["intervention_z"]
    n = len(intervention_z)

    train_idx, val_idx, test_idx = _stratified_split(
        intervention_z, cfg.val_fraction, cfg.test_fraction, cfg.seed
    )

    feature_means, feature_stds = _masked_mean_std(
        npz["pre_values"], npz["pre_mask"], train_idx, num_features=num_features
    )

    ds_train = AoUJepaDataset(npz, train_idx, feature_means, feature_stds, cfg.max_seq_len)
    ds_val = AoUJepaDataset(npz, val_idx, feature_means, feature_stds, cfg.max_seq_len)
    ds_test = AoUJepaDataset(npz, test_idx, feature_means, feature_stds, cfg.max_seq_len)

    # pin_memory when cuda available
    pin = torch.cuda.is_available()
    common = dict(num_workers=cfg.num_workers, pin_memory=pin)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **common)

    meta = {
        "num_patients": n,
        "num_features": num_features,
        "num_interventions": num_interventions,
        "intervention_labels": labels,
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "train_intervention_counts": np.bincount(
            intervention_z[train_idx], minlength=num_interventions
        ).tolist(),
        "val_intervention_counts": np.bincount(
            intervention_z[val_idx], minlength=num_interventions
        ).tolist(),
        "test_intervention_counts": np.bincount(
            intervention_z[test_idx], minlength=num_interventions
        ).tolist(),
        "max_seq_len": cfg.max_seq_len,
    }
    return train_loader, val_loader, test_loader, meta
