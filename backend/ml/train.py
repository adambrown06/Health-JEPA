"""
CausalJEPA Training Loop — ARCHITECTURE OUTLINE (not yet runnable).

This file documents the exact training contract for the Action-Conditional
Time-Series JEPA.  It will be fleshed out once the AoU extraction pipeline
has produced real training shards.

Training Protocol
-----------------
1.  loss = F.mse_loss(predicted_s_y, s_y)       — latent prediction loss
2.  loss.backward()                              — gradients only through context_encoder,
                                                    action_embedding, and predictor
3.  optimizer.step()                             — updates those three parameter groups
4.  model.update_target_encoder(momentum=0.99)   — EMA sync (no gradient)

The TargetEncoder's parameters have requires_grad=False, so the optimizer
must be constructed over only the trainable parameter groups.

Data Contract
-------------
Each batch from the DataLoader yields:
    context_x      : FloatTensor  (B, T_pre,  F)   — pre-intervention clinical time-series
    target_y       : FloatTensor  (B, T_post, F)   — post-intervention clinical time-series
    intervention_z : LongTensor   (B,)              — categorical action ID (0, 1, …, K−1)

Optional per-batch tensors (for irregular sampling / variable-length sequences):
    context_mask, target_mask             : FloatTensor (B, T, F) — 1=observed, 0=missing
    context_timestamps, target_timestamps : FloatTensor (B, T)    — seconds since window start
    context_padding_mask, target_padding_mask : BoolTensor (B, T) — True=padded (PyTorch convention)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ml.jepa_model import CausalJEPA


# ======================================================================
# Dataset — loads the .npz shards produced by the AoU extraction notebook
# ======================================================================

class AoUCounterfactualDataset(Dataset):
    """Reads the compressed tensor archive exported by
    ``notebooks/aou_extraction_pipeline.ipynb`` and yields padded
    (context_x, target_y, intervention_z) tuples.

    Variable-length sequences are zero-padded to the longest sequence
    in the shard; per-batch collation will re-pad to the batch maximum.
    """

    def __init__(self, npz_path: str | Path, max_seq_len: int = 128):
        data = np.load(npz_path, allow_pickle=True)
        self.person_ids = data["person_ids"]
        self.intervention_z = data["intervention_z"]
        self.pre_values = data["pre_values"]
        self.pre_mask = data["pre_mask"]
        self.pre_timestamps = data["pre_timestamps"]
        self.post_values = data["post_values"]
        self.post_mask = data["post_mask"]
        self.post_timestamps = data["post_timestamps"]
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.person_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Pad / truncate to ``max_seq_len`` and return tensors."""
        pre_v = self._pad_2d(self.pre_values[idx])
        pre_m = self._pad_2d(self.pre_mask[idx])
        pre_t = self._pad_1d(self.pre_timestamps[idx])
        post_v = self._pad_2d(self.post_values[idx])
        post_m = self._pad_2d(self.post_mask[idx])
        post_t = self._pad_1d(self.post_timestamps[idx])

        return {
            "context_x": torch.from_numpy(pre_v).float(),
            "context_mask": torch.from_numpy(pre_m).float(),
            "context_timestamps": torch.from_numpy(pre_t).float(),
            "target_y": torch.from_numpy(post_v).float(),
            "target_mask": torch.from_numpy(post_m).float(),
            "target_timestamps": torch.from_numpy(post_t).float(),
            "intervention_z": torch.tensor(int(self.intervention_z[idx]), dtype=torch.long),
        }

    def _pad_2d(self, arr: np.ndarray) -> np.ndarray:
        T, F = arr.shape
        if T >= self.max_seq_len:
            return arr[: self.max_seq_len]
        padded = np.zeros((self.max_seq_len, F), dtype=arr.dtype)
        padded[:T] = arr
        return padded

    def _pad_1d(self, arr: np.ndarray) -> np.ndarray:
        T = arr.shape[0]
        if T >= self.max_seq_len:
            return arr[: self.max_seq_len]
        padded = np.zeros(self.max_seq_len, dtype=np.float32)
        padded[:T] = arr.astype(np.float32)
        return padded


# ======================================================================
# Training Configuration Outline
# ======================================================================

class TrainingConfig:
    """All hyper-parameters in one place (will become CLI / Hydra later)."""
    npz_path: str = "training_data/patient_tensors.npz"
    manifest_path: str = "training_data/manifest.json"
    num_features: int = 32
    num_interventions: int = 2
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    z_dim: int = 32
    predictor_hidden: int = 512
    predictor_layers: int = 3
    dropout: float = 0.1
    ema_momentum: float = 0.99
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_seq_len: int = 128
    num_epochs: int = 100
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Training Loop (OUTLINE — not yet invoked)
# ======================================================================

def build_model(cfg: TrainingConfig) -> CausalJEPA:
    return CausalJEPA(
        num_features=cfg.num_features,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        num_interventions=cfg.num_interventions,
        z_dim=cfg.z_dim,
        predictor_hidden=cfg.predictor_hidden,
        predictor_layers=cfg.predictor_layers,
        ema_momentum=cfg.ema_momentum,
    )


def build_optimizer(model: CausalJEPA, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Only optimise the three trainable parameter groups.
    The TargetEncoder has requires_grad=False so filter(requires_grad) excludes it."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def train_one_step(
    model: CausalJEPA,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
) -> float:
    """Single training step — documents the exact gradient / EMA protocol.

    Steps
    -----
    1. Forward pass  → (predicted_s_y, s_y)
    2. loss = MSE(predicted_s_y, s_y)
    3. loss.backward()          — grads flow through context_encoder, action_embedding, predictor
    4. optimizer.step()         — updates those three groups only
    5. model.update_target_encoder(momentum)  — EMA sync of target_encoder (no grad)
    """
    model.train()
    optimizer.zero_grad()

    predicted_s_y, s_y = model(
        context_x=batch["context_x"],
        target_y=batch["target_y"],
        intervention_z=batch["intervention_z"],
        context_mask=batch.get("context_mask"),
        target_mask=batch.get("target_mask"),
        context_timestamps=batch.get("context_timestamps"),
        target_timestamps=batch.get("target_timestamps"),
    )

    loss = F.mse_loss(predicted_s_y, s_y)
    loss.backward()
    optimizer.step()
    model.update_target_encoder(momentum=cfg.ema_momentum)

    return loss.item()


# ======================================================================
# Entry point (placeholder — will be activated once data is available)
# ======================================================================

def train(cfg: Optional[TrainingConfig] = None) -> None:
    """Full training loop.  NOT YET IMPLEMENTED — this is the outline.

    Pseudo-code:
        dataset  = AoUCounterfactualDataset(cfg.npz_path, cfg.max_seq_len)
        loader   = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        model    = build_model(cfg).to(cfg.device)
        optimizer = build_optimizer(model, cfg)

        for epoch in range(cfg.num_epochs):
            for batch in loader:
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                loss = train_one_step(model, batch, optimizer, cfg)
            # checkpoint, logging, validation …
    """
    raise NotImplementedError(
        "Training loop is outlined but not yet activated.  "
        "Run the AoU extraction notebook first to produce training shards, "
        "then uncomment and invoke this function."
    )


if __name__ == "__main__":
    train()
