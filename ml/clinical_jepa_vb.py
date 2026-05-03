"""
Version B — Paired pre/post + intervention JEPA (architecture only).

Data contract (per example)
---------------------------
- *Pre* window: multivariate clinical series before index (labs / vitals / wearables).
- *Post* window: multivariate series after washout (outcomes / follow-up).
- *Intervention* z: one-hot (or soft) vector over K intervention classes.

Forward (training)
------------------
- Encode pre  → s_x
- Encode post → s_y  (stop-gradient on s_y so the target path does not
  backprop into the encoder from the post branch — shared-encoder variant;
  dual-EMA variant uses a frozen target stack instead.)
- Predictor(s_x, z) → ŝ_y
- Loss (implemented outside this module): e.g. MSE(ŝ_y, stopgrad(s_y)).

Two variants are provided:
- ``ClinicalJEPAB``: single shared ``TransformerEncoder`` for pre and post.
- ``ClinicalJEPABDualEMA``: context encoder on pre, EMA target encoder on post
  (same structural pattern as ``ClinicalJEPA`` in ``jepa_model.py``, but with
  separate tensors instead of masks on one sequence).

Inference
---------
- ``encode_pre`` / ``encode_post`` (or context encoder only) for Qdrant / API.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ml.jepa_model import (
    ContextEncoder,
    Predictor,
    TargetEncoder,
    TimeSeriesEncoder,
    TransformerEncoder,
)


class ClinicalJEPAB(nn.Module):
    """Shared-encoder Version B: one encoder for pre and post; s_y detached."""

    def __init__(
        self,
        num_features: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_interventions: int = 8,
        predictor_hidden: int = 512,
        predictor_layers: int = 3,
    ):
        super().__init__()
        enc_kw = dict(
            num_features=num_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.encoder = TransformerEncoder(**enc_kw)
        self.predictor = Predictor(
            d_model=d_model,
            intervention_dim=num_interventions,
            hidden_dim=predictor_hidden,
            n_hidden_layers=predictor_layers,
            dropout=dropout,
        )
        self.d_model = d_model
        self.num_interventions = num_interventions

    def forward_train(
        self,
        pre_values: torch.Tensor,
        pre_mask: torch.Tensor,
        pre_timestamps: torch.Tensor,
        post_values: torch.Tensor,
        post_mask: torch.Tensor,
        post_timestamps: torch.Tensor,
        intervention: torch.Tensor,
        padding_pre: Optional[torch.Tensor] = None,
        padding_post: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pre_* / post_* : (B, T_pre/post, F) values, masks, seconds since window start
        intervention   : (B, K)
        padding_*      : (B, T) True = pad (PyTorch convention), optional
        """
        _, s_x = self.encoder(
            pre_values, pre_mask, pre_timestamps, src_key_padding_mask=padding_pre
        )
        with torch.no_grad():
            _, s_y_raw = self.encoder(
                post_values, post_mask, post_timestamps, src_key_padding_mask=padding_post
            )
        s_y = s_y_raw.detach()
        s_y_hat = self.predictor(s_x, intervention)
        return {"s_x": s_x, "s_y": s_y, "s_y_hat": s_y_hat}

    @torch.no_grad()
    def encode_pre(
        self,
        pre_values: torch.Tensor,
        pre_mask: torch.Tensor,
        pre_timestamps: torch.Tensor,
        padding_pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pooled embedding for the pre-index window (deployment / Qdrant)."""
        _, pooled = self.encoder(
            pre_values, pre_mask, pre_timestamps, src_key_padding_mask=padding_pre
        )
        return pooled


class ClinicalJEPABDualEMA(nn.Module):
    """Dual-stack Version B: context encoder on pre, EMA target encoder on post."""

    def __init__(
        self,
        num_features: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_interventions: int = 8,
        predictor_hidden: int = 512,
        predictor_layers: int = 3,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        enc_kw = dict(
            num_features=num_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.context_encoder = ContextEncoder(**enc_kw)
        self.target_encoder = TargetEncoder(**enc_kw)
        self._init_target_from_context()

        self.predictor = Predictor(
            d_model=d_model,
            intervention_dim=num_interventions,
            hidden_dim=predictor_hidden,
            n_hidden_layers=predictor_layers,
            dropout=dropout,
        )
        self.ema_decay = ema_decay
        self.d_model = d_model
        self.num_interventions = num_interventions

    def _init_target_from_context(self) -> None:
        for tp, cp in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            tp.data.copy_(cp.data)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for tp, cp in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            tp.data.mul_(self.ema_decay).add_(cp.data, alpha=1.0 - self.ema_decay)

    def forward_train(
        self,
        pre_values: torch.Tensor,
        pre_mask: torch.Tensor,
        pre_timestamps: torch.Tensor,
        post_values: torch.Tensor,
        post_mask: torch.Tensor,
        post_timestamps: torch.Tensor,
        intervention: torch.Tensor,
        padding_pre: Optional[torch.Tensor] = None,
        padding_post: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        _, s_x = self.context_encoder(
            pre_values, pre_mask, pre_timestamps, src_key_padding_mask=padding_pre
        )
        with torch.no_grad():
            _, s_y = self.target_encoder(
                post_values, post_mask, post_timestamps, src_key_padding_mask=padding_post
            )
        s_y_hat = self.predictor(s_x, intervention)
        return {"s_x": s_x, "s_y": s_y.detach(), "s_y_hat": s_y_hat}

    @torch.no_grad()
    def encode_pre(
        self,
        pre_values: torch.Tensor,
        pre_mask: torch.Tensor,
        pre_timestamps: torch.Tensor,
        padding_pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, pooled = self.context_encoder(
            pre_values, pre_mask, pre_timestamps, src_key_padding_mask=padding_pre
        )
        return pooled


def paired_jepa_loss(
    s_y_hat: torch.Tensor,
    s_y: torch.Tensor,
) -> torch.Tensor:
    """L2 latent prediction loss; call after forward_train. ``s_y`` must be detached upstream."""
    return nn.functional.mse_loss(s_y_hat, s_y)
