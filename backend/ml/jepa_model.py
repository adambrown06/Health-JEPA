"""
Action-Conditional Time-Series JEPA (Joint-Embedding Predictive Architecture).

Architecture
------------
CausalJEPA composes three sub-networks to learn causal clinical trajectories
in a shared latent space **without reconstructing the raw input data**:

    ContextEncoder  – Transformer over pre-intervention time-series   → s_x
    TargetEncoder   – Structurally identical; weights = EMA(Context)  → s_y  (stop-gradient)
    Predictor       – MLP mapping [s_x ‖ embed(z)] → predicted ŝ_y

The TargetEncoder never receives gradients.  After each optimizer step the
caller invokes ``update_target_encoder(momentum)`` to apply the exponential
moving-average update:  θ_target ← m·θ_target + (1−m)·θ_context.

References
----------
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture," CVPR 2023.
- Adapted for irregular, multi-variate clinical time-series with continuous
  temporal positional encodings and explicit missing-value masks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Building Blocks
# ======================================================================

class ContinuousTemporalEncoding(nn.Module):
    """Sinusoidal positional encoding driven by *actual elapsed seconds*
    rather than integer positions — critical for irregularly sampled
    clinical data (labs every 90 days vs. wearables every 5 minutes).
    """

    def __init__(self, d_model: int, max_period: float = 1e6):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (
            max_period ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timestamps : (batch, seq_len)  — seconds elapsed since t₀.

        Returns
        -------
        (batch, seq_len, d_model) positional embedding.
        """
        t = timestamps.unsqueeze(-1).float()              # (B, T, 1)
        sinusoid = t * self.inv_freq.unsqueeze(0).unsqueeze(0)  # (B, T, d//2)
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)


class InputProjection(nn.Module):
    """Projects raw clinical features + missing-value mask into d_model,
    then adds the continuous temporal positional encoding."""

    def __init__(self, num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(num_features * 2, d_model)
        self.temporal_enc = ContinuousTemporalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        values     : (B, T, F)  clinical values (NaN replaced with 0 before entry)
        mask       : (B, T, F)  1=observed, 0=missing
        timestamps : (B, T)     seconds since t₀
        """
        x = torch.cat([values * mask, mask], dim=-1)
        x = self.linear(x)
        x = x + self.temporal_enc(timestamps)
        return self.dropout(self.layer_norm(x))


# ======================================================================
# Time-Series Encoder (shared architecture for Context & Target)
# ======================================================================

class TimeSeriesEncoder(nn.Module):
    """Transformer encoder stack shared structurally by Context and Target."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = InputProjection(num_features, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        timestamps: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        seq_out : (B, T, d_model) — full sequence representations
        pooled  : (B, d_model)    — mean-pooled embedding
        """
        x = self.input_proj(values, mask, timestamps)
        seq_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            active = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (seq_out * active).sum(dim=1) / active.sum(dim=1).clamp(min=1.0)
        else:
            pooled = seq_out.mean(dim=1)

        pooled = self.pool_proj(pooled)
        return seq_out, pooled


# Backward-compatible aliases used by clinical_jepa_vb.py
ContextEncoder = TimeSeriesEncoder
TransformerEncoder = TimeSeriesEncoder


class TargetEncoder(TimeSeriesEncoder):
    """Structurally identical to TimeSeriesEncoder.
    Weights are updated exclusively via EMA — no gradients ever flow here."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for p in self.parameters():
            p.requires_grad = False


# ======================================================================
# Predictor — AdaLN-conditioned (DiT-style, LeWorldModel-style)
# ======================================================================

class AdaLN(nn.Module):
    """Adaptive Layer Normalization (Peebles & Xie 2023, "Scalable Diffusion
    Models with Transformers"; LeWorldModel 2026).

    Replaces LayerNorm's fixed (gamma, beta) affine parameters with per-sample
    parameters produced from a conditioning vector ``c`` (here: the intervention
    embedding z):

        AdaLN(h, c) = (1 + gamma(c)) * (h - mean) / std + beta(c)

    Initialized AdaLN-Zero style: (gamma, beta) projections start at 0 so the
    normalization is an identity at step 0 (predictor behaves as if z is ignored)
    and the model must *learn* to exploit z-conditioning as it benefits training.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        gamma, beta = self.cond_proj(c).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class AdaLNBlock(nn.Module):
    """Residual MLP block with AdaLN conditioning and a zero-init residual gate:

        h' = h + alpha(c) * MLP( AdaLN(h, c) )

    ``alpha(c)`` is initialized to 0 (AdaLN-Zero), so the block is an identity
    map at initialization. This keeps training stable and lets the optimizer
    discover when to let z influence the predictor's activations.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.adaln = AdaLN(hidden_dim, cond_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        residual = h
        h = self.adaln(h, c)
        h = self.drop(self.act(self.fc(h)))
        alpha = self.gate(c)
        return residual + alpha * h


class ConcatPredictor(nn.Module):
    """Vanilla concat-MLP predictor (the pre-AdaLN baseline).

        h = concat(s_x, embed(z))  →  MLP  →  ŝ_y

    Kept as a clean ablation to show the effect of AdaLN conditioning.
    """

    def __init__(
        self,
        d_model: int = 256,
        z_dim: int = 32,
        hidden_dim: int = 512,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_model + z_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(n_hidden_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, s_x: torch.Tensor, embedded_z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_x, embedded_z], dim=-1))


class Predictor(nn.Module):
    """Intervention-conditioned predictor:

        s_x ─► input_proj ─► [AdaLNBlock(z)] × n ─► AdaLN(z) ─► output_proj ─► ŝ_y

    Instead of the old ``concat(s_x, z) → MLP`` design, z modulates *every*
    predictor layer via AdaLN. This is much stronger conditioning: the 3 drug
    embeddings can't get "averaged out" through a wide hidden layer because
    they control that layer's normalization statistics directly.
    """

    def __init__(
        self,
        d_model: int = 256,
        z_dim: int = 32,
        hidden_dim: int = 512,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_model, hidden_dim)
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim, cond_dim=z_dim, dropout=dropout)
            for _ in range(n_hidden_layers)
        ])
        self.final_adaln = AdaLN(hidden_dim, cond_dim=z_dim)
        self.output_proj = nn.Linear(hidden_dim, d_model)

    def forward(self, s_x: torch.Tensor, embedded_z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        s_x        : (B, d_model)  context embedding from ContextEncoder
        embedded_z : (B, z_dim)    dense intervention embedding

        Returns
        -------
        (B, d_model) predicted future-state embedding ŝ_y
        """
        h = self.input_proj(s_x)
        for block in self.blocks:
            h = block(h, embedded_z)
        h = self.final_adaln(h, embedded_z)
        return self.output_proj(h)


# ======================================================================
# CausalJEPA — Top-Level Module
# ======================================================================

class CausalJEPA(nn.Module):
    """Action-Conditional Time-Series JEPA.

    Composes:
      - context_encoder  (learns via backprop)
      - target_encoder   (stop-gradient; updated via EMA only)
      - action_embedding (nn.Embedding mapping categorical z → dense vector)
      - predictor        (MLP predicting target embedding from context + action)
    """

    def __init__(
        self,
        num_features: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_interventions: int = 8,
        z_dim: int = 32,
        predictor_hidden: int = 512,
        predictor_layers: int = 3,
        predictor_style: str = "adaln",  # "adaln" | "concat"
        ema_momentum: float = 0.99,
    ):
        super().__init__()
        if predictor_style not in {"adaln", "concat"}:
            raise ValueError(f"predictor_style must be 'adaln' or 'concat', got {predictor_style!r}")
        self.predictor_style = predictor_style

        encoder_kwargs = dict(
            num_features=num_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # ── Encoders ──────────────────────────────────────────────────
        self.context_encoder = TimeSeriesEncoder(**encoder_kwargs)
        self.target_encoder = TargetEncoder(**encoder_kwargs)
        self._init_target_from_context()

        # ── Intervention embedding (categorical → dense) ─────────────
        self.action_embedding = nn.Embedding(
            num_embeddings=num_interventions,
            embedding_dim=z_dim,
        )

        # ── Predictor  [s_x ‖ embed(z)] → ŝ_y ───────────────────────
        predictor_cls = Predictor if predictor_style == "adaln" else ConcatPredictor
        self.predictor = predictor_cls(
            d_model=d_model,
            z_dim=z_dim,
            hidden_dim=predictor_hidden,
            n_hidden_layers=predictor_layers,
            dropout=dropout,
        )

        self.ema_momentum = ema_momentum
        self.d_model = d_model
        self.num_interventions = num_interventions
        self.z_dim = z_dim

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def _init_target_from_context(self) -> None:
        """Cold-start: copy context weights into target verbatim."""
        for tp, cp in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            tp.data.copy_(cp.data)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """Exponential moving-average update:
            θ_target ← m · θ_target + (1 − m) · θ_context
        """
        m = momentum if momentum is not None else self.ema_momentum
        for tp, cp in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            tp.data.mul_(m).add_(cp.data, alpha=1.0 - m)

    # ------------------------------------------------------------------
    # Forward (training) — exact spec from the architecture document
    # ------------------------------------------------------------------

    def forward(
        self,
        context_x: torch.Tensor,
        target_y: torch.Tensor,
        intervention_z: torch.LongTensor,
        context_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        context_timestamps: Optional[torch.Tensor] = None,
        target_timestamps: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        context_x          : (B, T_pre, F)  pre-intervention clinical values
        target_y           : (B, T_post, F) post-intervention clinical values
        intervention_z     : (B,)           LongTensor of categorical action IDs
        context_mask       : (B, T_pre, F)  observation mask (1=observed). Defaults to all-1.
        target_mask        : (B, T_post, F) observation mask. Defaults to all-1.
        context_timestamps : (B, T_pre)     seconds since window start. Defaults to arange.
        target_timestamps  : (B, T_post)    seconds since window start. Defaults to arange.
        *_padding_mask     : (B, T)         True = padded position (PyTorch convention)

        Returns
        -------
        predicted_s_y : (B, d_model) — predictor output
        s_y           : (B, d_model) — target anchor (detached, no grad)
        """
        B = context_x.shape[0]
        device = context_x.device

        if context_mask is None:
            context_mask = torch.ones_like(context_x)
        if target_mask is None:
            target_mask = torch.ones_like(target_y)
        if context_timestamps is None:
            context_timestamps = torch.arange(
                context_x.shape[1], device=device, dtype=torch.float
            ).unsqueeze(0).expand(B, -1)
        if target_timestamps is None:
            target_timestamps = torch.arange(
                target_y.shape[1], device=device, dtype=torch.float
            ).unsqueeze(0).expand(B, -1)

        # 1) Encode context  → s_x   (gradients flow)
        _, s_x = self.context_encoder(
            context_x, context_mask, context_timestamps,
            src_key_padding_mask=context_padding_mask,
        )

        # 2) Encode target   → s_y   (STOP-GRADIENT)
        with torch.no_grad():
            _, s_y = self.target_encoder(
                target_y, target_mask, target_timestamps,
                src_key_padding_mask=target_padding_mask,
            )
        s_y = s_y.detach()

        # 3) Embed the intervention
        embedded_z = self.action_embedding(intervention_z)  # (B, z_dim)

        # 4-5) Predict the target embedding from [s_x ‖ embedded_z]
        predicted_s_y = self.predictor(s_x, embedded_z)

        # 6) Return both for the caller to compute MSE loss
        return predicted_s_y, s_y

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a full (unmasked) time-series into its latent embedding."""
        B = values.shape[0]
        device = values.device
        if mask is None:
            mask = torch.ones_like(values)
        if timestamps is None:
            timestamps = torch.arange(
                values.shape[1], device=device, dtype=torch.float
            ).unsqueeze(0).expand(B, -1)
        _, pooled = self.context_encoder(
            values, mask, timestamps, src_key_padding_mask=padding_mask,
        )
        return pooled

    @torch.no_grad()
    def predict_counterfactual(
        self,
        origin_embedding: torch.Tensor,
        intervention_z: torch.LongTensor,
    ) -> torch.Tensor:
        """Given an origin embedding and intervention index, predict the
        future-state embedding (counterfactual trajectory in latent space)."""
        embedded_z = self.action_embedding(intervention_z)
        return self.predictor(origin_embedding, embedded_z)


# Keep the old name importable for any code that references it
ClinicalJEPA = CausalJEPA
