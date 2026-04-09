"""Data contract for Version B (paired pre/post) JEPA training — documentation only.

EHR extractors (e.g. All of Us OMOP → tensors) should produce batches that match
these shapes. No tensor data lives here; this documents the pipeline I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PairedSequenceBatchSpec(BaseModel):
    """Expected tensor shapes for one training step (batch-first)."""

    batch_size: int = Field(..., description="B")
    pre_seq_len: int = Field(..., description="T_pre (after padding)")
    post_seq_len: int = Field(..., description="T_post (after padding)")
    num_features: int = Field(..., description="F — fixed vocabulary size")

    # Tensors (described, not stored):
    # pre_values, pre_mask, pre_timestamps  → (B, T_pre, F), (B, T_pre, F), (B, T_pre)
    # post_values, post_mask, post_timestamps → (B, T_post, F), ...
    # intervention (one-hot or soft)        → (B, K)
    # padding_pre, padding_post (optional)    → (B, T_*), True = pad token


class VersionBTrainingManifest(BaseModel):
    """High-level manifest for exported training shards (e.g. Parquet + sidecar)."""

    cohort_id: str = Field(..., description="Phenotype / cohort name")
    index_definition: str = Field(..., description="How index date was defined")
    pre_window_days: int
    post_window_days: int
    washout_days: int
    num_interventions: int = Field(..., description="K — matches model intervention_dim")
    feature_vocab_path: str = Field(..., description="JSON mapping concept_id → column index")
