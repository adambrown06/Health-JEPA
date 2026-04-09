"""Pydantic models for clinical time-series patient data (OMOP/FHIR/wearable)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


class DataSource(str, Enum):
    OMOP = "omop"
    FHIR = "fhir"
    WEARABLE = "wearable"
    LAB = "lab"


class ClinicalObservation(BaseModel):
    """Single clinical measurement at one point in time."""

    timestamp: datetime
    concept_id: str = Field(
        ..., description="OMOP concept_id or FHIR Observation.code"
    )
    value: Optional[float] = Field(
        None,
        description="Numeric value; None signals a structurally missing reading",
    )
    unit: Optional[str] = None
    source: DataSource = DataSource.OMOP
    is_missing: bool = Field(
        default=False,
        description="Explicit flag so downstream imputation can distinguish "
        "true NaN from a zero-valued measurement",
    )

    @model_validator(mode="after")
    def _sync_missing_flag(self) -> "ClinicalObservation":
        if self.value is None:
            object.__setattr__(self, "is_missing", True)
        return self


class ClinicalTimeSeries(BaseModel):
    """
    Ordered, multivariate clinical time-series for a single patient.

    Handles irregular sampling and explicit NaN tracking so the JEPA encoder
    can apply learned imputation masks rather than naive zero-fill.
    """

    patient_id: str
    observations: list[ClinicalObservation] = Field(..., min_length=1)
    feature_names: list[str] = Field(
        ...,
        description="Ordered feature column names for tensor alignment "
        "(e.g. ['hba1c', 'fasting_glucose', 'rhr', 'hrv', ...])",
    )
    sequence_length: int = Field(
        ...,
        gt=0,
        description="Number of discrete time-steps in the padded tensor",
    )
    sampling_rate_seconds: Optional[float] = Field(
        None,
        description="Expected cadence between observations; None = irregular",
    )

    age: Optional[int] = None
    sex: Optional[str] = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def has_irregular_timestamps(self) -> bool:
        if len(self.observations) < 2:
            return False
        deltas = [
            (self.observations[i + 1].timestamp - self.observations[i].timestamp).total_seconds()
            for i in range(len(self.observations) - 1)
        ]
        if self.sampling_rate_seconds is not None:
            tolerance = self.sampling_rate_seconds * 0.1
            return any(abs(d - self.sampling_rate_seconds) > tolerance for d in deltas)
        return len(set(round(d) for d in deltas)) > 1

    def to_tensor_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build padded arrays suitable for PyTorch ingestion.

        Returns
        -------
        values : ndarray, shape (sequence_length, num_features)
            Clinical values with NaN for missing.
        mask : ndarray, shape (sequence_length, num_features)
            1.0 where observed, 0.0 where missing.
        timestamps : ndarray, shape (sequence_length,)
            Seconds elapsed since the first observation (continuous positional
            signal for the temporal encoder).
        """
        num_features = len(self.feature_names)
        feature_index = {name: i for i, name in enumerate(self.feature_names)}

        values = np.full(
            (self.sequence_length, num_features), fill_value=np.nan, dtype=np.float32
        )
        mask = np.zeros((self.sequence_length, num_features), dtype=np.float32)
        timestamps = np.zeros(self.sequence_length, dtype=np.float64)

        t0 = self.observations[0].timestamp
        for step, obs in enumerate(self.observations[: self.sequence_length]):
            col = feature_index.get(obs.concept_id)
            if col is not None and not obs.is_missing:
                values[step, col] = obs.value  # type: ignore[assignment]
                mask[step, col] = 1.0
            timestamps[step] = (obs.timestamp - t0).total_seconds()

        return values, mask, timestamps


class PatientDemographics(BaseModel):
    """Lightweight demographics subset used by the ranking layer."""

    patient_id: str
    age: int
    sex: str
    bmi: Optional[float] = None
    primary_risk_label: Optional[str] = None
