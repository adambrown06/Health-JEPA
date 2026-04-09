"""Pydantic models for the inference / async-job API layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl

from schemas.neighborhood import NeighborhoodResponse
from schemas.patient import ClinicalTimeSeries, PatientDemographics


# ------------------------------------------------------------------
# Request
# ------------------------------------------------------------------
class InferenceJobRequest(BaseModel):
    """Payload submitted by the frontend to kick off a full analysis."""

    clinical_time_series: ClinicalTimeSeries
    demographics: Optional[PatientDemographics] = None
    webhook_url: Optional[HttpUrl] = Field(
        None,
        description="If provided, the worker POSTs the result here on completion",
    )
    callback_job_id: Optional[str] = Field(
        None,
        description="Client-generated correlation ID; echoed back in the callback",
    )
    requested_interventions: Optional[list[str]] = Field(
        None,
        description="Subset of intervention codes to simulate; None = run all",
    )


# ------------------------------------------------------------------
# Response — immediate (enqueue acknowledgement)
# ------------------------------------------------------------------
class InferenceJobAck(BaseModel):
    """Returned synchronously by the /analyze endpoint."""

    job_id: str
    status: str = "queued"
    poll_url: str = Field(
        ..., description="GET this URL to check status / retrieve result"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ------------------------------------------------------------------
# Response — final (returned by the worker / polling endpoint)
# ------------------------------------------------------------------
class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulatedTrajectory(BaseModel):
    """One counterfactual trajectory for a single intervention."""

    intervention_id: str
    intervention_name: str
    predicted_embedding: list[float]
    predicted_coordinate: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    delta_risk_score: float = Field(
        ..., description="Negative = risk reduction, positive = risk increase"
    )


class TwinMatch(BaseModel):
    """A real historical patient matched via vector similarity."""

    patient_id: str
    similarity: float
    intervention_taken: str
    actual_outcome: str
    outcome_months: int
    coordinate: dict[str, float]


class RankedIntervention(BaseModel):
    """Final ranked recommendation combining JEPA prediction + twin validation."""

    rank: int
    intervention_id: str
    intervention_name: str
    jepa_confidence: float
    twin_concordance: float = Field(
        ...,
        description="Fraction of matched twins where real outcome aligned "
        "with JEPA prediction direction",
    )
    combined_score: float
    simulated_trajectory: SimulatedTrajectory
    supporting_twins: list[TwinMatch]


class AnalysisResult(BaseModel):
    """Complete output of the async analysis pipeline."""

    job_id: str
    status: JobStatus = JobStatus.COMPLETED
    origin_embedding: list[float]
    origin_coordinate: dict[str, float]
    cluster_id: int
    cluster_name: str
    ranked_interventions: list[RankedIntervention]
    narrative_summary: str = Field(
        ..., description="LLM-generated plain-language RAG synthesis"
    )
    neighborhood: Optional[NeighborhoodResponse] = Field(
        None,
        description="3D neighbourhood payload with twin records and "
        "RAG-generated intervention roadmap",
    )
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class InferenceJobStatus(BaseModel):
    """Polling response wrapping either progress or the final result."""

    job_id: str
    status: JobStatus
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
