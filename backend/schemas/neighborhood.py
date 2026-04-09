"""Pydantic models for the Neighborhood RAG frontend payload.

These models structure the final JSON that drives the 3D
neighbourhood visualisation and intervention roadmap UI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TwinRecord(BaseModel):
    """A single historical neighbour plotted on the 3D map."""

    patient_id: str
    relative_coordinates: tuple[float, float, float] = Field(
        ...,
        description="(dx, dy, dz) offset from the user's origin embedding "
        "in the projected 3-D space",
    )
    intervention_taken: str
    outcome_status: str = Field(
        ..., description="'Positive' or 'Negative'"
    )


class RAGIntervention(BaseModel):
    """LLM-generated, twin-backed intervention summary."""

    title: str
    success_rate: float = Field(..., ge=0.0, le=1.0)
    generated_explanation: str = Field(
        ...,
        description="2-paragraph consumer-friendly summary grounded "
        "exclusively in twin outcome data",
    )
    highlighted_twin_ids: list[str] = Field(
        default_factory=list,
        description="Patient IDs of the most relevant supporting twins "
        "for the frontend to visually emphasise",
    )


class NeighborhoodResponse(BaseModel):
    """Top-level payload consumed by the 3D neighbourhood frontend."""

    origin_coordinate: dict[str, float] = Field(
        ..., description="User's projected (x, y, z) in the galaxy"
    )
    neighborhood_twins: list[TwinRecord]
    roadmap: list[RAGIntervention] = Field(
        ...,
        description="Ordered list of intervention recommendations "
        "with LLM-generated explanations",
    )
