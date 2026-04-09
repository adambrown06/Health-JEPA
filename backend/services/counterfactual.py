"""
Counterfactual Causal Engine.

Ties JEPA inference to the Vector DB:
1. Simulate future trajectories per intervention through the Predictor.
2. Retrieve real-world twins from Qdrant.
3. Compare theoretical predictions against historical ground truth.
4. Deterministically rank interventions by safety and concordance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from config import settings
from ml.jepa_model import ClinicalJEPA
from services.vector_store import VectorDBClient

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Intervention registry — extend as new protocols are validated
# ------------------------------------------------------------------
@dataclass
class InterventionSpec:
    id: str
    name: str
    index: int  # position in the one-hot vector
    description: str = ""


DEFAULT_INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec("glp1", "GLP-1 Receptor Agonist (Semaglutide)", 0,
                     "Initiate semaglutide 0.25 mg weekly, titrate to 1.0 mg"),
    InterventionSpec("zone2", "Zone 2 Cardio 180 min/week", 1,
                     "Structured aerobic program targeting 60-70% max HR"),
    InterventionSpec("med_diet", "Mediterranean Diet + CGM", 2,
                     "Mediterranean dietary pattern with CGM feedback loops"),
    InterventionSpec("metformin", "Metformin 500 mg BID", 3,
                     "First-line oral hypoglycaemic for insulin resistance"),
    InterventionSpec("resistance", "Resistance Training 3×/week", 4,
                     "Progressive overload targeting major muscle groups"),
    InterventionSpec("sleep_opt", "Sleep Optimisation Protocol", 5,
                     "CBT-I plus sleep hygiene targeting 7-9 hrs, >60 min deep sleep"),
    InterventionSpec("cgm_only", "CGM-Guided Nutrition (no medication)", 6,
                     "Continuous glucose monitor with dietary feedback only"),
    InterventionSpec("combo", "Metformin + Zone 2 + Med Diet", 7,
                     "Combined pharmacological and lifestyle intervention"),
]


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------
class CausalEngine:
    """Orchestrates counterfactual simulation and twin-based validation."""

    def __init__(
        self,
        model: ClinicalJEPA,
        vector_db: VectorDBClient,
        interventions: list[InterventionSpec] | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.vector_db = vector_db
        self.interventions = interventions or DEFAULT_INTERVENTIONS
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # 1.  Simulate trajectories
    # ------------------------------------------------------------------
    def simulate_trajectories(
        self,
        origin_tensor: torch.Tensor,
    ) -> dict[str, dict[str, Any]]:
        """Run the JEPA Predictor for every registered intervention.

        Parameters
        ----------
        origin_tensor : (1, d_model) or (d_model,) — the patient's origin embedding.

        Returns
        -------
        Mapping intervention_id → {
            "name", "embedding" (list[float]),
            "delta_norm" (float), "description"
        }
        """
        if origin_tensor.dim() == 1:
            origin_tensor = origin_tensor.unsqueeze(0)
        origin_tensor = origin_tensor.to(self.device)

        trajectories: dict[str, dict[str, Any]] = {}

        for spec in self.interventions:
            z = torch.zeros(1, self.model.num_interventions, device=self.device)
            z[0, spec.index] = 1.0

            predicted = self.model.predict_intervention(origin_tensor, z)  # (1, d)

            delta = predicted - origin_tensor
            delta_norm = delta.norm(dim=-1).item()

            trajectories[spec.id] = {
                "name": spec.name,
                "description": spec.description,
                "embedding": predicted.squeeze(0).cpu().tolist(),
                "delta_norm": delta_norm,
            }

        return trajectories

    # ------------------------------------------------------------------
    # 2.  Validate against real twins
    # ------------------------------------------------------------------
    def validate_with_reality(
        self,
        origin_embedding: list[float],
        simulated_trajectories: dict[str, dict[str, Any]],
        twin_limit: int = 100,
        prefetched_twins: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Query Qdrant for actual twins, compare real outcomes against
        theoretical predictions, and return a deterministic ranking.

        Parameters
        ----------
        prefetched_twins : If supplied, skip the Qdrant query and use
            these twins directly.  Useful when the caller already
            fetched twins for use by other pipeline stages (e.g. RAG).

        Returns
        -------
        Sorted list (best-first) of dicts with keys:
            rank, intervention_id, intervention_name,
            jepa_confidence, twin_concordance, combined_score,
            simulated_trajectory, supporting_twins
        """
        if prefetched_twins is not None:
            twins = prefetched_twins
        else:
            twins = self.vector_db.find_origin_twins(
                user_embedding=origin_embedding, limit=twin_limit
            )

        # Group twins by the intervention they actually received
        twins_by_intervention: dict[str, list[dict]] = {}
        for tw in twins:
            key = tw["intervention_taken"]
            twins_by_intervention.setdefault(key, []).append(tw)

        ranked: list[dict[str, Any]] = []

        for spec in self.interventions:
            traj = simulated_trajectories.get(spec.id)
            if traj is None:
                continue

            predicted_emb = torch.tensor(traj["embedding"], dtype=torch.float32)

            matching_twins = twins_by_intervention.get(spec.id, [])
            matching_twins += twins_by_intervention.get(spec.name, [])

            concordant = 0
            total = len(matching_twins) or 1  # avoid div-by-zero

            for tw in matching_twins:
                outcome = tw.get("actual_clinical_outcome", "")
                is_positive = self._classify_outcome(outcome)
                if is_positive:
                    concordant += 1

            twin_concordance = concordant / total

            # JEPA confidence: inverse of delta-norm (smaller shift ⇒ more conservative)
            # normalised to [0, 1] via sigmoid
            jepa_confidence = torch.sigmoid(
                torch.tensor(-traj["delta_norm"] + 2.0)
            ).item()

            combined = 0.6 * twin_concordance + 0.4 * jepa_confidence

            ranked.append(
                {
                    "intervention_id": spec.id,
                    "intervention_name": spec.name,
                    "jepa_confidence": round(jepa_confidence, 4),
                    "twin_concordance": round(twin_concordance, 4),
                    "combined_score": round(combined, 4),
                    "simulated_trajectory": {
                        "intervention_id": spec.id,
                        "intervention_name": spec.name,
                        "predicted_embedding": traj["embedding"],
                        "predicted_coordinate": self._embedding_to_3d(
                            traj["embedding"]
                        ),
                        "confidence": round(jepa_confidence, 4),
                        "delta_risk_score": round(-traj["delta_norm"], 4),
                    },
                    "supporting_twins": [
                        {
                            "patient_id": tw["patient_id"],
                            "similarity": round(tw["similarity"], 4),
                            "intervention_taken": tw["intervention_taken"],
                            "actual_outcome": tw["actual_clinical_outcome"],
                            "outcome_months": tw.get("outcome_months", 0),
                            "coordinate": tw.get("coordinate", {}),
                        }
                        for tw in matching_twins[:5]
                    ],
                }
            )

        ranked.sort(key=lambda r: r["combined_score"], reverse=True)
        for i, entry in enumerate(ranked, start=1):
            entry["rank"] = i

        return ranked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_outcome(outcome_text: str) -> bool:
        """Heuristic classifier for outcome polarity.
        In production, replace with a fine-tuned NLP classifier."""
        positive_signals = [
            "reversed", "improved", "stabilized", "decreased", "dropped",
            "reduced", "remission", "resolved", "normal",
        ]
        negative_signals = [
            "progressed", "worsened", "increased", "elevated", "developed",
            "hospitalized", "fatal",
        ]
        text = outcome_text.lower()
        pos = sum(1 for s in positive_signals if s in text)
        neg = sum(1 for s in negative_signals if s in text)
        return pos > neg

    @staticmethod
    def _embedding_to_3d(embedding: list[float]) -> dict[str, float]:
        """Project a high-dimensional embedding to 3-D for visualisation.
        Production should use a fitted UMAP/PaCMAP; this is a PCA-style stub."""
        import numpy as np

        arr = np.array(embedding, dtype=np.float32)
        # Deterministic pseudo-projection: take first 3 principal directions
        # via fixed stride over the embedding vector
        stride = max(len(arr) // 3, 1)
        x = float(np.mean(arr[:stride]))
        y = float(np.mean(arr[stride : stride * 2]))
        z = float(np.mean(arr[stride * 2 : stride * 3]))
        return {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}
