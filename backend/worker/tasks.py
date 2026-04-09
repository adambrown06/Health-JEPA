"""
Celery tasks for the Cohort Compass inference pipeline.

Pipeline stages (executed sequentially inside a single task):
    1. Tensor preparation  — convert ClinicalTimeSeries → PyTorch tensors
    2. JEPA encoding       — produce the patient origin embedding
    3. Trajectory sim      — run the Predictor for each intervention
    4. Twin retrieval      — explicit vector search in Qdrant (100 nearest)
    5. Outcome ranking     — CausalEngine with pre-fetched twins
    6. RAG synthesis       — ClinicalRAG → per-intervention summaries
    7. Neighbourhood build — assemble NeighborhoodResponse for 3D UI
    8. Persist & callback  — cache result, fire webhook if requested
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx
import numpy as np
import torch

from config import settings
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# ======================================================================
# Lazy singletons — initialised once per worker process, not per task
# ======================================================================
_model: Any = None
_vector_db: Any = None
_engine: Any = None
_rag: Any = None


def _get_model():
    global _model
    if _model is None:
        from ml.jepa_model import ClinicalJEPA

        _model = ClinicalJEPA(
            num_features=settings.num_features,
            d_model=settings.embedding_dim,
            n_heads=settings.jepa_num_heads,
            n_layers=settings.jepa_num_layers,
            d_ff=settings.jepa_ff_dim,
            dropout=settings.jepa_dropout,
            num_interventions=settings.num_interventions,
            ema_decay=settings.ema_decay,
        )
        checkpoint_path = settings.jepa_checkpoint_path
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            _model.load_state_dict(state)
            logger.info("Loaded JEPA checkpoint from %s", checkpoint_path)
        except FileNotFoundError:
            logger.warning(
                "No checkpoint at %s — running with random weights (dev mode).",
                checkpoint_path,
            )
        _model.eval()
    return _model


def _get_vector_db():
    global _vector_db
    if _vector_db is None:
        from services.vector_store import VectorDBClient

        _vector_db = VectorDBClient()
    return _vector_db


def _get_engine():
    global _engine
    if _engine is None:
        from services.counterfactual import CausalEngine

        _engine = CausalEngine(model=_get_model(), vector_db=_get_vector_db())
    return _engine


def _get_rag_engine():
    global _rag
    if _rag is None:
        from services.rag_engine import ClinicalRAG

        _rag = ClinicalRAG()
    return _rag


# ======================================================================
# Main task
# ======================================================================
@celery_app.task(
    bind=True,
    name="worker.tasks.run_full_analysis",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def run_full_analysis(self, payload: dict) -> dict:
    """Execute the full Cohort Compass inference pipeline.

    Parameters
    ----------
    payload : serialised InferenceJobRequest (dict).

    Returns
    -------
    Serialised AnalysisResult dict (includes NeighborhoodResponse).
    """
    job_id = self.request.id or payload.get("callback_job_id", "unknown")
    self.update_state(state="RUNNING", meta={"progress": 0.0, "stage": "preparing"})

    try:
        # ---- 1. Tensor preparation --------------------------------
        ts_data = payload["clinical_time_series"]
        feature_names = ts_data["feature_names"]
        seq_len = ts_data["sequence_length"]

        values, mask, timestamps = _build_tensors(ts_data, feature_names, seq_len)
        self.update_state(state="RUNNING", meta={"progress": 0.10, "stage": "encoding"})

        # ---- 2. JEPA encoding → origin embedding -------------------
        model = _get_model()
        origin_embedding = model.encode(values, mask, timestamps)  # (1, d)
        origin_list = origin_embedding.squeeze(0).cpu().tolist()
        self.update_state(state="RUNNING", meta={"progress": 0.20, "stage": "simulating"})

        # ---- 3. Counterfactual trajectory simulation ---------------
        engine = _get_engine()
        trajectories = engine.simulate_trajectories(origin_embedding.squeeze(0))
        self.update_state(state="RUNNING", meta={"progress": 0.30, "stage": "twin_search"})

        # ---- 4. Twin retrieval (explicit, shared across stages) ----
        vector_db = _get_vector_db()
        twins = vector_db.find_origin_twins(user_embedding=origin_list, limit=100)
        self.update_state(state="RUNNING", meta={"progress": 0.40, "stage": "ranking"})

        # ---- 5. Outcome ranking (uses pre-fetched twins) -----------
        ranked = engine.validate_with_reality(
            origin_list, trajectories, prefetched_twins=twins
        )
        self.update_state(state="RUNNING", meta={"progress": 0.55, "stage": "rag_synthesis"})

        # ---- 6. RAG synthesis (async → sync bridge) ----------------
        rag = _get_rag_engine()
        rag_summaries = asyncio.run(
            rag.generate_intervention_summaries(ranked, twins, top_k=3)
        )
        self.update_state(state="RUNNING", meta={"progress": 0.75, "stage": "building_neighborhood"})

        # ---- 7. Build NeighborhoodResponse -------------------------
        from services.counterfactual import CausalEngine

        origin_coord = CausalEngine._embedding_to_3d(origin_list)
        neighborhood = _build_neighborhood(origin_coord, twins, rag_summaries)
        narrative = _narrative_from_rag(rag_summaries, ranked)
        self.update_state(state="RUNNING", meta={"progress": 0.90, "stage": "persisting"})

        # ---- 8. Assemble final result & persist --------------------
        result = {
            "job_id": job_id,
            "status": "completed",
            "origin_embedding": origin_list,
            "origin_coordinate": origin_coord,
            "cluster_id": _assign_cluster(origin_coord),
            "cluster_name": "Metabolic Risk",
            "ranked_interventions": ranked,
            "narrative_summary": narrative,
            "neighborhood": neighborhood,
            "completed_at": datetime.utcnow().isoformat(),
        }

        _fire_webhook(payload.get("webhook_url"), result)
        self.update_state(state="RUNNING", meta={"progress": 1.0, "stage": "done"})

        return result

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        self.update_state(
            state="FAILURE",
            meta={"progress": 0.0, "stage": "error", "error": str(exc)},
        )
        raise self.retry(exc=exc)


# ======================================================================
# Internal helpers
# ======================================================================

def _build_tensors(
    ts_data: dict, feature_names: list[str], seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the serialised ClinicalTimeSeries dict into padded PyTorch tensors."""
    num_features = len(feature_names)
    feature_index = {name: i for i, name in enumerate(feature_names)}

    vals = np.full((seq_len, num_features), fill_value=0.0, dtype=np.float32)
    msk = np.zeros((seq_len, num_features), dtype=np.float32)
    ts = np.zeros(seq_len, dtype=np.float64)

    observations = ts_data.get("observations", [])
    if observations:
        t0 = observations[0]["timestamp"]

    for step, obs in enumerate(observations[:seq_len]):
        col = feature_index.get(obs["concept_id"])
        if col is not None and obs.get("value") is not None and not obs.get("is_missing", False):
            vals[step, col] = obs["value"]
            msk[step, col] = 1.0
        elapsed = obs.get("timestamp", t0)
        if isinstance(elapsed, str):
            from datetime import datetime as dt

            elapsed = dt.fromisoformat(elapsed)
            t0_dt = dt.fromisoformat(t0) if isinstance(t0, str) else t0
            ts[step] = (elapsed - t0_dt).total_seconds()

    values_t = torch.from_numpy(vals).unsqueeze(0)      # (1, T, F)
    mask_t = torch.from_numpy(msk).unsqueeze(0)          # (1, T, F)
    timestamps_t = torch.from_numpy(ts).unsqueeze(0)     # (1, T)
    return values_t, mask_t, timestamps_t


def _build_neighborhood(
    origin_coord: dict[str, float],
    twins: list[dict[str, Any]],
    rag_summaries: list,
) -> dict[str, Any]:
    """Assemble the NeighborhoodResponse dict for the 3D frontend."""
    from services.rag_engine import _is_positive

    ox = origin_coord.get("x", 0.0)
    oy = origin_coord.get("y", 0.0)
    oz = origin_coord.get("z", 0.0)

    twin_records: list[dict[str, Any]] = []
    for tw in twins:
        coord = tw.get("coordinate", {})
        twin_records.append({
            "patient_id": tw["patient_id"],
            "relative_coordinates": [
                round(coord.get("x", 0.0) - ox, 4),
                round(coord.get("y", 0.0) - oy, 4),
                round(coord.get("z", 0.0) - oz, 4),
            ],
            "intervention_taken": tw.get("intervention_taken", "unknown"),
            "outcome_status": "Positive" if _is_positive(tw) else "Negative",
        })

    roadmap = [
        s.model_dump() if hasattr(s, "model_dump") else s
        for s in rag_summaries
    ]

    return {
        "origin_coordinate": origin_coord,
        "neighborhood_twins": twin_records,
        "roadmap": roadmap,
    }


def _narrative_from_rag(
    rag_summaries: list,
    ranked: list[dict[str, Any]],
) -> str:
    """Build the top-level narrative_summary from RAG intervention summaries."""
    if not rag_summaries:
        return _template_narrative(ranked)

    paragraphs: list[str] = []
    for i, summary in enumerate(rag_summaries):
        title = summary.title if hasattr(summary, "title") else summary.get("title", "")
        rate = summary.success_rate if hasattr(summary, "success_rate") else summary.get("success_rate", 0)
        paragraphs.append(
            f"**{i + 1}. {title}** (success rate: {rate:.0%})"
        )

    top = rag_summaries[0]
    explanation = (
        top.generated_explanation
        if hasattr(top, "generated_explanation")
        else top.get("generated_explanation", "")
    )

    return (
        "Based on causal trajectory analysis and validation against "
        "real-world clinical twins, the following interventions are "
        "recommended in order of combined confidence:\n\n"
        + "\n".join(paragraphs)
        + f"\n\n{explanation}"
    )


def _template_narrative(ranked: list[dict]) -> str:
    """Deterministic fallback narrative when RAG produces no results."""
    if not ranked:
        return "Insufficient data to generate intervention recommendations."

    top = ranked[0]
    lines = [
        f"Based on causal trajectory analysis, the highest-ranked intervention is "
        f"**{top['intervention_name']}** (combined score: {top['combined_score']:.2f}).",
        f"JEPA model confidence: {top['jepa_confidence']:.1%}. "
        f"Twin concordance from {len(top.get('supporting_twins', []))} matched "
        f"historical patients: {top['twin_concordance']:.1%}.",
    ]
    if len(ranked) > 1:
        runner = ranked[1]
        lines.append(
            f"Runner-up: **{runner['intervention_name']}** "
            f"(score: {runner['combined_score']:.2f})."
        )
    return " ".join(lines)


def _fire_webhook(url: str | None, result: dict) -> None:
    """POST result to the caller's webhook, best-effort."""
    if not url:
        return
    try:
        httpx.post(str(url), json=result, timeout=10.0)
    except Exception:
        logger.warning("Webhook delivery to %s failed", url)


def _assign_cluster(coord: dict[str, float]) -> int:
    """Placeholder cluster assignment based on proximity to known centroids."""
    centroids = {
        1: (0.0, 0.0, 0.0),
        0: (16.0, 0.0, 0.0),
        2: (-12.0, 0.0, 12.0),
        3: (-12.0, 0.0, -12.0),
        4: (0.0, 0.0, -16.0),
    }
    cx, cy, cz = coord.get("x", 0), coord.get("y", 0), coord.get("z", 0)
    best_id, best_dist = 1, float("inf")
    for cid, (mx, my, mz) in centroids.items():
        d = ((cx - mx) ** 2 + (cy - my) ** 2 + (cz - mz) ** 2) ** 0.5
        if d < best_dist:
            best_id, best_dist = cid, d
    return best_id
