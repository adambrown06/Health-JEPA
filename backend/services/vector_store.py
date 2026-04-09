"""Qdrant-backed vector store for patient embeddings and twin retrieval."""

from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import settings

logger = logging.getLogger(__name__)

DISTANCE = models.Distance.COSINE
EMBEDDING_DIM = settings.embedding_dim


class VectorDBClient:
    """Thin, typed wrapper around the Qdrant client scoped to patient embeddings."""

    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        grpc_port: int = settings.qdrant_grpc_port,
        api_key: Optional[str] = settings.qdrant_api_key,
        collection: str = settings.qdrant_collection,
    ):
        self._collection = collection
        self._client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            prefer_grpc=True,
        )
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """Create the collection with HNSW indexing if it doesn't exist."""
        try:
            self._client.get_collection(self._collection)
            logger.info("Qdrant collection '%s' already exists.", self._collection)
        except (UnexpectedResponse, Exception):
            logger.info("Creating Qdrant collection '%s'…", self._collection)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=DISTANCE,
                    on_disk=True,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                    full_scan_threshold=10_000,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def upsert_patient(
        self,
        patient_id: str,
        origin_embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Store or update a patient's JEPA-generated embedding.

        Parameters
        ----------
        patient_id       : Unique patient identifier (used as Qdrant point ID hash).
        origin_embedding : Dense vector from ClinicalJEPA.encode().
        metadata         : Must include at minimum:
                           - intervention_taken (str)
                           - actual_clinical_outcome (str)
                           May also include demographics, timestamps, etc.
        """
        required_keys = {"intervention_taken", "actual_clinical_outcome"}
        missing = required_keys - set(metadata.keys())
        if missing:
            raise ValueError(f"Metadata missing required keys: {missing}")

        self._client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=self._deterministic_id(patient_id),
                    vector=origin_embedding,
                    payload={"patient_id": patient_id, **metadata},
                )
            ],
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def find_origin_twins(
        self,
        user_embedding: list[float],
        limit: int = 100,
        score_threshold: Optional[float] = None,
        filters: Optional[models.Filter] = None,
    ) -> list[dict[str, Any]]:
        """HNSW nearest-neighbour search for historical trajectory twins.

        Returns a list of dicts, each containing:
            - patient_id
            - similarity (float)
            - intervention_taken
            - actual_clinical_outcome
            - full payload
        """
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=False,
        )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=user_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filters,
            search_params=search_params,
        )

        twins: list[dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            twins.append(
                {
                    "patient_id": payload.get("patient_id", str(hit.id)),
                    "similarity": hit.score,
                    "intervention_taken": payload.get("intervention_taken", "unknown"),
                    "actual_clinical_outcome": payload.get(
                        "actual_clinical_outcome", "unknown"
                    ),
                    "outcome_months": payload.get("outcome_months", 0),
                    "coordinate": payload.get("coordinate", {}),
                    "payload": payload,
                }
            )
        return twins

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _deterministic_id(patient_id: str) -> str:
        """Produce a deterministic UUID-format string from patient_id
        so upserts are idempotent."""
        import hashlib
        import uuid

        hex_dig = hashlib.sha256(patient_id.encode()).hexdigest()
        return str(uuid.UUID(hex_dig[:32]))

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
