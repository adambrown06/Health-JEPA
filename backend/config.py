from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str | None = None
    qdrant_collection: str = "patient_embeddings"

    # --- Celery / Redis ---
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # --- JEPA Model ---
    jepa_checkpoint_path: str = "checkpoints/jepa_latest.pt"
    jepa_vb_checkpoint_path: str = "checkpoints/jepa_vb_latest.pt"
    jepa_vb_use_dual_ema: bool = False
    embedding_dim: int = 256
    num_features: int = 32
    num_interventions: int = 8
    jepa_num_heads: int = 8
    jepa_num_layers: int = 6
    jepa_ff_dim: int = 1024
    jepa_dropout: float = 0.1
    ema_decay: float = 0.996

    # --- API ---
    api_v1_prefix: str = "/api/v1"
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
    ]

    # --- LLM / RAG ---
    openai_api_key: str | None = None
    rag_model: str = "gpt-4o"
    rag_neighborhood_model: str = "gpt-4o-mini"
    rag_max_concurrent: int = 3
    rag_request_timeout: float = 30.0


settings = Settings()
