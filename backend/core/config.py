# backend/core/config.py
"""
Central configuration module using Pydantic Settings.
All environment variables are validated and typed here.
Single source of truth for all configuration.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.
    Using Pydantic ensures type safety and validation at startup.
    """

    # ── Project ───────────────────────────────────────────────────────────
    PROJECT_NAME: str = "YouTube Knowledge Engine"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Production-grade YouTube Channel RAG System"

    # ── Groq LLM ──────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    GROQ_MODEL: str = "llama3-70b-8192"
    GROQ_MAX_TOKENS: int = 2048
    GROQ_TEMPERATURE: float = 0.1

    # ── Embedding ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_CACHE_DIR: Path = Path("./data/cache/embeddings")

    # ── ChromaDB ──────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: Path = Path("./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = "youtube_knowledge_base"

    # ── Processing ────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    FRAME_EXTRACTION_FPS: float = 0.5
    MAX_FRAMES_PER_VIDEO: int = 20
    BLIP_MODEL: str = "Salesforce/blip-image-captioning-base"
    BLIP_ENABLED: bool = True

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K: int = 5
    HYBRID_ALPHA: float = 0.7          # weight for vector vs BM25 (1.0 = pure vector)
    RERANKER_ENABLED: bool = True
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # ── API ───────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = False
    SECRET_KEY: str = "change_me_to_a_secure_random_string"
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60        # seconds

    # ── Frontend ──────────────────────────────────────────────────────────
    STREAMLIT_PORT: int = 8501
    BACKEND_URL: str = "http://localhost:8000"

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("./data/logs")
    LOG_MAX_BYTES: int = 10_485_760    # 10 MB
    LOG_BACKUP_COUNT: int = 5

    # ── yt-dlp ────────────────────────────────────────────────────────────
    YTDLP_COOKIES_FILE: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    YTDLP_MAX_VIDEOS: int = 500
    YTDLP_SLEEP_INTERVAL: int = 1

    # ── Concurrency ───────────────────────────────────────────────────────
    INGESTION_CONCURRENCY: int = 4
    EMBEDDING_CONCURRENCY: int = 8

    @validator("EMBEDDING_CACHE_DIR", "CHROMA_PERSIST_DIR", "LOG_DIR", pre=True)
    def create_dirs(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @validator("HYBRID_ALPHA")
    def validate_alpha(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("HYBRID_ALPHA must be between 0.0 and 1.0")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    lru_cache ensures we only parse .env once per process lifetime.
    """
    return Settings()
