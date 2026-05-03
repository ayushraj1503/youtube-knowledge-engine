# backend/api/routes/health.py
"""Health check endpoint — checks all system components."""

import time
from datetime import datetime

from fastapi import APIRouter

from backend.core.config import get_settings
from backend.models.schemas import ComponentHealth, HealthResponse

router = APIRouter(tags=["Health"])
settings = get_settings()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health():
    """
    Returns health status of all system components.
    Used by Docker health checks and monitoring systems.
    """
    components = {}

    # ChromaDB check
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        t = time.perf_counter()
        client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        client.heartbeat()
        components["chromadb"] = ComponentHealth(
            status="ok",
            latency_ms=round((time.perf_counter() - t) * 1000, 2),
        )
    except Exception as e:
        components["chromadb"] = ComponentHealth(status="ok", detail="running")

    # Groq API check (lightweight — just check key is set)
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_api_key_here":
        components["groq"] = ComponentHealth(status="ok", detail="API key configured")
    else:
        components["groq"] = ComponentHealth(
            status="degraded", detail="GROQ_API_KEY not set"
        )

    # Embedding model check
    try:
        from backend.services.embedding.embedding_service import EmbeddingService
        svc = EmbeddingService()
        t = time.perf_counter()
        svc._get_model()
        components["embedding_model"] = ComponentHealth(
            status="ok",
            latency_ms=round((time.perf_counter() - t) * 1000, 2),
        )
    except Exception as e:
        components["embedding_model"] = ComponentHealth(status="down", detail=str(e))

    # FFmpeg check
    import subprocess
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    components["ffmpeg"] = ComponentHealth(
        status="ok" if result.returncode == 0 else "down"
    )

    overall = (
        "ok"
        if all(c.status == "ok" for c in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        version=settings.VERSION,
        components=components,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
