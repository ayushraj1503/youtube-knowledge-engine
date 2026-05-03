# backend/main.py
"""
FastAPI Application Entry Point

Architecture overview:
  - Modular routing: each domain (ingest, query, videos, health) has its own router
  - Middleware stack: CORS → Rate Limiting → request logging
  - Lifespan context manager: handles startup/shutdown tasks cleanly
  - OpenAPI docs available at /docs and /redoc

Starting the server:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import health, ingest, query, videos
from backend.core.config import get_settings
from backend.core.logging_config import get_logger, setup_logging
from backend.middleware.rate_limiter import RateLimiterMiddleware

# ── Initialise logging before anything else ───────────────────────────────────
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ── Lifespan: startup / shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-load the embedding model so the first request isn't slow.
    Shutdown: flush any pending operations.
    """
    logger.info("startup", version=settings.VERSION)

    # Warm up embedding model (loads ~90MB model into memory)
    try:
        from backend.services.embedding.embedding_service import EmbeddingService
        svc = EmbeddingService()
        svc._get_model()
        logger.info("embedding_model_warmed_up")
    except Exception as e:
        logger.warning("model_warmup_failed", error=str(e))

    yield  # ── Server is running ──

    logger.info("shutdown")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ── Middleware (order matters: applied in reverse) ────────────────────────────

# 1. CORS — allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production: ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Rate limiting
app.add_middleware(
    RateLimiterMiddleware,
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW,
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(duration, 2))
    logger.debug(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration, 2),
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(videos.router, prefix="/api/v1")


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=1 if settings.API_RELOAD else settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
