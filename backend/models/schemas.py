# backend/models/schemas.py
"""
Pydantic v2 schemas — the canonical data contracts for the entire system.
All services communicate through these typed models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, validator


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class IngestionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    BM25 = "bm25"


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO METADATA
# ══════════════════════════════════════════════════════════════════════════════

class VideoMetadata(BaseModel):
    video_id: str
    title: str
    channel_id: str
    channel_name: str
    url: str
    duration_seconds: int
    published_at: Optional[str] = None
    view_count: Optional[int] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: List[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT / CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

class TranscriptSegment(BaseModel):
    """Raw transcript segment from youtube-transcript-api."""
    text: str
    start: float          # seconds from video start
    duration: float


class TextChunk(BaseModel):
    """Processed, cleaned chunk ready for embedding."""
    chunk_id: str
    video_id: str
    video_title: str
    channel_name: str
    text: str
    start_time: float     # seconds — used for timestamp links
    end_time: float
    chunk_index: int
    total_chunks: int
    frame_caption: Optional[str] = None   # BLIP caption if extracted
    metadata: Dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ══════════════════════════════════════════════════════════════════════════════

class IngestChannelRequest(BaseModel):
    channel_url: str = Field(..., description="YouTube channel URL or handle (@channel)")
    max_videos: Optional[int] = Field(None, ge=1, le=5000)
    force_reingest: bool = False
    extract_frames: bool = True


class IngestChannelResponse(BaseModel):
    job_id: str
    status: IngestionStatus
    message: str
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    videos_discovered: Optional[int] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)


class IngestionJobStatus(BaseModel):
    job_id: str
    status: IngestionStatus
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    videos_total: int = 0
    videos_processed: int = 0
    videos_failed: int = 0
    chunks_created: int = 0
    embeddings_stored: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# QUERY / RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    channel_id: Optional[str] = None     # filter to a specific channel
    video_ids: Optional[List[str]] = None  # filter to specific videos
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: SearchMode = SearchMode.HYBRID
    rerank: bool = True
    rewrite_query: bool = True           # LLM query rewriting


class RetrievedChunk(BaseModel):
    chunk_id: str
    video_id: str
    video_title: str
    channel_name: str
    text: str
    start_time: float
    end_time: float
    score: float                          # relevance score (0-1)
    timestamp_url: str                    # deep-link to exact moment
    frame_caption: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: List[RetrievedChunk]
    search_mode: SearchMode
    latency_ms: float
    tokens_used: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO LISTING
# ══════════════════════════════════════════════════════════════════════════════

class VideoListResponse(BaseModel):
    videos: List[VideoMetadata]
    total: int
    page: int
    page_size: int
    has_next: bool


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

class ComponentHealth(BaseModel):
    status: str          # "ok" | "degraded" | "down"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, ComponentHealth]
    uptime_seconds: float


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

class EvalRequest(BaseModel):
    query: str
    expected_video_ids: List[str]        # ground truth
    top_k: int = 5


class EvalResponse(BaseModel):
    query: str
    precision_at_k: float
    recall_at_k: float
    mrr: float                           # Mean Reciprocal Rank
    retrieved_video_ids: List[str]
