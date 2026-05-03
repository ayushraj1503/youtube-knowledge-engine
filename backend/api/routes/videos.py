# backend/api/routes/videos.py
"""Video listing endpoint."""

from fastapi import APIRouter, Query

from backend.models.schemas import VideoListResponse, VideoMetadata
from backend.services.embedding.embedding_service import EmbeddingService

router = APIRouter(prefix="/videos", tags=["Videos"])
_embedding_service = EmbeddingService()


@router.get(
    "",
    response_model=VideoListResponse,
    summary="List all ingested videos",
)
async def list_videos(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    channel_id: str = Query(default=None),
):
    """
    Paginated list of all videos in the knowledge base.
    Supports filtering by channel_id.
    """
    result = await _embedding_service.get_all_videos(
        page=page, page_size=page_size
    )

    videos = []
    for meta in result["videos"]:
        # Filter by channel if requested
        if channel_id and meta.get("channel_id") != channel_id:
            continue
        videos.append(
            VideoMetadata(
                video_id=meta.get("video_id", ""),
                title=meta.get("video_title", ""),
                channel_id=meta.get("channel_id", ""),
                channel_name=meta.get("channel_name", ""),
                url=meta.get("url", ""),
                duration_seconds=int(meta.get("duration_seconds", 0)),
                published_at=meta.get("published_at"),
            )
        )

    return VideoListResponse(
        videos=videos,
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        has_next=result["has_next"],
    )
