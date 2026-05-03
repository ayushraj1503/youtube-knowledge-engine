# backend/api/routes/ingest.py
"""Ingestion endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from backend.core.logging_config import get_logger
from backend.models.schemas import IngestChannelRequest, IngestChannelResponse, IngestionJobStatus, IngestionStatus
from backend.services.ingestion.channel_ingester import ChannelIngester, get_job, list_jobs

router = APIRouter(prefix="/ingest", tags=["Ingestion"])
logger = get_logger(__name__)
_ingester = ChannelIngester()


@router.post(
    "/channel",
    response_model=IngestChannelResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest an entire YouTube channel",
)
async def ingest_channel(request: IngestChannelRequest):
    """
    Start async ingestion of a YouTube channel.
    Returns a job_id immediately; poll /ingest/status/{job_id} for progress.

    - **channel_url**: Full channel URL or @handle
    - **max_videos**: Cap number of videos (default: env YTDLP_MAX_VIDEOS)
    - **force_reingest**: Re-process already-ingested videos
    - **extract_frames**: Enable BLIP visual captioning pipeline
    """
    try:
        job_id = await _ingester.ingest_channel(
            channel_url=request.channel_url,
            max_videos=request.max_videos,
            force_reingest=request.force_reingest,
            extract_frames=request.extract_frames,
        )

        return IngestChannelResponse(
            job_id=job_id,
            status=IngestionStatus.PENDING,
            message="Ingestion job started. Poll /ingest/status/{job_id} for progress.",
        )
    except Exception as e:
        logger.error("ingest_channel_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start ingestion: {str(e)}",
        )


@router.get(
    "/status/{job_id}",
    response_model=IngestionJobStatus,
    summary="Get ingestion job status",
)
async def get_ingestion_status(job_id: str):
    """Poll this endpoint to check ingestion progress."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return job


@router.get(
    "/jobs",
    response_model=list,
    summary="List all ingestion jobs",
)
async def list_ingestion_jobs():
    """Returns all ingestion jobs (for monitoring / debugging)."""
    return list_jobs()

class IngestVideoRequest(BaseModel):
    video_url: str
    force_reingest: bool = False
    extract_frames: bool = False

@router.post(
    "/video",
    response_model=IngestChannelResponse,
    status_code=202,
    summary="Ingest a single YouTube video",
)
async def ingest_single_video(request: IngestVideoRequest):
    try:
        job_id = await _ingester.ingest_video(
            video_url=request.video_url,
            force_reingest=request.force_reingest,
            extract_frames=request.extract_frames,
        )
        return IngestChannelResponse(
            job_id=job_id,
            status=IngestionStatus.PENDING,
            message="Single video ingestion started.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))