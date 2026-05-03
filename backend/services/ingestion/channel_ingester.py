# backend/services/ingestion/channel_ingester.py
"""
Ingestion Service — the entry point for processing an entire YouTube channel.

Responsibilities:
  1. Fetch all video IDs from a channel via yt-dlp
  2. Download metadata for each video
  3. Retrieve transcripts via youtube-transcript-api
  4. Dispatch to the processing pipeline
  5. Track ingestion job state

Design decisions:
  - Asyncio-based concurrency: we can process N videos in parallel without
    blocking the FastAPI event loop (using asyncio.Semaphore for back-pressure).
  - yt-dlp is called in a ThreadPoolExecutor because it is a synchronous library.
  - Retry logic wraps each video individually so one failure doesn't abort the job.
"""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

import yt_dlp
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import (
    IngestionJobStatus,
    IngestionStatus,
    TranscriptSegment,
    VideoMetadata,
)
from backend.services.processing.pipeline import ProcessingPipeline
from backend.utils.retry import async_retry

logger = get_logger(__name__)
settings = get_settings()

# In-memory job store (swap for Redis in production for multi-worker setups)
_jobs: Dict[str, IngestionJobStatus] = {}


def get_job(job_id: str) -> Optional[IngestionJobStatus]:
    return _jobs.get(job_id)


def list_jobs() -> List[IngestionJobStatus]:
    return list(_jobs.values())


class ChannelIngester:
    """
    Orchestrates full channel ingestion.
    Each public method is async; blocking I/O runs in a thread pool.
    """

    def __init__(self):
        self.settings = get_settings()
        self.pipeline = ProcessingPipeline()
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.INGESTION_CONCURRENCY,
            thread_name_prefix="ingester",
        )
        self._semaphore = asyncio.Semaphore(self.settings.INGESTION_CONCURRENCY)

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    async def ingest_channel(
        self,
        channel_url: str,
        max_videos: Optional[int] = None,
        force_reingest: bool = False,
        extract_frames: bool = True,
    ) -> str:
        """
        Start async ingestion job for a YouTube channel.
        Returns job_id immediately; processing continues in background.
        """
        job_id = str(uuid.uuid4())
        max_videos = max_videos or self.settings.YTDLP_MAX_VIDEOS

        job = IngestionJobStatus(
            job_id=job_id,
            status=IngestionStatus.PENDING,
            started_at=datetime.utcnow(),
        )
        _jobs[job_id] = job

        # Fire-and-forget — FastAPI's event loop handles this
        asyncio.create_task(
            self._run_ingestion(
                job_id, channel_url, max_videos, force_reingest, extract_frames
            )
        )

        logger.info("ingestion_started", job_id=job_id, channel_url=channel_url)
        return job_id

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL ORCHESTRATION
    # ──────────────────────────────────────────────────────────────────────

    async def _run_ingestion(
        self,
        job_id: str,
        channel_url: str,
        max_videos: int,
        force_reingest: bool,
        extract_frames: bool,
    ) -> None:
        job = _jobs[job_id]
        job.status = IngestionStatus.RUNNING

        try:
            # Step 1: fetch video list
            logger.info("fetching_video_list", job_id=job_id, url=channel_url)
            videos = await self._fetch_channel_videos(channel_url, max_videos)

            job.channel_id = videos[0].channel_id if videos else None
            job.channel_name = videos[0].channel_name if videos else None
            job.videos_total = len(videos)
            job.progress_percent = 5.0

            logger.info(
                "videos_discovered", job_id=job_id, count=len(videos)
            )

            # Step 2: process each video concurrently
            tasks = [
                self._process_video(
                    job_id, video, force_reingest, extract_frames
                )
                for video in videos
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Step 3: finalise
            failed = job.videos_failed
            job.status = (
                IngestionStatus.COMPLETED
                if failed == 0
                else IngestionStatus.PARTIAL
            )
            job.completed_at = datetime.utcnow()
            job.progress_percent = 100.0

            logger.info(
                "ingestion_complete",
                job_id=job_id,
                processed=job.videos_processed,
                failed=failed,
                chunks=job.chunks_created,
            )

        except Exception as exc:
            logger.error("ingestion_failed", job_id=job_id, error=str(exc))
            job.status = IngestionStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()

    async def _process_video(
        self,
        job_id: str,
        video: VideoMetadata,
        force_reingest: bool,
        extract_frames: bool,
    ) -> None:
        async with self._semaphore:
            job = _jobs[job_id]
            try:
                # Fetch transcript
                segments = await self._fetch_transcript(video.video_id)
                if not segments:
                    logger.warning(
                        "no_transcript", video_id=video.video_id, title=video.title
                    )
                    job.videos_failed += 1
                    return

                # Hand off to processing pipeline
                chunks = await self.pipeline.process(
                    video=video,
                    segments=segments,
                    extract_frames=extract_frames,
                    force_reingest=force_reingest,
                )

                job.videos_processed += 1
                job.chunks_created += len(chunks)
                job.progress_percent = min(
                    95.0,
                    5.0 + 90.0 * job.videos_processed / max(job.videos_total, 1),
                )

            except Exception as exc:
                logger.error(
                    "video_processing_failed",
                    video_id=video.video_id,
                    error=str(exc),
                )
                job.videos_failed += 1

    # ──────────────────────────────────────────────────────────────────────
    # yt-dlp helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _fetch_channel_videos(
        self, channel_url: str, max_videos: int
    ) -> List[VideoMetadata]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_channel_videos_sync,
            channel_url,
            max_videos,
        )

    def _fetch_channel_videos_sync(
        self, channel_url: str, max_videos: int
    ) -> List[VideoMetadata]:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,         # fast — no full download
            "playlistend": max_videos,
            "sleep_interval": self.settings.YTDLP_SLEEP_INTERVAL,
        }
        if self.settings.YTDLP_COOKIES_FILE:
            ydl_opts["cookiefile"] = self.settings.YTDLP_COOKIES_FILE

        videos: List[VideoMetadata] = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            entries = info.get("entries", [])

            for entry in entries:
                if not entry or entry.get("_type") == "url":
                    # For flat extraction we need to re-fetch each video
                    # to get duration etc — or we use what's available
                    pass
                try:
                    videos.append(
                        VideoMetadata(
                            video_id=entry.get("id", ""),
                            title=entry.get("title", "Untitled"),
                            channel_id=info.get("channel_id", info.get("id", "")),
                            channel_name=info.get(
                                "channel", info.get("uploader", "Unknown")
                            ),
                            url=f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                            duration_seconds=int(entry.get("duration") or 0),
                            published_at=entry.get("upload_date"),
                            view_count=entry.get("view_count"),
                            description=entry.get("description", "")[:500],
                            thumbnail_url=entry.get("thumbnail"),
                            tags=entry.get("tags", []) or [],
                        )
                    )
                except Exception as e:
                    logger.warning("video_metadata_parse_error", error=str(e))

        return videos

    # ──────────────────────────────────────────────────────────────────────
    # Transcript helpers
    # ──────────────────────────────────────────────────────────────────────

    @async_retry(max_attempts=3, delay=2.0, backoff=2.0)
    async def _fetch_transcript(
        self, video_id: str
    ) -> Optional[List[TranscriptSegment]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_transcript_sync,
            video_id,
        )

    def _fetch_transcript_sync(
        self, video_id: str
    ) -> Optional[List[TranscriptSegment]]:
        try:
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id)
            return [
                TranscriptSegment(
                    text=s.text,
                    start=s.start,
                    duration=s.duration,
                )
                for s in transcript
            ]
        except Exception as e:
            logger.error("transcript_fetch_error", video_id=video_id, error=str(e))
            return None

    
    async def ingest_video(
        self,
        video_url: str,
        force_reingest: bool = False,
        extract_frames: bool = False,
    ) -> str:
        job_id = str(uuid.uuid4())
        job = IngestionJobStatus(
            job_id=job_id,
            status=IngestionStatus.PENDING,
            started_at=datetime.utcnow(),
            videos_total=1,
        )
        _jobs[job_id] = job
        asyncio.create_task(
            self._run_single_video_ingestion(
                job_id, video_url, force_reingest, extract_frames
            )
        )
        logger.info("single_video_ingestion_started", job_id=job_id, url=video_url)
        return job_id

    async def _run_single_video_ingestion(
        self,
        job_id: str,
        video_url: str,
        force_reingest: bool,
        extract_frames: bool,
    ) -> None:
        job = _jobs[job_id]
        job.status = IngestionStatus.RUNNING
        try:
            video_id = self._extract_video_id(video_url)
            if not video_id:
                raise ValueError(f"Could not extract video ID from: {video_url}")
            loop = asyncio.get_event_loop()
            video = await loop.run_in_executor(
                self._executor,
                self._fetch_single_video_metadata,
                video_id,
                video_url,
            )
            job.channel_name = video.channel_name
            job.channel_id = video.channel_id
            job.progress_percent = 20.0
            segments = await self._fetch_transcript(video_id)
            if not segments:
                job.status = IngestionStatus.FAILED
                job.error_message = "No transcript available for this video"
                job.completed_at = datetime.utcnow()
                return
            job.progress_percent = 50.0
            chunks = await self.pipeline.process(
                video=video,
                segments=segments,
                extract_frames=extract_frames,
                force_reingest=force_reingest,
            )
            job.videos_processed = 1
            job.chunks_created = len(chunks)
            job.status = IngestionStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress_percent = 100.0
            logger.info("single_video_complete", job_id=job_id, chunks=len(chunks))
        except Exception as exc:
            logger.error("single_video_failed", job_id=job_id, error=str(exc))
            job.status = IngestionStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()

    def _extract_video_id(self, url: str) -> str:
        import re
        patterns = [r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})']
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return ""

    def _fetch_single_video_metadata(self, video_id: str, video_url: str):
        ydl_opts = {"quiet": True, "no_warnings": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return VideoMetadata(
                    video_id=video_id,
                    title=info.get("title", "Untitled"),
                    channel_id=info.get("channel_id", ""),
                    channel_name=info.get("channel", info.get("uploader", "Unknown")),
                    url=video_url,
                    duration_seconds=int(info.get("duration") or 0),
                    published_at=info.get("upload_date"),
                    view_count=info.get("view_count"),
                    description=info.get("description", "")[:500] if info.get("description") else "",
                    thumbnail_url=info.get("thumbnail"),
                    tags=info.get("tags", []) or [],
                )
        except Exception as e:
            logger.error("single_video_metadata_error", video_id=video_id, error=str(e))
            return VideoMetadata(
                video_id=video_id,
                title=f"Video {video_id}",
                channel_id="unknown",
                channel_name="Unknown",
                url=video_url,
                duration_seconds=0,
            )
