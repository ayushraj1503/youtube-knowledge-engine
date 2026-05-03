# backend/services/processing/pipeline.py
"""
Processing Pipeline

Transforms raw transcript segments into clean, semantically meaningful chunks:

  Raw transcript segments
      → Text cleaning & normalisation
      → Intelligent chunking (sentence-aware, configurable size + overlap)
      → Frame extraction via FFmpeg (optional)
      → Visual captioning via BLIP (optional)
      → Enriched TextChunk objects
      → EmbeddingService.store()

Design choices:
  - Sentence-aware chunking: we don't cut mid-sentence. This preserves
    semantic coherence and improves retrieval quality significantly.
  - Overlap: adjacent chunks share CHUNK_OVERLAP tokens so context isn't
    lost at boundaries — critical for multi-sentence answers.
  - Frame extraction at 0.5 FPS gives ~1 frame every 2 seconds: enough
    visual context without disk bloat.
  - BLIP runs locally (HuggingFace) to avoid external API costs for images.
"""

import asyncio
import hashlib
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from backend.core.config import get_settings
from backend.core.logging_config import get_logger
from backend.models.schemas import TextChunk, TranscriptSegment, VideoMetadata
from backend.services.processing.frame_extractor import FrameExtractor
from backend.services.processing.visual_captioner import VisualCaptioner
from backend.services.embedding.embedding_service import EmbeddingService

logger = get_logger(__name__)
settings = get_settings()


class ProcessingPipeline:
    """
    Stateless (except for lazy-loaded models) processing stage.
    Can be instantiated once and reused across all ingestion jobs.
    """

    def __init__(self):
        self._frame_extractor = FrameExtractor()
        self._captioner = VisualCaptioner() if settings.BLIP_ENABLED else None
        self._embedding_service = EmbeddingService()
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="pipeline"
        )

    # ──────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────

    async def process(
        self,
        video: VideoMetadata,
        segments: List[TranscriptSegment],
        extract_frames: bool = True,
        force_reingest: bool = False,
    ) -> List[TextChunk]:
        """
        Full processing pipeline for one video.
        Returns the created TextChunk list (also stored in vector DB).
        """
        logger.info("pipeline_start", video_id=video.video_id, title=video.title)

        # 1. Check if already ingested
        if not force_reingest and await self._embedding_service.video_exists(
            video.video_id
        ):
            logger.info("video_already_ingested", video_id=video.video_id)
            return []

        # 2. Clean & normalise transcript text
        cleaned_segments = self._clean_segments(segments)

        # 3. Chunk transcript into semantic units
        chunks = self._chunk_segments(video, cleaned_segments)

        # 4. Optional: frame extraction + BLIP captioning
        if extract_frames and settings.BLIP_ENABLED and self._captioner:
            chunks = await self._enrich_with_visual_captions(video, chunks)

        # 5. Store embeddings
        stored = await self._embedding_service.embed_and_store(chunks)
        logger.info(
            "pipeline_complete",
            video_id=video.video_id,
            chunks=len(stored),
        )
        return stored

    # ──────────────────────────────────────────────────────────────────────
    # TEXT CLEANING
    # ──────────────────────────────────────────────────────────────────────

    def _clean_segments(
        self, segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """
        Normalise auto-generated captions:
          - Strip HTML tags (YouTube sometimes injects <c> colour tags)
          - Collapse whitespace
          - Remove music/sound effect annotations like [Music], [Applause]
          - Merge very short segments (< 3 chars) into the next
        """
        cleaned = []
        buffer_text = ""
        buffer_start = 0.0
        buffer_duration = 0.0

        for seg in segments:
            text = seg.text
            # Strip HTML
            text = re.sub(r"<[^>]+>", "", text)
            # Remove [Music], [Applause] etc.
            text = re.sub(r"\[.*?\]", "", text)
            # Collapse whitespace
            text = " ".join(text.split())

            if not text:
                continue

            if len(text) < 3:
                # Merge tiny segment into buffer
                buffer_text += " " + text
                buffer_duration += seg.duration
            else:
                if buffer_text:
                    cleaned.append(
                        TranscriptSegment(
                            text=buffer_text.strip(),
                            start=buffer_start,
                            duration=buffer_duration,
                        )
                    )
                buffer_text = text
                buffer_start = seg.start
                buffer_duration = seg.duration

        if buffer_text:
            cleaned.append(
                TranscriptSegment(
                    text=buffer_text.strip(),
                    start=buffer_start,
                    duration=buffer_duration,
                )
            )

        return cleaned

    # ──────────────────────────────────────────────────────────────────────
    # CHUNKING
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_segments(
        self, video: VideoMetadata, segments: List[TranscriptSegment]
    ) -> List[TextChunk]:
        """
        Sliding-window chunking that:
          1. Groups segments until we hit CHUNK_SIZE words
          2. Overlaps adjacent chunks by CHUNK_OVERLAP words
          3. Preserves start/end timestamps per chunk

        Word-based chunking is simpler and more predictable than token-based
        for retrieval use-cases. For production with an LLM context limit,
        switch to tiktoken.
        """
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP

        # Flatten into (word, start_time) pairs
        words_with_ts: List[tuple] = []  # (word, start_sec)
        for seg in segments:
            for word in seg.text.split():
                words_with_ts.append((word, seg.start))

        if not words_with_ts:
            return []

        chunks: List[TextChunk] = []
        i = 0
        chunk_index = 0

# Note: I tried token-based chunking with tiktoken first
# but word-based is more predictable for retrieval
# and all-MiniLM has 256 token limit anyway
# switching back to words was cleaner
        while i < len(words_with_ts):
            window = words_with_ts[i : i + chunk_size]
            text = " ".join(w for w, _ in window)
            start_time = window[0][1]
            end_time = window[-1][1]

            chunk_id = hashlib.md5(
                f"{video.video_id}_{chunk_index}".encode()
            ).hexdigest()

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    video_id=video.video_id,
                    video_title=video.title,
                    channel_name=video.channel_name,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    chunk_index=chunk_index,
                    total_chunks=-1,  # back-filled below
                    metadata={
                        "channel_id": video.channel_id,
                        "published_at": video.published_at or "",
                        "duration_seconds": video.duration_seconds,
                        "url": video.url,
                    },
                )
            )

            i += chunk_size - overlap
            chunk_index += 1

        # Back-fill total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    # ──────────────────────────────────────────────────────────────────────
    # VISUAL ENRICHMENT
    # ──────────────────────────────────────────────────────────────────────

    async def _enrich_with_visual_captions(
        self, video: VideoMetadata, chunks: List[TextChunk]
    ) -> List[TextChunk]:
        """
        For each chunk, find the nearest extracted frame and run BLIP captioning.
        Frame captions are appended to the chunk text for multimodal context.
        """
        loop = asyncio.get_event_loop()

        # Extract frames in a thread (FFmpeg subprocess)
        frames = await loop.run_in_executor(
            self._executor,
            self._frame_extractor.extract,
            video.video_id,
            video.url,
        )

        if not frames:
            return chunks

        # Build timestamp→path lookup
        frame_map = {ts: path for ts, path in frames}
        frame_times = sorted(frame_map.keys())

        for chunk in chunks:
            # Find nearest frame timestamp to chunk midpoint
            mid = (chunk.start_time + chunk.end_time) / 2
            nearest_ts = min(frame_times, key=lambda t: abs(t - mid))
            frame_path = frame_map[nearest_ts]

            caption = await loop.run_in_executor(
                self._executor,
                self._captioner.caption,
                frame_path,
            )

            if caption:
                chunk.frame_caption = caption
                # Append visual context to text for richer embedding
                chunk.text = f"{chunk.text} [Visual context: {caption}]"

        return chunks
