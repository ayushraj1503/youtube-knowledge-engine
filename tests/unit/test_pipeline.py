# tests/unit/test_pipeline.py
"""Unit tests for processing pipeline components."""

import pytest
from unittest.mock import MagicMock, patch

from backend.models.schemas import TranscriptSegment, VideoMetadata
from backend.services.processing.pipeline import ProcessingPipeline


@pytest.fixture
def video():
    return VideoMetadata(
        video_id="test_vid_001",
        title="Introduction to Machine Learning",
        channel_id="UC_test",
        channel_name="Test Channel",
        url="https://www.youtube.com/watch?v=test_vid_001",
        duration_seconds=600,
    )


@pytest.fixture
def raw_segments():
    return [
        TranscriptSegment(text="Hello and welcome to this video", start=0.0, duration=3.0),
        TranscriptSegment(text="<c>Today</c> we [Music] will learn", start=3.0, duration=3.0),
        TranscriptSegment(text="about machine learning concepts", start=6.0, duration=3.0),
        TranscriptSegment(text="specifically neural networks", start=9.0, duration=3.0),
        TranscriptSegment(text="and deep learning techniques", start=12.0, duration=3.0),
    ]


@pytest.fixture
def pipeline():
    with patch("backend.services.processing.pipeline.EmbeddingService"):
        with patch("backend.services.processing.pipeline.VisualCaptioner"):
            with patch("backend.services.processing.pipeline.FrameExtractor"):
                p = ProcessingPipeline()
                return p


class TestTextCleaning:
    def test_strips_html_tags(self, pipeline, raw_segments):
        cleaned = pipeline._clean_segments(raw_segments)
        for seg in cleaned:
            assert "<" not in seg.text
            assert ">" not in seg.text

    def test_removes_music_annotations(self, pipeline, raw_segments):
        cleaned = pipeline._clean_segments(raw_segments)
        for seg in cleaned:
            assert "[Music]" not in seg.text

    def test_normalises_whitespace(self, pipeline):
        segs = [TranscriptSegment(text="hello   world  ", start=0.0, duration=1.0)]
        cleaned = pipeline._clean_segments(segs)
        assert cleaned[0].text == "hello world"

    def test_empty_segments_skipped(self, pipeline):
        segs = [
            TranscriptSegment(text="[Music]", start=0.0, duration=1.0),
            TranscriptSegment(text="actual text here", start=1.0, duration=2.0),
        ]
        cleaned = pipeline._clean_segments(segs)
        assert len(cleaned) == 1
        assert cleaned[0].text == "actual text here"


class TestChunking:
    def test_produces_chunks(self, pipeline, video, raw_segments):
        cleaned = pipeline._clean_segments(raw_segments)
        chunks = pipeline._chunk_segments(video, cleaned)
        assert len(chunks) > 0

    def test_chunks_have_timestamps(self, pipeline, video, raw_segments):
        cleaned = pipeline._clean_segments(raw_segments)
        chunks = pipeline._chunk_segments(video, cleaned)
        for chunk in chunks:
            assert chunk.start_time >= 0.0
            assert chunk.end_time >= chunk.start_time

    def test_chunks_have_video_metadata(self, pipeline, video, raw_segments):
        cleaned = pipeline._clean_segments(raw_segments)
        chunks = pipeline._chunk_segments(video, cleaned)
        for chunk in chunks:
            assert chunk.video_id == video.video_id
            assert chunk.video_title == video.title
            assert chunk.channel_name == video.channel_name

    def test_chunk_ids_are_unique(self, pipeline, video, raw_segments):
        cleaned = pipeline._clean_segments(raw_segments)
        chunks = pipeline._chunk_segments(video, cleaned)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_total_chunks_back_filled(self, pipeline, video, raw_segments):
        cleaned = pipeline._clean_segments(raw_segments)
        chunks = pipeline._chunk_segments(video, cleaned)
        expected_total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == expected_total


# tests/unit/test_retrieval.py
class TestTimestampUrl:
    def test_timestamp_url_format(self):
        from backend.services.retrieval.retrieval_service import _make_timestamp_url

        url = _make_timestamp_url("dQw4w9WgXcQ", 125.5)
        assert "youtube.com/watch" in url
        assert "v=dQw4w9WgXcQ" in url
        assert "t=125s" in url

    def test_timestamp_url_zero(self):
        from backend.services.retrieval.retrieval_service import _make_timestamp_url

        url = _make_timestamp_url("abc123", 0.0)
        assert "t=0s" in url


# tests/unit/test_schemas.py
class TestSchemas:
    def test_ingest_request_validates_url(self):
        from backend.models.schemas import IngestChannelRequest

        req = IngestChannelRequest(channel_url="https://www.youtube.com/@test")
        assert req.channel_url == "https://www.youtube.com/@test"
        assert req.force_reingest is False

    def test_query_request_validates_min_length(self):
        from backend.models.schemas import QueryRequest
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            QueryRequest(query="ab")  # too short

    def test_hybrid_alpha_validation(self):
        from backend.core.config import Settings
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            # Should fail — alpha must be 0.0-1.0
            Settings(
                GROQ_API_KEY="test",
                HYBRID_ALPHA=1.5,
            )
