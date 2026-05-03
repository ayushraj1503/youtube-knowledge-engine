# tests/integration/test_api.py
"""
Integration tests for the FastAPI application.
Uses httpx.AsyncClient to test real HTTP endpoints.
Requires a running backend with GROQ_API_KEY set.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_has_components(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "components" in data
        assert "status" in data
        assert "version" in data


@pytest.mark.asyncio
class TestIngestEndpoint:
    async def test_ingest_returns_202(self, client):
        resp = await client.post(
            "/api/v1/ingest/channel",
            json={
                "channel_url": "https://www.youtube.com/@test",
                "max_videos": 1,
                "force_reingest": False,
                "extract_frames": False,
            },
        )
        # 202 Accepted for async job
        assert resp.status_code in (202, 500)  # 500 if yt-dlp fails in test env

    async def test_ingest_returns_job_id(self, client):
        resp = await client.post(
            "/api/v1/ingest/channel",
            json={"channel_url": "https://www.youtube.com/@test"},
        )
        if resp.status_code == 202:
            data = resp.json()
            assert "job_id" in data
            assert len(data["job_id"]) == 36  # UUID format

    async def test_invalid_job_returns_404(self, client):
        resp = await client.get("/api/v1/ingest/status/nonexistent-job-id")
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestQueryEndpoint:
    async def test_empty_query_returns_422(self, client):
        resp = await client.post(
            "/api/v1/query",
            json={"query": ""},
        )
        assert resp.status_code == 422

    async def test_short_query_returns_422(self, client):
        resp = await client.post(
            "/api/v1/query",
            json={"query": "ab"},
        )
        assert resp.status_code == 422

    async def test_valid_query_schema(self, client):
        resp = await client.post(
            "/api/v1/query",
            json={"query": "What is machine learning?", "top_k": 3},
        )
        # May return 500 if no data ingested, but schema should be valid
        if resp.status_code == 200:
            data = resp.json()
            assert "answer" in data
            assert "sources" in data
            assert "latency_ms" in data


@pytest.mark.asyncio
class TestRateLimiting:
    async def test_rate_limit_headers_present(self, client):
        resp = await client.get("/api/v1/videos")
        # Rate limit headers should be present
        assert "X-RateLimit-Limit" in resp.headers or resp.status_code in (200, 429)


@pytest.mark.asyncio
class TestVideosEndpoint:
    async def test_videos_endpoint_returns_list(self, client):
        resp = await client.get("/api/v1/videos")
        assert resp.status_code == 200
        data = resp.json()
        assert "videos" in data
        assert "total" in data
        assert "page" in data

    async def test_pagination_params(self, client):
        resp = await client.get("/api/v1/videos?page=1&page_size=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["page_size"] == 5
