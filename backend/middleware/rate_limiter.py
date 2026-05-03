# backend/middleware/rate_limiter.py
"""
Token bucket rate limiter implemented as FastAPI middleware.
Limits requests per IP address within a sliding window.

For multi-worker / multi-instance deployments, swap the in-memory
store for a Redis-backed counter (e.g. using redis-py with INCR + EXPIRE).
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Dict, Tuple

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.config import get_settings
from backend.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter.
    State: {ip: [(timestamp, count), ...]}
    """

    def __init__(self, app, max_requests: int, window_seconds: int):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, list] = defaultdict(list)
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()

        with self._lock:
            # Purge old entries outside the window
            self._buckets[client_ip] = [
                ts
                for ts in self._buckets[client_ip]
                if now - ts < self.window_seconds
            ]

            if len(self._buckets[client_ip]) >= self.max_requests:
                oldest = self._buckets[client_ip][0]
                retry_after = self.window_seconds - (now - oldest)

                logger.warning(
                    "rate_limit_exceeded", ip=client_ip, path=request.url.path
                )

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after_seconds": round(retry_after, 1),
                    },
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )

            self._buckets[client_ip].append(now)

        response = await call_next(request)

        # Expose rate limit headers
        remaining = max(
            0,
            self.max_requests - len(self._buckets[client_ip]),
        )
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)

        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Extract real client IP, honoring X-Forwarded-For from reverse proxies.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
