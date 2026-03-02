"""
Rate-limiting middleware for FastAPI.

Uses a token-bucket algorithm keyed by client IP.  Fully
in-process — no Redis dependency required (but could be
extended to use Redis for multi-worker deployments).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Dict

from fastapi import Request, Response

if TYPE_CHECKING:
    from fastapi import FastAPI
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class TokenBucket:
    """Thread-safe token bucket.

    Attributes:
        capacity: Maximum tokens the bucket can hold.
        refill_rate: Tokens added per second.
    """

    def __init__(
        self,
        capacity: int = 60,
        refill_rate: float = 1.0,
    ) -> None:
        """Initialise the bucket.

        Args:
            capacity: Max tokens (burst size).
            refill_rate: Tokens replenished per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume *tokens*.  Returns True on success."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(
                self.capacity,
                self._tokens + elapsed * self.refill_rate,
            )
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available(self) -> float:
        """Tokens currently available (informational)."""
        return self._tokens


class RateLimiter:
    """Per-IP rate limiter backed by token buckets.

    Attributes:
        capacity: Bucket capacity per IP.
        refill_rate: Refill rate per IP (tokens/sec).
    """

    def __init__(
        self,
        capacity: int = 60,
        refill_rate: float = 1.0,
    ) -> None:
        """Initialise the rate limiter.

        Args:
            capacity: Max burst per IP.
            refill_rate: Sustained rate per IP.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return True if *key* is within its rate limit."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    self.capacity,
                    self.refill_rate,
                )
        return self._buckets[key].consume()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that applies per-IP rate limiting.

    Attach to a FastAPI app::

        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
        )
    """

    def __init__(
        self,
        app: FastAPI,  # type: ignore[override]
        requests_per_minute: int = 60,
    ) -> None:
        """Initialise middleware.

        Args:
            app: The ASGI application.
            requests_per_minute: Allowed requests per minute per IP.
        """
        super().__init__(app)
        capacity = requests_per_minute
        refill_rate = requests_per_minute / 60.0
        self.limiter = RateLimiter(
            capacity=capacity,
            refill_rate=refill_rate,
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Check rate limit before forwarding the request."""
        client_ip = request.client.host if request.client else "unknown"

        if not self.limiter.allow(client_ip):
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Try again later.",
                },
            )

        response = await call_next(request)
        return response
