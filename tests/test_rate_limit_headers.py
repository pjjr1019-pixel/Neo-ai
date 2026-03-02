"""Tests for rate-limit headers and allow_with_info.

Covers:
- RateLimiter.allow_with_info: returns (allowed, remaining, capacity)
- RateLimitMiddleware: X-RateLimit-* headers on success and 429
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from python_ai.rate_limiter import RateLimitMiddleware, RateLimiter


# ── RateLimiter.allow_with_info ──────────────────────────────


class TestAllowWithInfo:
    """Tests for RateLimiter.allow_with_info."""

    def test_first_call_allowed(self):
        """First call is allowed with remaining < capacity."""
        rl = RateLimiter(capacity=10, refill_rate=1.0)
        allowed, remaining, cap = rl.allow_with_info("user1")
        assert allowed is True
        assert cap == 10
        assert 0 <= remaining <= 10

    def test_exhausted_returns_false(self):
        """After exhausting tokens, allowed is False."""
        rl = RateLimiter(capacity=2, refill_rate=0.01)
        rl.allow_with_info("user1")
        rl.allow_with_info("user1")
        allowed, remaining, cap = rl.allow_with_info("user1")
        assert allowed is False
        assert remaining == 0

    def test_capacity_matches(self):
        """Reported capacity matches configured value."""
        rl = RateLimiter(capacity=42, refill_rate=1.0)
        _, _, cap = rl.allow_with_info("key")
        assert cap == 42

    def test_per_key_tracking(self):
        """Different keys have independent buckets."""
        rl = RateLimiter(capacity=1, refill_rate=0.01)
        a1, _, _ = rl.allow_with_info("a")
        b1, _, _ = rl.allow_with_info("b")
        assert a1 is True
        assert b1 is True


# ── RateLimitMiddleware headers ──────────────────────────────


def _make_app(rpm: int = 60) -> FastAPI:
    """Build a tiny FastAPI app with rate limiting."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)

    @app.get("/ping")
    def _ping():
        """Health check."""
        return {"ok": True}

    return app


class TestRateLimitHeaders:
    """Tests for X-RateLimit-* headers."""

    def test_success_has_limit_header(self):
        """Successful responses include X-RateLimit-Limit."""
        client = TestClient(_make_app(rpm=60))
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert resp.headers["X-RateLimit-Limit"] == "60"

    def test_success_has_remaining_header(self):
        """Successful responses include X-RateLimit-Remaining."""
        client = TestClient(_make_app(rpm=60))
        resp = client.get("/ping")
        assert "X-RateLimit-Remaining" in resp.headers
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        assert 0 <= remaining <= 60

    def test_429_has_retry_after(self):
        """429 response includes Retry-After header."""
        app = _make_app(rpm=2)
        client = TestClient(app)
        client.get("/ping")
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_429_remaining_is_zero(self):
        """429 response shows X-RateLimit-Remaining: 0."""
        app = _make_app(rpm=1)
        client = TestClient(app)
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429
        assert resp.headers["X-RateLimit-Remaining"] == "0"

    def test_429_body(self):
        """429 response JSON body has expected detail."""
        app = _make_app(rpm=1)
        client = TestClient(app)
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]
