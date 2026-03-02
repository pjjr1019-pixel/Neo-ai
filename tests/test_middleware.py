"""Tests for NEO middleware module.

Covers:
- CorrelationIDMiddleware: header injection, reuse, UUID generation
- RequestLoggingMiddleware: timing, log levels, skip paths
- register_exception_handlers: NeoBaseError → JSON responses
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from python_ai.exceptions import (
    NeoAuthError,
    NeoDataError,
    NeoModelError,
    NeoModelNotTrainedError,
    NeoRateLimitError,
)
from python_ai.middleware import (
    CorrelationIDMiddleware,
    RequestLoggingMiddleware,
    register_exception_handlers,
)


# ── Helpers ───────────────────────────────────────────────────


def _make_app(
    *,
    correlation: bool = True,
    logging: bool = False,
    exception_handlers: bool = False,
) -> FastAPI:
    """Build a minimal FastAPI app with selected middleware."""
    app = FastAPI()

    if exception_handlers:
        register_exception_handlers(app)

    if logging:
        app.add_middleware(RequestLoggingMiddleware)

    if correlation:
        app.add_middleware(CorrelationIDMiddleware)

    @app.get("/ping")
    def _ping():
        return {"ok": True}

    @app.get("/health")
    def _health():
        return {"status": "healthy"}

    return app


# ── CorrelationIDMiddleware tests ─────────────────────────────


class TestCorrelationIDMiddleware:
    """Tests for X-Correlation-ID injection."""

    def test_generates_uuid_when_absent(self) -> None:
        """A UUID-4 is generated when no header is sent."""
        client = TestClient(_make_app())
        resp = client.get("/ping")

        assert resp.status_code == 200
        cid = resp.headers.get("X-Correlation-ID")
        assert cid is not None
        # Must be a valid UUID-4
        parsed = uuid.UUID(cid, version=4)
        assert str(parsed) == cid

    def test_reuses_client_correlation_id(self) -> None:
        """If the client sends X-Correlation-ID it is echoed back."""
        client = TestClient(_make_app())
        custom_cid = "my-custom-id-12345"
        resp = client.get(
            "/ping",
            headers={"X-Correlation-ID": custom_cid},
        )

        assert resp.status_code == 200
        assert resp.headers["X-Correlation-ID"] == custom_cid

    def test_different_requests_get_different_ids(self) -> None:
        """Each request without a header gets a unique UUID."""
        client = TestClient(_make_app())
        ids = {
            client.get("/ping").headers["X-Correlation-ID"]
            for _ in range(5)
        }
        assert len(ids) == 5


# ── RequestLoggingMiddleware tests ────────────────────────────


class TestRequestLoggingMiddleware:
    """Tests for request/response logging."""

    def test_skips_health_endpoint(self, caplog) -> None:
        """Requests to /health should not be logged."""
        app = _make_app(correlation=False, logging=True)
        client = TestClient(app)

        with caplog.at_level("DEBUG", logger="python_ai.middleware"):
            client.get("/health")

        assert not any(
            "/health" in rec.message for rec in caplog.records
        )

    def test_logs_normal_request(self, caplog) -> None:
        """A normal request should produce an INFO log line."""
        app = _make_app(correlation=False, logging=True)
        client = TestClient(app)

        with caplog.at_level("DEBUG", logger="python_ai.middleware"):
            resp = client.get("/ping")

        assert resp.status_code == 200
        info_records = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "/ping" in r.message
        ]
        assert len(info_records) >= 1


# ── Exception handler tests ──────────────────────────────────


class TestExceptionHandlers:
    """Tests for NeoBaseError exception handler registration."""

    @staticmethod
    def _app_raising(exc):
        """Build an app whose /fail endpoint raises *exc*."""
        app = FastAPI()
        register_exception_handlers(app)
        app.add_middleware(CorrelationIDMiddleware)

        @app.get("/fail")
        def _fail():
            raise exc

        return app

    def test_model_error_returns_500(self) -> None:
        """NeoModelError should yield HTTP 500."""
        app = self._app_raising(
            NeoModelError("model broken", context={"v": 1})
        )
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")

        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "NeoModelError"
        assert body["code"] == "NEO-500"
        assert body["message"] == "model broken"
        assert body["detail"] == {"v": 1}
        assert body["request_id"] is not None

    def test_auth_error_returns_401(self) -> None:
        """NeoAuthError should yield HTTP 401."""
        app = self._app_raising(NeoAuthError("bad token"))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")

        assert resp.status_code == 401
        body = resp.json()
        assert body["type"] == "NeoAuthError"
        assert body["code"] == "NEO-401"

    def test_rate_limit_error_returns_429(self) -> None:
        """NeoRateLimitError should yield HTTP 429."""
        app = self._app_raising(NeoRateLimitError("slow down"))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")

        assert resp.status_code == 429
        assert resp.json()["code"] == "NEO-429"

    def test_data_error_returns_422(self) -> None:
        """NeoDataError should yield HTTP 422."""
        app = self._app_raising(NeoDataError("bad data"))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")

        assert resp.status_code == 422
        assert resp.json()["code"] == "NEO-422"

    def test_subclass_inherits_parent_code(self) -> None:
        """NeoModelNotTrainedError (subclass) maps via MRO."""
        app = self._app_raising(
            NeoModelNotTrainedError("not trained yet")
        )
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")

        # Subclass of NeoModelError → 500
        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "NeoModelNotTrainedError"
        # Code should be NEO-500 (from NeoModelError parent)
        assert body["code"] == "NEO-500"

    def test_error_body_has_no_detail_when_empty(self) -> None:
        """When context is empty, detail should be None."""
        app = self._app_raising(NeoModelError("oops"))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")

        assert resp.json()["detail"] is None

    def test_correlation_id_in_error_response(self) -> None:
        """Error response should include the request's correlation ID."""
        app = self._app_raising(NeoModelError("boom"))
        client = TestClient(app, raise_server_exceptions=False)
        custom_cid = "err-cid-123"
        resp = client.get(
            "/fail",
            headers={"X-Correlation-ID": custom_cid},
        )

        body = resp.json()
        assert body["request_id"] == custom_cid
        assert resp.headers["X-Correlation-ID"] == custom_cid
