"""
FastAPI middleware for NEO Hybrid AI.

Provides:
- Correlation-ID injection (``X-Correlation-ID`` header)
- Request/response logging with timing
- Consistent error-response handling for ``NeoBaseError`` exceptions

All middleware conforms to Starlette ``BaseHTTPMiddleware`` protocol.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import Response

from python_ai.exceptions import (
    NeoAPIError,
    NeoAuthError,
    NeoBaseError,
    NeoConfigError,
    NeoDataError,
    NeoModelError,
    NeoRateLimitError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CorrelationIDMiddleware",
    "RequestLoggingMiddleware",
    "register_exception_handlers",
]

# ── HTTP status mapping for exception types ───────────────────

_STATUS_MAP: Dict[type, int] = {
    NeoRateLimitError: 429,
    NeoAuthError: 401,
    NeoDataError: 422,
    NeoModelError: 500,
    NeoAPIError: 400,
    NeoConfigError: 500,
    NeoBaseError: 500,
}

# ── Error-code prefix for each category ───────────────────────

_ERROR_PREFIX: Dict[type, str] = {
    NeoRateLimitError: "NEO-429",
    NeoAuthError: "NEO-401",
    NeoDataError: "NEO-422",
    NeoModelError: "NEO-500",
    NeoAPIError: "NEO-400",
    NeoConfigError: "NEO-503",
    NeoBaseError: "NEO-999",
}


# ── Correlation-ID middleware ─────────────────────────────────


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Inject ``X-Correlation-ID`` into every request/response.

    If the client sends a correlation ID header it is re-used;
    otherwise a UUID-4 is generated.  The ID is attached to the
    ``request.state`` for downstream logging.
    """

    HEADER = "X-Correlation-ID"

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request, inject correlation ID."""
        cid = request.headers.get(self.HEADER) or str(uuid.uuid4())
        request.state.correlation_id = cid

        response = await call_next(request)
        response.headers[self.HEADER] = cid
        return response


# ── Request/Response logging middleware ───────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status, and duration for every request.

    Skips ``/health`` and ``/docs`` to reduce log noise.
    Logs at INFO level for 2xx/3xx, WARNING for 4xx, ERROR for 5xx.
    """

    _SKIP_PATHS = frozenset({"/health", "/docs", "/openapi.json"})

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Log timing and status for the request."""
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        cid = getattr(request.state, "correlation_id", "-")
        msg = (
            "%s %s -> %d (%.1fms) [cid=%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            cid,
        )

        if response.status_code >= 500:
            logger.error(*msg)
        elif response.status_code >= 400:
            logger.warning(*msg)
        else:
            logger.info(*msg)

        return response


# ── Exception handlers ────────────────────────────────────────


def _build_error_body(
    exc: NeoBaseError,
    request: Request,
) -> Dict[str, Any]:
    """Build the standard JSON error body.

    Schema::

        {
          "type": "NeoModelError",
          "code": "NEO-500",
          "message": "Model is not trained",
          "detail": { ... context ... },
          "request_id": "<correlation-id>"
        }
    """
    exc_type = type(exc)
    code = _ERROR_PREFIX.get(exc_type, "NEO-999")
    # Walk the MRO to find the most specific matching prefix
    for cls in exc_type.__mro__:
        if cls in _ERROR_PREFIX:
            code = _ERROR_PREFIX[cls]
            break

    cid = getattr(getattr(request, "state", None), "correlation_id", None)
    return {
        "type": exc_type.__name__,
        "code": code,
        "message": exc.detail,
        "detail": exc.context if exc.context else None,
        "request_id": cid,
    }


def _status_for(exc: NeoBaseError) -> int:
    """Return the HTTP status code for the given exception."""
    for cls in type(exc).__mro__:
        if cls in _STATUS_MAP:
            return _STATUS_MAP[cls]
    return 500


def register_exception_handlers(application: FastAPI) -> None:
    """Register ``NeoBaseError`` exception handlers on *application*.

    Call this once during app startup to ensure all Neo-domain
    exceptions are caught and returned in the standard error format.
    """

    @application.exception_handler(NeoBaseError)
    async def _handle_neo_error(
        request: Request,
        exc: NeoBaseError,
    ) -> JSONResponse:
        status = _status_for(exc)
        body = _build_error_body(exc, request)
        logger.error(
            "NeoBaseError [%s] %s: %s (cid=%s)",
            body["code"],
            body["type"],
            exc.detail,
            body["request_id"],
        )
        return JSONResponse(status_code=status, content=body)
