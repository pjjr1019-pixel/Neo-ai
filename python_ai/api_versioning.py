"""
API Versioning Middleware for NEO Hybrid AI.

Provides a lightweight ASGI middleware that routes
requests to versioned handlers based on the URL prefix
(e.g. ``/v1/predict``, ``/v2/predict``).
"""

import logging
import re
from typing import Any, Callable, Dict, Tuple

logger = logging.getLogger(__name__)

# Matches /v<major> at the start of the path
_VERSION_RE = re.compile(r"^/v(\d+)")


class APIVersionMiddleware:
    """ASGI middleware for URL-prefix-based versioning.

    Strips the ``/vN`` prefix from the path before
    forwarding to the inner ASGI app, and injects the
    version number into the ASGI scope.

    Optionally enforces a minimum and maximum supported
    version, returning 400 for unsupported versions.

    Args:
        app: The inner ASGI application.
        default_version: Version assumed when no prefix
            is present (default 1).
        min_version: Lowest supported version.
        max_version: Highest supported version.
    """

    def __init__(
        self,
        app: Any,
        default_version: int = 1,
        min_version: int = 1,
        max_version: int = 1,
    ) -> None:
        """Initialise the middleware."""
        self.app = app
        self._default = default_version
        self._min = min_version
        self._max = max_version

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        """Process an ASGI connection.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "/")
        version, new_path = self._extract_version(path)

        if version < self._min or version > self._max:
            await self._send_error(
                send,
                400,
                f"API version v{version} not supported "
                f"(range: v{self._min}–v{self._max})",
            )
            return

        scope["path"] = new_path
        scope["api_version"] = version
        await self.app(scope, receive, send)

    def _extract_version(self, path: str) -> Tuple[int, str]:
        """Parse version prefix from URL path.

        Args:
            path: Original request path.

        Returns:
            ``(version_int, stripped_path)``
        """
        match = _VERSION_RE.match(path)
        if match:
            ver = int(match.group(1))
            stripped = path[match.end() :]
            if not stripped:
                stripped = "/"
            return ver, stripped
        return self._default, path

    @staticmethod
    async def _send_error(
        send: Callable[..., Any],
        status: int,
        message: str,
    ) -> None:
        """Send a plain-text error response.

        Args:
            send: ASGI send callable.
            status: HTTP status code.
            message: Error body text.
        """
        body = message.encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"text/plain"],
                    [
                        b"content-length",
                        str(len(body)).encode(),
                    ],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


def get_api_version(scope: Dict[str, Any]) -> int:
    """Read the API version injected by the middleware.

    Args:
        scope: ASGI/Starlette request scope.

    Returns:
        Integer API version (defaults to 1).
    """
    return int(scope.get("api_version", 1))
