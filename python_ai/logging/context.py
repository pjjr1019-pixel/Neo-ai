"""Logging context management for request tracing."""

import contextvars
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("correlation_id", default=None)
)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID.

    Returns:
        Correlation ID string or None if not set.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current context.

    Args:
        correlation_id: ID to set. Generates UUID if None.

    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _correlation_id.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear correlation ID from context."""
    _correlation_id.set(None)


@contextmanager
def correlation_id_context(
    correlation_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """Context manager for correlation ID scope.

    Args:
        correlation_id: ID to use. Generates UUID if None.

    Yields:
        The correlation ID.

    Example:
        with correlation_id_context("req-123") as cid:
            logger.info("Processing request")  # Includes cid
    """
    token = _correlation_id.set(correlation_id or str(uuid.uuid4()))
    try:
        cid = _correlation_id.get()
        yield cid or ""
    finally:
        _correlation_id.reset(token)


@dataclass
class LogContext:
    """Structured logging context for operations.

    Tracks operation timing and context for structured logging.

    Example:
        with LogContext(operation="train_model", model_id="123") as ctx:
            # ... do training ...
            ctx.add_result("accuracy", 0.95)
        # Logs timing and results automatically
    """

    operation: str
    correlation_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    _start_time: Optional[datetime] = field(
        default=None, init=False, repr=False
    )
    _end_time: Optional[datetime] = field(default=None, init=False, repr=False)
    _results: Dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def __enter__(self) -> "LogContext":
        """Enter context and start timing."""
        from python_ai.logging.logger import get_logger

        self._start_time = datetime.now(timezone.utc)

        # Set correlation ID if provided
        if self.correlation_id:
            set_correlation_id(self.correlation_id)
        elif get_correlation_id() is None:
            set_correlation_id()

        self._logger = get_logger(self.operation, self.extra)
        self._logger.info(
            f"Starting {self.operation}",
            extra={"event": "start"},
        )

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Exit context and log results."""
        self._end_time = datetime.now(timezone.utc)
        if self._start_time:
            delta = self._end_time - self._start_time
            duration_ms = delta.total_seconds() * 1000
        else:
            duration_ms = 0.0

        log_data = {
            "event": "complete",
            "duration_ms": round(duration_ms, 2),
            "results": self._results,
        }

        if exc_type is not None:
            log_data["event"] = "error"
            log_data["error_type"] = exc_type.__name__
            log_data["error_message"] = str(exc_val)
            self._logger.error(
                f"Failed {self.operation}: {exc_val}",
                extra=log_data,
                exc_info=True,
            )
        else:
            self._logger.info(
                f"Completed {self.operation} in {duration_ms:.2f}ms",
                extra=log_data,
            )

    def add_result(self, key: str, value: Any) -> None:
        """Add result to log context.

        Args:
            key: Result key.
            value: Result value.
        """
        self._results[key] = value

    @property
    def duration_ms(self) -> Optional[float]:
        """Get operation duration in milliseconds."""
        if self._start_time is None:
            return None
        end = self._end_time or datetime.now(timezone.utc)
        return (end - self._start_time).total_seconds() * 1000


class RequestContext:
    """HTTP request context for web frameworks.

    Tracks request metadata for logging.
    """

    def __init__(
        self,
        method: str,
        path: str,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
    ):
        """Initialize request context.

        Args:
            method: HTTP method.
            path: Request path.
            correlation_id: Request correlation ID.
            user_id: Authenticated user ID.
            client_ip: Client IP address.
        """
        self.method = method
        self.path = path
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.user_id = user_id
        self.client_ip = client_ip
        self.start_time = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging.

        Returns:
            Dict of request context.
        """
        return {
            "method": self.method,
            "path": self.path,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "client_ip": self.client_ip,
        }

    def log_response(
        self,
        status_code: int,
        response_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create response log data.

        Args:
            status_code: HTTP response status.
            response_size: Response body size in bytes.

        Returns:
            Dict of response data for logging.
        """
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - self.start_time).total_seconds() * 1000

        return {
            **self.to_dict(),
            "status_code": status_code,
            "response_size": response_size,
            "duration_ms": round(duration_ms, 2),
        }
