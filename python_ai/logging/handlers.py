"""Custom logging handlers and formatters."""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Outputs logs as single-line JSON objects suitable for
    log aggregation systems like ELK, Splunk, or CloudWatch.
    """

    def __init__(
        self,
        include_traceback: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """Initialize JSON formatter.

        Args:
            include_traceback: Include full traceback on exceptions.
            extra_fields: Static fields to include in every log.
        """
        super().__init__()
        self.include_traceback = include_traceback
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: Log record to format.

        Returns:
            JSON-formatted log string.
        """
        from python_ai.logging.context import get_correlation_id

        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            log_obj["correlation_id"] = correlation_id

        # Add extra fields
        log_obj.update(self.extra_fields)

        # Add custom fields from extra dict
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_obj.update(record.extra)

        # Add exception info
        if record.exc_info and self.include_traceback:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,  # type: ignore
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_obj, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development.

    Uses ANSI colors to highlight log levels for easier reading.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(
        self,
        include_timestamp: bool = True,
        use_colors: Optional[bool] = None,
    ):
        """Initialize colored formatter.

        Args:
            include_timestamp: Include timestamp in output.
            use_colors: Force color usage. Auto-detect if None.
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.use_colors = (
            use_colors if use_colors is not None else self._supports_color()
        )

    @staticmethod
    def _supports_color() -> bool:
        """Check if terminal supports colors."""
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        # Windows needs special handling
        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except Exception:
                return False
        return True

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format.

        Returns:
            Colored log string.
        """
        from python_ai.logging.context import get_correlation_id

        parts = []

        if self.include_timestamp:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"[{ts}]")

        # Add correlation ID if present
        correlation_id = get_correlation_id()
        if correlation_id:
            parts.append(f"[{correlation_id[:8]}]")

        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{self.BOLD}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        parts.append(level_str)
        parts.append(f"{record.name}:")
        parts.append(record.getMessage())

        message = " ".join(parts)

        # Add exception info
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            if self.use_colors:
                exc_text = f"{self.COLORS['ERROR']}{exc_text}{self.RESET}"
            message = f"{message}\n{exc_text}"

        return message


class RotatingJSONFileHandler(RotatingFileHandler):
    """Rotating file handler with JSON formatting.

    Automatically rotates logs when they reach a certain size
    and outputs structured JSON for log aggregation.
    """

    def __init__(
        self,
        filename: str,
        max_bytes: int = 50 * 1024 * 1024,  # 50MB default
        backup_count: int = 10,
        encoding: str = "utf-8",
    ):
        """Initialize rotating JSON file handler.

        Args:
            filename: Path to log file.
            max_bytes: Max file size before rotation.
            backup_count: Number of backup files to keep.
            encoding: File encoding.
        """
        super().__init__(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )
        self.setFormatter(JSONFormatter())


class ErrorTracker:
    """Track and aggregate error occurrences.

    Useful for detecting error spikes and implementing
    circuit breakers or alerts.
    """

    def __init__(self, window_seconds: int = 300):
        """Initialize error tracker.

        Args:
            window_seconds: Time window for tracking errors.
        """
        self.window_seconds = window_seconds
        self._errors: list = []
        self._lock_available = True

    def record_error(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an error occurrence.

        Args:
            error_type: Type/category of error.
            message: Error message.
            context: Additional context.
        """
        now = datetime.now(timezone.utc)
        self._errors.append(
            {
                "timestamp": now,
                "type": error_type,
                "message": message,
                "context": context or {},
            }
        )
        self._cleanup_old_errors(now)

    def _cleanup_old_errors(self, now: datetime) -> None:
        """Remove errors outside time window."""
        from datetime import timedelta

        cutoff = now - timedelta(seconds=self.window_seconds)
        self._errors = [e for e in self._errors if e["timestamp"] > cutoff]

    def get_error_count(self, error_type: Optional[str] = None) -> int:
        """Get count of errors in current window.

        Args:
            error_type: Filter by error type. All if None.

        Returns:
            Number of errors.
        """
        if error_type is None:
            return len(self._errors)
        return len([e for e in self._errors if e["type"] == error_type])

    def get_error_summary(self) -> Dict[str, int]:
        """Get error counts by type.

        Returns:
            Dict mapping error types to counts.
        """
        summary: Dict[str, int] = {}
        for error in self._errors:
            error_type = error["type"]
            summary[error_type] = summary.get(error_type, 0) + 1
        return summary
