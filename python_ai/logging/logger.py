"""Logger utilities and adapters."""

import logging
from typing import Any, Dict, Optional


class LoggerAdapter(logging.LoggerAdapter):
    """Enhanced logger adapter with extra context support.

    Allows adding structured context to log messages
    without modifying the log call signature.
    """

    def __init__(
        self,
        logger: logging.Logger,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Initialize logger adapter.

        Args:
            logger: Base logger instance.
            extra: Extra context to include in all logs.
        """
        super().__init__(logger, extra or {})

    def process(
        self,
        msg: str,
        kwargs: Dict[str, Any],
    ) -> tuple:
        """Process log message with extra context.

        Args:
            msg: Log message.
            kwargs: Keyword arguments.

        Returns:
            Processed message and kwargs tuple.
        """
        # Merge adapter extra with call-time extra
        combined_extra = dict(self.extra)
        if "extra" in kwargs:
            combined_extra.update(kwargs["extra"])
        kwargs["extra"] = {"extra": combined_extra}
        return msg, kwargs

    def with_context(self, **context: Any) -> "LoggerAdapter":
        """Create new adapter with additional context.

        Args:
            **context: Additional context fields.

        Returns:
            New LoggerAdapter with merged context.
        """
        merged = dict(self.extra)
        merged.update(context)
        return LoggerAdapter(self.logger, merged)

    def debug(
        self,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, msg, *args, exc_info=exc_info, **kwargs)

    def info(
        self,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log info message."""
        self.log(logging.INFO, msg, *args, exc_info=exc_info, **kwargs)

    def warning(
        self,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log warning message."""
        self.log(logging.WARNING, msg, *args, exc_info=exc_info, **kwargs)

    def error(
        self,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        self.log(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(
        self,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log critical message."""
        self.log(logging.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)

    def exception(
        self,
        msg: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log exception with traceback."""
        kwargs["exc_info"] = True
        self.error(msg, *args, **kwargs)


_loggers: Dict[str, LoggerAdapter] = {}


def get_logger(
    name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> LoggerAdapter:
    """Get or create a logger adapter.

    Args:
        name: Logger name (usually module name).
        extra: Extra context to include in logs.

    Returns:
        LoggerAdapter instance.
    """
    from python_ai.logging.config import LogConfig

    key = f"{name}:{hash(frozenset((extra or {}).items()))}"

    if key not in _loggers:
        # Get app logger or create basic one
        config = LogConfig()
        base_logger = logging.getLogger(config.app_name)

        # Create child logger for this module
        child_logger = base_logger.getChild(name)

        _loggers[key] = LoggerAdapter(child_logger, extra or {})

    return _loggers[key]


def clear_loggers() -> None:
    """Clear cached loggers. Useful for testing."""
    _loggers.clear()
