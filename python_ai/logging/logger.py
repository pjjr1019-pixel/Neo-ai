"""Logger utilities and adapters."""

import logging
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


class LoggerAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
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
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> Tuple[object, MutableMapping[str, Any]]:
        """Process log message with extra context.

        Args:
            msg: Log message.
            kwargs: Keyword arguments.

        Returns:
            Processed message and kwargs tuple.
        """
        # Merge adapter extra with call-time extra
        extra_dict: Mapping[str, object] = self.extra or {}
        combined_extra: Dict[str, Any] = dict(extra_dict)
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
        extra_dict: Mapping[str, object] = self.extra or {}
        merged: Dict[str, Any] = dict(extra_dict)
        merged.update(context)
        return LoggerAdapter(self.logger, merged)


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
