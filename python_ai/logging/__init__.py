"""Logging infrastructure for NEO AI Platform.

This module provides centralized, structured logging with support for:
- JSON formatting for production
- Colored console output for development
- Log rotation and archival
- Exception tracking with context
- Request tracing with correlation IDs
"""

from python_ai.logging.config import LogConfig, setup_logging
from python_ai.logging.logger import get_logger, LoggerAdapter
from python_ai.logging.handlers import (
    JSONFormatter,
    ColoredFormatter,
    RotatingJSONFileHandler,
)
from python_ai.logging.context import (
    LogContext,
    correlation_id_context,
    get_correlation_id,
    set_correlation_id,
)

__all__ = [
    "LogConfig",
    "setup_logging",
    "get_logger",
    "LoggerAdapter",
    "JSONFormatter",
    "ColoredFormatter",
    "RotatingJSONFileHandler",
    "LogContext",
    "correlation_id_context",
    "get_correlation_id",
    "set_correlation_id",
]
