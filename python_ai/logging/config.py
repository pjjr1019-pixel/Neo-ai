"""Logging configuration module."""

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""

    JSON = "json"
    CONSOLE = "console"
    PLAIN = "plain"


@dataclass
class LogConfig:
    """Logging configuration.

    Attributes:
        level: Minimum log level to capture.
        format: Output format (json, console, plain).
        log_dir: Directory for log files.
        app_name: Application name for log identification.
        enable_file_logging: Whether to write logs to file.
        max_file_size_mb: Max size of each log file in MB.
        backup_count: Number of backup files to keep.
        enable_console: Whether to output to console.
        include_traceback: Include full traceback on errors.
        correlation_id_header: HTTP header name for correlation ID.
    """

    level: LogLevel = field(
        default_factory=lambda: LogLevel(
            os.getenv("LOG_LEVEL", "INFO").upper()
        )
    )
    format: LogFormat = field(
        default_factory=lambda: LogFormat(
            os.getenv("LOG_FORMAT", "console").lower()
        )
    )
    log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("LOG_DIR", "logs"))
    )
    app_name: str = field(
        default_factory=lambda: os.getenv("APP_NAME", "neo-ai")
    )
    enable_file_logging: bool = field(
        default_factory=lambda: os.getenv("LOG_TO_FILE", "false").lower()
        == "true"
    )
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("LOG_MAX_FILE_SIZE_MB", "50"))
    )
    backup_count: int = field(
        default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "10"))
    )
    enable_console: bool = True
    include_traceback: bool = True
    correlation_id_header: str = "X-Correlation-ID"


def setup_logging(config: Optional[LogConfig] = None) -> logging.Logger:
    """Set up application logging.

    Args:
        config: Logging configuration. Uses defaults if None.

    Returns:
        Root logger instance.
    """
    from python_ai.logging.handlers import (
        JSONFormatter,
        ColoredFormatter,
        RotatingJSONFileHandler,
    )

    if config is None:
        config = LogConfig()

    # Get or create root logger
    root_logger = logging.getLogger(config.app_name)
    root_logger.setLevel(getattr(logging, config.level.value))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.value))

        if config.format == LogFormat.JSON:
            console_handler.setFormatter(JSONFormatter())
        elif config.format == LogFormat.CONSOLE:
            console_handler.setFormatter(
                ColoredFormatter(include_timestamp=True)
            )
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        root_logger.addHandler(console_handler)

    # File handler
    if config.enable_file_logging:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = config.log_dir / f"{config.app_name}.log"

        file_handler = RotatingJSONFileHandler(
            filename=str(log_file),
            max_bytes=config.max_file_size_mb * 1024 * 1024,
            backup_count=config.backup_count,
        )
        file_handler.setLevel(getattr(logging, config.level.value))
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    return root_logger
