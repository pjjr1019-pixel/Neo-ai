"""Tests for logging infrastructure."""

import json
import logging
import os
from unittest.mock import patch

from python_ai.logging.config import (
    LogConfig,
    LogFormat,
    LogLevel,
    setup_logging,
)
from python_ai.logging.context import (
    LogContext,
    RequestContext,
    clear_correlation_id,
    correlation_id_context,
    get_correlation_id,
    set_correlation_id,
)
from python_ai.logging.handlers import (
    ColoredFormatter,
    ErrorTracker,
    JSONFormatter,
)
from python_ai.logging.logger import LoggerAdapter, clear_loggers, get_logger


class TestLogConfig:
    """Tests for LogConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == LogLevel.INFO
        assert config.format == LogFormat.CONSOLE
        assert config.app_name == "neo-ai"
        assert config.enable_console is True
        assert config.enable_file_logging is False

    def test_env_override_level(self):
        """Test environment variable overrides level."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            config = LogConfig()
            assert config.level == LogLevel.DEBUG

    def test_env_override_format(self):
        """Test environment variable overrides format."""
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
            config = LogConfig()
            assert config.format == LogFormat.JSON


class TestSetupLogging:
    """Tests for setup_logging function."""

    def teardown_method(self):
        """Reset loggers after each test."""
        clear_loggers()
        logging.getLogger("neo-ai").handlers.clear()

    def test_setup_default_config(self):
        """Test setup with default configuration."""
        logger = setup_logging()
        assert logger.name == "neo-ai"
        assert len(logger.handlers) > 0

    def test_setup_json_format(self):
        """Test setup with JSON format."""
        config = LogConfig(format=LogFormat.JSON)
        logger = setup_logging(config)
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_setup_console_format(self):
        """Test setup with console format."""
        config = LogConfig(format=LogFormat.CONSOLE)
        logger = setup_logging(config)
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, ColoredFormatter)


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_format_basic_message(self):
        """Test formatting basic log message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_format_with_correlation_id(self):
        """Test formatting includes correlation ID."""
        formatter = JSONFormatter()
        set_correlation_id("test-123")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Message with correlation",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["correlation_id"] == "test-123"
        clear_correlation_id()

    def test_format_exception(self):
        """Test formatting exception info."""
        formatter = JSONFormatter(include_traceback=True)

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "Test error" in data["exception"]["message"]


class TestColoredFormatter:
    """Tests for ColoredFormatter."""

    def test_format_without_colors(self):
        """Test formatting without colors."""
        formatter = ColoredFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "INFO" in result
        assert "Test message" in result

    def test_format_includes_timestamp(self):
        """Test formatting includes timestamp."""
        formatter = ColoredFormatter(
            include_timestamp=True,
            use_colors=False,
        )
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "[" in result  # Timestamp brackets


class TestCorrelationId:
    """Tests for correlation ID management."""

    def teardown_method(self):
        """Clear correlation ID after each test."""
        clear_correlation_id()

    def test_set_and_get(self):
        """Test setting and getting correlation ID."""
        set_correlation_id("test-id")
        assert get_correlation_id() == "test-id"

    def test_auto_generate(self):
        """Test auto-generating correlation ID."""
        cid = set_correlation_id()
        assert cid is not None
        assert len(cid) > 0
        assert get_correlation_id() == cid

    def test_context_manager(self):
        """Test correlation ID context manager."""
        with correlation_id_context("ctx-123") as cid:
            assert cid == "ctx-123"
            assert get_correlation_id() == "ctx-123"

        # Should be cleared after context
        assert get_correlation_id() is None

    def test_nested_contexts(self):
        """Test nested correlation ID contexts."""
        with correlation_id_context("outer"):
            assert get_correlation_id() == "outer"
            with correlation_id_context("inner"):
                assert get_correlation_id() == "inner"
            # Restores outer
            assert get_correlation_id() == "outer"


class TestLogContext:
    """Tests for LogContext."""

    def teardown_method(self):
        """Clear state after each test."""
        clear_correlation_id()
        clear_loggers()

    def test_context_timing(self):
        """Test context tracks timing."""
        with LogContext(operation="test_op") as ctx:
            pass

        assert ctx.duration_ms is not None
        assert ctx.duration_ms >= 0

    def test_context_adds_results(self):
        """Test adding results to context."""
        with LogContext(operation="test_op") as ctx:
            ctx.add_result("score", 0.95)
            ctx.add_result("count", 100)

        assert ctx._results["score"] == 0.95
        assert ctx._results["count"] == 100

    def test_context_sets_correlation_id(self):
        """Test context sets correlation ID if not present."""
        with LogContext(operation="test_op"):
            cid = get_correlation_id()
            assert cid is not None

    def test_context_uses_provided_correlation_id(self):
        """Test context uses provided correlation ID."""
        with LogContext(
            operation="test_op",
            correlation_id="provided-id",
        ):
            assert get_correlation_id() == "provided-id"


class TestRequestContext:
    """Tests for RequestContext."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        ctx = RequestContext(
            method="GET",
            path="/api/test",
            correlation_id="req-123",
            user_id="user-456",
            client_ip="127.0.0.1",
        )

        data = ctx.to_dict()
        assert data["method"] == "GET"
        assert data["path"] == "/api/test"
        assert data["correlation_id"] == "req-123"
        assert data["user_id"] == "user-456"

    def test_log_response(self):
        """Test creating response log data."""
        ctx = RequestContext(
            method="POST",
            path="/api/submit",
        )

        response_data = ctx.log_response(
            status_code=200,
            response_size=1024,
        )

        assert response_data["status_code"] == 200
        assert response_data["response_size"] == 1024
        assert "duration_ms" in response_data


class TestLoggerAdapter:
    """Tests for LoggerAdapter."""

    def teardown_method(self):
        """Clear loggers after each test."""
        clear_loggers()

    def test_get_logger(self):
        """Test getting logger."""
        logger = get_logger("test_module")
        assert isinstance(logger, LoggerAdapter)

    def test_with_context(self):
        """Test creating logger with additional context."""
        logger = get_logger("test", extra={"key": "value"})
        new_logger = logger.with_context(another="field")

        assert new_logger.extra["key"] == "value"
        assert new_logger.extra["another"] == "field"


class TestErrorTracker:
    """Tests for ErrorTracker."""

    def test_record_error(self):
        """Test recording errors."""
        tracker = ErrorTracker(window_seconds=60)
        tracker.record_error("ValidationError", "Invalid input")

        assert tracker.get_error_count() == 1

    def test_error_count_by_type(self):
        """Test counting errors by type."""
        tracker = ErrorTracker()
        tracker.record_error("ValidationError", "Error 1")
        tracker.record_error("ValidationError", "Error 2")
        tracker.record_error("DatabaseError", "Error 3")

        assert tracker.get_error_count("ValidationError") == 2
        assert tracker.get_error_count("DatabaseError") == 1

    def test_error_summary(self):
        """Test getting error summary."""
        tracker = ErrorTracker()
        tracker.record_error("TypeA", "Msg 1")
        tracker.record_error("TypeA", "Msg 2")
        tracker.record_error("TypeB", "Msg 3")

        summary = tracker.get_error_summary()
        assert summary["TypeA"] == 2
        assert summary["TypeB"] == 1
