"""Tests for resilience wiring module.

Covers:
- db_resilient_call: success path, retry on transient error
- external_api_call: success path, retry on transient error
- Circuit breaker integration
"""

import pytest

from python_ai.resilience import (
    api_circuit_breaker,
    db_circuit_breaker,
    db_resilient_call,
    external_api_call,
)


class TestDbResilientCall:
    """Tests for db_resilient_call."""

    def setup_method(self):
        """Reset circuit breaker state between tests."""
        db_circuit_breaker.reset()

    def test_success_path(self):
        """Successful function call is returned as-is."""
        result = db_resilient_call(lambda: 42)
        assert result == 42

    def test_passes_args(self):
        """Positional and keyword arguments are forwarded."""

        def add(a, b, offset=0):
            """Add two numbers with offset."""
            return a + b + offset

        result = db_resilient_call(add, 1, 2, offset=10)
        assert result == 13

    def test_retries_on_connection_error(self):
        """Transient ConnectionError triggers retry."""
        calls = {"count": 0}

        def flaky():
            """Fail once then succeed."""
            calls["count"] += 1
            if calls["count"] < 2:
                raise ConnectionError("transient")
            return "ok"

        result = db_resilient_call(flaky, max_attempts=3)
        assert result == "ok"
        assert calls["count"] == 2

    def test_raises_after_max_attempts(self):
        """Persistent failure exhausts retries."""

        def always_fail():
            """Always fail."""
            raise ConnectionError("persistent")

        with pytest.raises(ConnectionError, match="persistent"):
            db_resilient_call(
                always_fail, max_attempts=2, timeout=1.0
            )

    def test_non_retryable_error_not_retried(self):
        """Non-retryable error propagates immediately."""
        calls = {"count": 0}

        def bad():
            """Raise a ValueError."""
            calls["count"] += 1
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            db_resilient_call(bad, max_attempts=3)
        assert calls["count"] == 1


class TestExternalApiCall:
    """Tests for external_api_call."""

    def setup_method(self):
        """Reset circuit breaker state between tests."""
        api_circuit_breaker.reset()

    def test_success_path(self):
        """Successful function call is returned as-is."""
        result = external_api_call(lambda: "data")
        assert result == "data"

    def test_passes_args(self):
        """Arguments forwarded correctly."""

        def greet(name, prefix="Hello"):
            """Greet a person."""
            return f"{prefix} {name}"

        result = external_api_call(greet, "Neo", prefix="Hi")
        assert result == "Hi Neo"

    def test_retries_on_timeout_error(self):
        """Transient TimeoutError triggers retry."""
        calls = {"count": 0}

        def flaky():
            """Fail once then succeed."""
            calls["count"] += 1
            if calls["count"] < 2:
                raise TimeoutError("slow")
            return "fast"

        result = external_api_call(flaky, max_attempts=3)
        assert result == "fast"
        assert calls["count"] == 2

    def test_non_retryable_not_retried(self):
        """Non-retryable errors propagate immediately."""
        calls = {"count": 0}

        def bad():
            """Raise a RuntimeError."""
            calls["count"] += 1
            raise RuntimeError("bug")

        with pytest.raises(RuntimeError, match="bug"):
            external_api_call(bad, max_attempts=3)
        assert calls["count"] == 1


class TestCircuitBreakerInstances:
    """Tests for pre-configured circuit breaker instances."""

    def test_db_circuit_breaker_config(self):
        """DB breaker has expected thresholds."""
        assert db_circuit_breaker.failure_threshold == 5
        assert db_circuit_breaker.recovery_timeout == 30.0

    def test_api_circuit_breaker_config(self):
        """API breaker has expected thresholds."""
        assert api_circuit_breaker.failure_threshold == 3
        assert api_circuit_breaker.recovery_timeout == 60.0

    def test_circuit_breakers_are_distinct(self):
        """DB and API breakers are separate instances."""
        assert db_circuit_breaker is not api_circuit_breaker
