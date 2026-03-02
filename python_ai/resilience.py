"""
Resilience wiring for NEO Hybrid AI production services.

Pre-configures :class:`CircuitBreaker`, :func:`retry`, and
:class:`GracefulDegradation` instances for database and external-API
operations, and exposes them as ready-to-use helpers.

Usage::

    from python_ai.resilience import db_resilient_call

    result = db_resilient_call(repo.get_recent, session, limit=10)

"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from python_ai.circuit_breaker import CircuitBreaker
from python_ai.retry import retry_call
from python_ai.timeout import timeout_call

logger = logging.getLogger(__name__)

__all__ = [
    "api_circuit_breaker",
    "db_circuit_breaker",
    "db_resilient_call",
    "external_api_call",
]

T = TypeVar("T")

# ── Database circuit breaker ──────────────────────────────────
# Opens after 5 consecutive failures, recovers after 30 s.

db_circuit_breaker = CircuitBreaker(
    name="database",
    failure_threshold=5,
    recovery_timeout=30.0,
)

# ── External API circuit breaker ──────────────────────────────
# Opens after 3 consecutive failures, recovers after 60 s.

api_circuit_breaker = CircuitBreaker(
    name="external_api",
    failure_threshold=3,
    recovery_timeout=60.0,
)


# ── Retryable exception list for DB operations ───────────────

_DB_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,
)


# ── Resilient DB call ─────────────────────────────────────────


def db_resilient_call(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    timeout: float = 10.0,
    **kwargs: Any,
) -> T:
    """Execute *func* with retry, circuit breaker, and timeout.

    Wraps the function call with:
    1. **Timeout** — abort if it takes longer than *timeout* seconds.
    2. **Retry** — up to *max_attempts* with exponential back-off.
    3. **Circuit breaker** — skip calls entirely when the DB is
       unavailable.

    Args:
        func: Callable to execute (e.g., repository method).
        *args: Positional arguments forwarded to *func*.
        max_attempts: Maximum retry attempts.
        timeout: Per-call timeout in seconds.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        The return value of *func*.

    Raises:
        CircuitOpenError: When the DB circuit breaker is open.
        Exception: Any non-retryable exception from *func*.
    """

    def _guarded() -> T:
        """Invoke *func* inside the DB circuit breaker."""
        return db_circuit_breaker.call(
            lambda: timeout_call(func, args, kwargs, timeout=timeout)
        )

    return retry_call(
        _guarded,
        max_attempts=max_attempts,
        base_delay=0.5,
        max_delay=5.0,
        retryable_exceptions=_DB_RETRYABLE,
    )


# ── Resilient external API call ───────────────────────────────


def external_api_call(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    timeout: float = 15.0,
    **kwargs: Any,
) -> T:
    """Execute *func* with retry, circuit breaker, and timeout.

    Same semantics as :func:`db_resilient_call` but uses the
    ``api_circuit_breaker`` and longer timeouts suited for
    HTTP/REST calls.

    Args:
        func: Callable to execute.
        *args: Positional arguments.
        max_attempts: Retry attempts.
        timeout: Per-call timeout.
        **kwargs: Keyword arguments.

    Returns:
        The return value of *func*.
    """

    def _guarded() -> T:
        """Invoke *func* inside the API circuit breaker."""
        return api_circuit_breaker.call(
            lambda: timeout_call(func, args, kwargs, timeout=timeout)
        )

    return retry_call(
        _guarded,
        max_attempts=max_attempts,
        base_delay=1.0,
        max_delay=15.0,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )
