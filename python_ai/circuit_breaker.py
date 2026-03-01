"""
Circuit Breaker pattern for external API calls.

Prevents cascading failures by failing fast when a downstream
service is unresponsive.  After ``failure_threshold`` consecutive
failures the breaker *opens* and immediately raises
``CircuitOpenError`` for ``recovery_timeout`` seconds.  A single
probe request is then allowed through (*half-open*); if it
succeeds the breaker resets.
"""

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Possible states of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""


class CircuitBreaker:
    """Thread-safe Circuit Breaker.

    Usage::

        cb = CircuitBreaker(name="exchange")

        @cb
        def fetch_price(symbol: str) -> float:
            return exchange.fetch_ticker(symbol)["last"]

        # Or wrap inline:
        result = cb.call(exchange.fetch_ticker, "BTC/USDT")

    Attributes:
        name: Human-readable label (used in logs).
        failure_threshold: Failures before opening.
        recovery_timeout: Seconds before half-open probe.
        state: Current circuit state.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        """Initialise the circuit breaker.

        Args:
            name: Label for logging.
            failure_threshold: Consecutive failures before opening.
            recovery_timeout: Seconds before allowing a probe.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current state, accounting for recovery timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        "CircuitBreaker[%s] -> HALF_OPEN",
                        self.name,
                    )
            return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *func* through the circuit breaker.

        Args:
            func: Callable to invoke.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            Return value of *func*.

        Raises:
            CircuitOpenError: If the circuit is open.
        """
        current = self.state
        if current == CircuitState.OPEN:
            raise CircuitOpenError(
                f"CircuitBreaker[{self.name}] is OPEN — "
                f"retry after {self.recovery_timeout}s"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result  # type: ignore[return-value]
        except Exception as exc:
            self._on_failure(exc)
            raise

    def reset(self) -> None:
        """Manually reset to CLOSED."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info("CircuitBreaker[%s] manually RESET", self.name)

    # ── Decorator support ─────────────────────────────────────

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator: ``@breaker``."""
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Invoke the wrapped function through the circuit breaker."""
            return self.call(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    # ── Internal state transitions ────────────────────────────

    def _on_success(self) -> None:
        """Record a successful call and reset failure count."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    "CircuitBreaker[%s] -> CLOSED (probe OK)",
                    self.name,
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def _on_failure(self, exc: Exception) -> None:
        """Record a failed call and open circuit if threshold reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s] -> OPEN after %d failures "
                    "(last: %s)",
                    self.name,
                    self._failure_count,
                    exc,
                )
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s] -> OPEN (half-open probe "
                    "failed: %s)",
                    self.name,
                    exc,
                )

    # ── Inspection helpers ────────────────────────────────────

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    def to_dict(self) -> dict:
        """Serialise state for /health or /metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }
