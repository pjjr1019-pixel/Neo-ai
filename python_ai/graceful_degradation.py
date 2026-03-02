"""
Graceful Degradation Framework for NEO Hybrid AI.

Provides fallback strategies when external services are
unavailable.  Works alongside the circuit breaker to
deliver best-effort responses instead of hard failures.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FallbackResult:
    """Container for a degraded response.

    Attributes:
        value: The fallback value returned.
        degraded: ``True`` when a fallback was used.
        source: Label of the source that produced the
            value (e.g. ``"cache"``, ``"default"``).
        error: The original exception, if any.
    """

    def __init__(
        self,
        value: Any,
        degraded: bool = False,
        source: str = "primary",
        error: Optional[Exception] = None,
    ) -> None:
        """Initialise a FallbackResult."""
        self.value = value
        self.degraded = degraded
        self.source = source
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "value": self.value,
            "degraded": self.degraded,
            "source": self.source,
            "error": str(self.error) if self.error else None,
        }


class GracefulDegradation:
    """Execute a primary callable with ordered fallbacks.

    Attempts the primary function first.  On failure, tries
    each fallback in registration order.  If all fallbacks
    fail, returns the static default.

    Args:
        name: Human-readable label for logging.
        default: Value returned when everything fails.
    """

    def __init__(
        self,
        name: str = "service",
        default: Any = None,
    ) -> None:
        """Initialise with a service name and default."""
        self.name = name
        self._default = default
        self._fallbacks: list[tuple[str, Callable[..., Any]]] = []
        self._stats: Dict[str, int] = {
            "primary_ok": 0,
            "fallback_used": 0,
            "all_failed": 0,
        }

    def add_fallback(
        self,
        label: str,
        func: Callable[..., Any],
    ) -> None:
        """Register a named fallback callable.

        Fallbacks are tried in registration order.

        Args:
            label: Descriptive label (e.g. ``"cache"``).
            func: Callable that accepts the same args as
                the primary function.
        """
        self._fallbacks.append((label, func))

    def call(
        self,
        primary: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> FallbackResult:
        """Try *primary*, then fallbacks, then default.

        Args:
            primary: The main function to call.
            *args: Positional arguments forwarded.
            **kwargs: Keyword arguments forwarded.

        Returns:
            ``FallbackResult`` with the best available value.
        """
        try:
            result = primary(*args, **kwargs)
            self._stats["primary_ok"] += 1
            return FallbackResult(
                value=result,
                degraded=False,
                source="primary",
            )
        except Exception as primary_exc:
            logger.warning(
                "%s primary failed: %s",
                self.name,
                primary_exc,
            )

        for label, fallback_fn in self._fallbacks:
            try:
                result = fallback_fn(*args, **kwargs)
                self._stats["fallback_used"] += 1
                logger.info(
                    "%s degraded to fallback '%s'",
                    self.name,
                    label,
                )
                return FallbackResult(
                    value=result,
                    degraded=True,
                    source=label,
                )
            except Exception as fb_exc:
                logger.warning(
                    "%s fallback '%s' failed: %s",
                    self.name,
                    label,
                    fb_exc,
                )

        self._stats["all_failed"] += 1
        logger.error(
            "%s all fallbacks exhausted — using default",
            self.name,
        )
        return FallbackResult(
            value=self._default,
            degraded=True,
            source="default",
        )

    def stats(self) -> Dict[str, int]:
        """Return call statistics."""
        return dict(self._stats)


class CacheBackedFallback:
    """Caches the last successful result for use as fallback.

    Wrap around expensive calls whose last-known-good value
    is acceptable during transient outages.

    Args:
        ttl: Maximum age (seconds) of cached values.
    """

    def __init__(self, ttl: float = 300.0) -> None:
        """Initialise with a cache TTL."""
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if still fresh."""
        ts = self._timestamps.get(key)
        if ts is None:
            return None
        if (time.time() - ts) > self._ttl:
            return None
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def call_with_cache(
        self,
        key: str,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> FallbackResult:
        """Call *func*; on success cache the result.

        On failure return the cached value if available.

        Args:
            key: Cache lookup key.
            func: Primary callable.
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            ``FallbackResult`` with fresh or cached value.
        """
        try:
            result = func(*args, **kwargs)
            self.put(key, result)
            return FallbackResult(
                value=result,
                degraded=False,
                source="primary",
            )
        except Exception as exc:
            cached = self.get(key)
            if cached is not None:
                logger.info("Using cached value for '%s'", key)
                return FallbackResult(
                    value=cached,
                    degraded=True,
                    source="cache",
                    error=exc,
                )
            logger.error("No cached value for '%s' — raising", key)
            raise
