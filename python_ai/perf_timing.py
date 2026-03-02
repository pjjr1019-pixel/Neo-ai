"""
Performance Timing utilities for NEO Hybrid AI.

Provides a ``@timed`` decorator and a global timing registry
so critical-path latencies can be monitored and exposed via
/metrics or /health.
"""

import functools
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimingRegistry:
    """Collects and exposes function execution timings.

    Every ``@timed`` function records its latency here.

    Attributes:
        records: Mapping of function name → list of durations.
    """

    def __init__(self, max_per_key: int = 200) -> None:
        """Initialise the registry.

        Args:
            max_per_key: Max records kept per function name.
        """
        self.max_per_key = max_per_key
        self._data: Dict[str, List[float]] = defaultdict(list)

    def record(self, name: str, duration: float) -> None:
        """Record a timing sample.

        Args:
            name: Function or operation name.
            duration: Execution time in seconds.
        """
        bucket = self._data[name]
        bucket.append(duration)
        if len(bucket) > self.max_per_key:
            self._data[name] = bucket[-self.max_per_key :]

    def stats(self, name: str) -> Dict[str, float]:
        """Compute summary statistics for *name*.

        Args:
            name: Function name.

        Returns:
            Dict with count, mean, min, max, p95.
        """
        samples = self._data.get(name, [])
        if not samples:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
            }
        sorted_s = sorted(samples)
        idx_95 = max(0, int(len(sorted_s) * 0.95) - 1)
        return {
            "count": len(samples),
            "mean": sum(samples) / len(samples),
            "min": sorted_s[0],
            "max": sorted_s[-1],
            "p95": sorted_s[idx_95],
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        """Summary for every tracked function."""
        return {name: self.stats(name) for name in self._data}

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._data.clear()


# ── Global singleton ──────────────────────────────────────────

_registry = TimingRegistry()


def get_timing_registry() -> TimingRegistry:
    """Return the global TimingRegistry."""
    return _registry


# ── Decorator ─────────────────────────────────────────────────


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that measures and logs execution time.

    Automatically registers the result in the global
    ``TimingRegistry``.

    Usage::

        @timed
        def predict(features):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        """Measure and record execution time of the wrapped function."""
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            _registry.record(func.__qualname__, elapsed)
            if elapsed > 1.0:
                logger.warning(
                    "%s took %.3fs (slow!)",
                    func.__qualname__,
                    elapsed,
                )

    return wrapper  # type: ignore[return-value]
