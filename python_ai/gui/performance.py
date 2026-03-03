"""Performance helpers for GUI and backend refresh loops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class PerfSample:
    """Latency sample for one operation."""

    operation: str
    duration_ms: float


def profile_operation(name: str, func: Callable[[], object]) -> PerfSample:
    """Time operation execution and return a sample."""
    start = time.perf_counter()
    func()
    elapsed = (time.perf_counter() - start) * 1000.0
    return PerfSample(operation=name, duration_ms=elapsed)


def regression_check(
    baseline_ms: float,
    measured_ms: float,
    *,
    tolerance_ratio: float = 1.20,
) -> Dict[str, object]:
    """Check if measured latency regressed beyond tolerance."""
    limit = baseline_ms * tolerance_ratio
    return {
        "regressed": measured_ms > limit,
        "limit_ms": limit,
        "measured_ms": measured_ms,
    }
