"""Model selection utilities: purged CV, regime scoring, significance tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FoldResult:
    """Result for one purged walk-forward fold."""

    train_size: int
    test_size: int
    score: float


def purged_walk_forward_cv(
    returns: Sequence[float],
    *,
    n_splits: int = 5,
    purge_window: int = 1,
) -> List[FoldResult]:
    """Perform purged walk-forward CV over a returns series."""
    values = np.asarray(returns, dtype=np.float64)
    if len(values) < 3:
        return []

    n_splits = max(2, min(n_splits, len(values)))
    fold_size = max(1, len(values) // n_splits)
    out: List[FoldResult] = []

    for idx in range(n_splits):
        test_start = idx * fold_size
        if idx == n_splits - 1:
            test_end = len(values)
        else:
            test_end = (idx + 1) * fold_size
        train_end = max(0, test_start - purge_window)
        train = values[:train_end]
        test = values[test_start:test_end]
        if len(test) == 0:
            continue
        score = float(np.mean(test))
        out.append(FoldResult(len(train), len(test), score))
    return out


def detect_market_regime(returns: Sequence[float]) -> str:
    """Classify returns into bullish, bearish, or sideways regime."""
    values = np.asarray(returns, dtype=np.float64)
    mean_ret = float(np.mean(values)) if len(values) else 0.0
    if mean_ret > 0.001:
        return "bullish"
    if mean_ret < -0.001:
        return "bearish"
    return "sideways"


def regime_aware_score(returns_by_regime: Dict[str, Iterable[float]]) -> float:
    """Compute average regime score weighted equally by regime bucket."""
    scores = []
    for series in returns_by_regime.values():
        arr = np.asarray(list(series), dtype=np.float64)
        if len(arr):
            scores.append(float(np.mean(arr)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def paired_significance_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> Tuple[float, float]:
    """Run paired t-test returning (t_statistic, p_value)."""
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    candidate_arr = np.asarray(candidate, dtype=np.float64)
    n = min(len(baseline_arr), len(candidate_arr))
    if n < 2:
        return 0.0, 1.0
    diffs = candidate_arr[:n] - baseline_arr[:n]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff == 0.0:
        return 0.0, 1.0
    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Normal approximation for p-value to keep dependency-light behavior.
    z = abs(t_stat)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_val = max(0.0, min(1.0, 2.0 * (1.0 - cdf)))
    return float(t_stat), float(p_val)
