"""Rolling feature factory with vectorization and optional JIT."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Tuple

import numpy as np

from python_ai import vectorized_indicators as vi

try:  # pragma: no cover - optional dependency branch
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency branch
    _NUMBA_AVAILABLE = False


def _to_tuple(values: Iterable[float]) -> Tuple[float, ...]:
    """Convert an iterable of values to a hashable float tuple."""
    return tuple(float(v) for v in values)


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _sma_last_jit(values: np.ndarray, window: int) -> float:
        """Return last SMA value via numba-accelerated kernel."""
        if len(values) < window:
            return np.nan
        return float(values[-window:].mean())

else:

    def _sma_last_jit(values: np.ndarray, window: int) -> float:
        """Return last SMA value using numpy fallback implementation."""
        if len(values) < window:
            return np.nan
        return float(values[-window:].mean())


@lru_cache(maxsize=4096)
def sma_last_cached(prices: Tuple[float, ...], window: int) -> float:
    """Memoized SMA(last) for repeated per-symbol calls."""
    arr = np.asarray(prices, dtype=np.float64)
    return float(_sma_last_jit(arr, int(window)))


@lru_cache(maxsize=4096)
def ema_last_cached(prices: Tuple[float, ...], span: int) -> float:
    """Memoized EMA(last) using vectorized indicator implementation."""
    values = vi.ema(list(prices), span=span)
    return float(values[-1]) if len(values) else float("nan")


@lru_cache(maxsize=4096)
def rsi_last_cached(prices: Tuple[float, ...], window: int) -> float:
    """Memoized RSI(last) using vectorized indicator implementation."""
    values = vi.rsi(list(prices), window=window)
    return float(values[-1]) if len(values) else float("nan")


def compute_rolling_features(
    prices: Iterable[float],
    *,
    sma_window: int = 14,
    ema_span: int = 14,
    rsi_window: int = 14,
) -> Dict[str, float]:
    """Compute core rolling features from a price series."""
    series = _to_tuple(prices)
    if not series:
        return {"sma": 0.0, "ema": 0.0, "rsi": 0.0, "returns_1": 0.0}

    sma_last = sma_last_cached(series, sma_window)
    ema_last = ema_last_cached(series, ema_span)
    rsi_last = rsi_last_cached(series, rsi_window)
    prev = series[-2] if len(series) > 1 else series[-1]
    ret_1 = (series[-1] - prev) / (prev + 1e-12)

    return {
        "sma": float(sma_last if not np.isnan(sma_last) else 0.0),
        "ema": float(ema_last if not np.isnan(ema_last) else 0.0),
        "rsi": float(rsi_last if not np.isnan(rsi_last) else 0.0),
        "returns_1": float(ret_1),
    }


def clear_feature_cache() -> None:
    """Clear memoization caches for test isolation and memory control."""
    sma_last_cached.cache_clear()
    ema_last_cached.cache_clear()
    rsi_last_cached.cache_clear()
