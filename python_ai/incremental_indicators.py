"""
Incremental Technical Indicator computation for NEO Hybrid AI.

Wraps TechnicalIndicators to compute indicators incrementally:
only new bars trigger re-computation, and previously computed
values are reused from a rolling cache.
"""

import logging
from typing import Dict, List, Optional

from python_ai.data_pipeline import TechnicalIndicators

logger = logging.getLogger(__name__)


class IncrementalIndicators:
    """Cache-backed incremental indicator engine.

    Instead of recomputing indicators over the full price history
    every tick, this class tracks the last processed index and
    only computes from there forward.

    Usage::

        inc = IncrementalIndicators()
        inc.update(close_prices, high_prices, low_prices)
        feat = inc.latest_features()
    """

    def __init__(self, lookback: int = 100) -> None:
        """Initialise with a lookback window.

        Args:
            lookback: Maximum number of bars retained.
        """
        self.lookback = lookback
        self._indicators = TechnicalIndicators()
        self._close: List[float] = []
        self._high: List[float] = []
        self._low: List[float] = []
        self._last_idx = 0

        # Cached latest values
        self._cache: Dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────

    def update(
        self,
        close: List[float],
        high: Optional[List[float]] = None,
        low: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Feed new bars and recompute indicators.

        Only bars beyond ``_last_idx`` trigger fresh computation.

        Args:
            close: Full or appended close prices.
            high: Full or appended high prices (defaults to close).
            low: Full or appended low prices (defaults to close).

        Returns:
            Dict of latest indicator values.
        """
        high = high or close
        low = low or close

        new_bars = len(close) - self._last_idx
        if new_bars <= 0 and self._cache:
            return dict(self._cache)

        # Trim to lookback to bound memory
        self._close = close[-self.lookback :]
        self._high = high[-self.lookback :]
        self._low = low[-self.lookback :]
        self._last_idx = len(close)

        self._recompute()
        return dict(self._cache)

    def append_bar(
        self,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> Dict[str, float]:
        """Append a single new bar.

        Args:
            close: New closing price.
            high: New high price (defaults to close).
            low: New low price (defaults to close).

        Returns:
            Dict of latest indicator values.
        """
        self._close.append(close)
        self._high.append(high if high is not None else close)
        self._low.append(low if low is not None else close)

        # Trim to lookback
        if len(self._close) > self.lookback:
            self._close = self._close[-self.lookback :]
            self._high = self._high[-self.lookback :]
            self._low = self._low[-self.lookback :]

        self._last_idx += 1
        self._recompute()
        return dict(self._cache)

    def latest_features(self) -> Dict[str, float]:
        """Return the most recently computed feature dict (f0-f9).

        Returns:
            Dict with keys f0 … f9 matching DataPipeline format.
        """
        if not self._cache:
            return {f"f{i}": 0.0 for i in range(10)}
        return dict(self._cache)

    # ── Internal ──────────────────────────────────────────────

    def _recompute(self) -> None:
        """Recompute indicators from the rolling window."""
        c = self._close
        h = self._high
        lo = self._low

        if not c:
            return

        rsi = self._indicators.calculate_rsi(c)[-1]
        macd_vals, signal_vals = self._indicators.calculate_macd(c)
        macd_val = macd_vals[-1] if macd_vals else 0.0
        signal_val = signal_vals[-1] if signal_vals else 0.0

        upper, middle, lower = self._indicators.calculate_bollinger_bands(c)
        upper_val = upper[-1] if upper else 0.0
        lower_val = lower[-1] if lower else 0.0

        atr = self._indicators.calculate_atr(h, lo, c)[-1]
        sma = self._indicators.calculate_sma(c)[-1]

        self._cache = {
            "f0": float(rsi) / 100.0,
            "f1": float(macd_val),
            "f2": float(signal_val),
            "f3": float(upper_val - c[-1]),
            "f4": float(c[-1] - lower_val),
            "f5": float(atr),
            "f6": float(c[-1] - sma),
            "f7": (float((c[-1] - c[-2]) / c[-2]) if len(c) > 1 else 0.0),
            "f8": (float((c[-1] - c[-5]) / c[-5]) if len(c) > 5 else 0.0),
            "f9": (float((c[-1] - c[-10]) / c[-10]) if len(c) > 10 else 0.0),
            "rsi_raw": float(rsi),
            "macd_raw": float(macd_val),
            "atr_raw": float(atr),
            "sma_raw": float(sma),
        }
