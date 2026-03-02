"""
Data Pipeline for NEO Hybrid AI.

Computes technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
from OHLCV data and prepares feature vectors for model inference.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Canonical feature names ───────────────────────────────────
# Order matters: the ML model sees values in this order.
FEATURE_NAMES: Tuple[str, ...] = (
    "rsi_14",
    "macd_value",
    "macd_signal",
    "bb_upper_dist",
    "bb_lower_dist",
    "atr_14",
    "price_vs_sma",
    "return_1d",
    "return_5d",
    "return_10d",
)

__all__ = [
    "DataPipeline",
    "FEATURE_NAMES",
    "TechnicalIndicators",
    "get_pipeline",
]


class TechnicalIndicators:
    """Compute technical indicators from price data."""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index using Wilder's smoothing.

        Uses the exponential moving average method (Wilder's
        smoothing factor ``1/period``) which is the industry-standard
        RSI calculation used by TradingView, StockCharts, etc.

        Args:
            prices: List of closing prices.
            period: RSI period (default 14).

        Returns:
            List of RSI values (0-100).
        """
        if len(prices) < period + 1:
            return [50.0] * len(prices)

        prices_array = np.array(prices, dtype=float)
        deltas = np.diff(prices_array)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Seed with simple average for first ``period`` bars
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        rsi_values: List[float] = [50.0] * period

        # First real RSI value
        if avg_loss == 0:
            rsi_values.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

        # Wilder's exponential smoothing for remaining bars
        alpha = 1.0 / period
        for i in range(period, len(deltas)):
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha
            if avg_loss == 0:
                rsi_values.append(100.0 if avg_gain > 0 else 50.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

        return rsi_values

    @staticmethod
    def calculate_macd(
        prices: List[float], fast: int = 12, slow: int = 26
    ) -> Tuple[List[float], List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence).

        Uses vectorised pandas EWM for the fast / slow EMAs and the
        9-period signal line, avoiding Python-level loops.

        Args:
            prices: List of closing prices.
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).

        Returns:
            Tuple of (MACD values, Signal line values).
        """
        import pandas as pd

        s = pd.Series(prices, dtype=float)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow

        signal = macd.ewm(span=9, adjust=False).mean()

        return macd.tolist(), signal.tolist()

    @staticmethod
    def calculate_bollinger_bands(
        prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands.

        Uses ``pandas.Series.rolling`` for fully vectorised
        mean and standard-deviation computation.

        Args:
            prices: List of closing prices.
            period: SMA period (default 20).
            std_dev: Standard deviation multiplier (default 2).

        Returns:
            Tuple of (Upper band, Middle band, Lower band).
        """
        import pandas as pd

        s = pd.Series(prices, dtype=float)
        middle = s.rolling(window=period, min_periods=1).mean()
        std_arr = s.rolling(window=period, min_periods=1).std(ddof=0)

        upper = middle + std_arr * std_dev
        lower = middle - std_arr * std_dev

        return upper.tolist(), middle.tolist(), lower.tolist()

    @staticmethod
    def calculate_atr(
        high: List[float],
        low: List[float],
        close: List[float],
        period: int = 14,
    ) -> List[float]:
        """Calculate Average True Range (ATR).

        Args:
            high: List of high prices.
            low: List of low prices.
            close: List of close prices.
            period: ATR period (default 14).

        Returns:
            List of ATR values.
        """
        if len(high) < 2:
            return [0.0] * len(high)

        h = np.array(high, dtype=float)
        l_prices = np.array(low, dtype=float)
        c = np.array(close, dtype=float)

        tr1 = h - l_prices
        tr2 = np.abs(h - np.roll(c, 1))
        tr3 = np.abs(l_prices - np.roll(c, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.convolve(tr, [1 / period] * period, mode="same")

        return list(atr)

    @staticmethod
    def calculate_sma(prices: List[float], period: int = 20) -> List[float]:
        """Calculate Simple Moving Average (SMA).

        Uses ``pandas.Series.rolling`` for vectorised computation.

        Args:
            prices: List of closing prices.
            period: SMA period (default 20).

        Returns:
            List of SMA values.
        """
        import pandas as pd

        s = pd.Series(prices, dtype=float)
        sma = s.rolling(window=period, min_periods=1).mean()
        return list(sma)


class DataPipeline:
    """Data pipeline for preprocessing and feature engineering."""

    def __init__(self) -> None:
        """Initialize data pipeline."""
        self.indicators = TechnicalIndicators()
        self.price_history: Dict[str, Dict[str, List[float]]] = {}
        self.feature_cache: Dict[str, Any] = {}

    def update_price_data(
        self,
        symbol: str,
        ohlcv: Dict[str, List[float]],
    ) -> None:
        """Update OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD').
            ohlcv: Dict with 'open', 'high', 'low', 'close', 'volume' lists.
        """
        self.price_history[symbol] = ohlcv.copy()
        # Invalidate feature cache
        self.feature_cache.clear()

    def compute_features(self, symbol: str) -> Dict[str, float]:
        """Compute feature vector from latest price data.

        Args:
            symbol: Trading symbol.

        Returns:
            Dict with computed features.
        """
        if symbol not in self.price_history:
            return self._default_features()

        ohlcv = self.price_history[symbol]
        close = ohlcv.get("close", [])
        high = ohlcv.get("high", close)
        low = ohlcv.get("low", close)

        if not close:
            return self._default_features()

        # Compute indicators
        rsi = self.indicators.calculate_rsi(close)[-1]
        macd, signal = self.indicators.calculate_macd(close)
        macd_val = macd[-1] if macd else 0.0
        signal_val = signal[-1] if signal else 0.0
        upper, middle, lower = self.indicators.calculate_bollinger_bands(close)
        upper_val = upper[-1] if upper else 0.0
        lower_val = lower[-1] if lower else 0.0
        atr = self.indicators.calculate_atr(high, low, close)[-1]
        sma = self.indicators.calculate_sma(close)[-1]

        # Return feature dict with descriptive names
        return {
            "rsi_14": float(rsi) / 100.0,  # Normalize RSI (0-1)
            "macd_value": float(macd_val),
            "macd_signal": float(signal_val),
            "bb_upper_dist": float(
                upper_val - close[-1]
            ),  # Distance from upper band
            "bb_lower_dist": float(
                close[-1] - lower_val
            ),  # Distance from lower band
            "atr_14": float(atr),
            "price_vs_sma": float(close[-1] - sma),  # Price vs SMA
            "return_1d": float(
                (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            ),
            "return_5d": float(
                (close[-1] - close[-5]) / close[-5] if len(close) > 5 else 0
            ),
            "return_10d": float(
                (close[-1] - close[-10]) / close[-10] if len(close) > 10 else 0
            ),
        }

    @staticmethod
    def _default_features() -> Dict[str, float]:
        """Return default zero features."""
        return {name: 0.0 for name in FEATURE_NAMES}


# Global pipeline instance (singleton)
_pipeline: Optional[DataPipeline] = None


def get_pipeline() -> DataPipeline:
    """Get global data pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
    return _pipeline
