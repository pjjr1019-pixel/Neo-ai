"""
Data Pipeline for NEO Hybrid AI.

Computes technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
from OHLCV data and prepares feature vectors for model inference.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TechnicalIndicators:
    """Compute technical indicators from price data."""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index (RSI).

        Args:
            prices: List of closing prices.
            period: RSI period (default 14).

        Returns:
            List of RSI values.
        """
        if len(prices) < period + 1:
            return [50.0] * len(prices)

        prices_array = np.array(prices, dtype=float)
        deltas = np.diff(prices_array)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        rsi_values = []
        for i in range(len(prices)):
            if i < period:
                rsi_values.append(50.0)
            else:
                if avg_loss == 0:
                    rsi = 100.0 if avg_gain > 0 else 50.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                rsi_values.append(rsi)

        return rsi_values

    @staticmethod
    def calculate_macd(
        prices: List[float], fast: int = 12, slow: int = 26
    ) -> Tuple[List[float], List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: List of closing prices.
            fast: Fast EMA period (default 12).
            slow: Slow EMA period (default 26).

        Returns:
            Tuple of (MACD values, Signal line values).
        """
        prices_array = np.array(prices, dtype=float)

        # Simple EMA calculation
        ema_fast = prices_array.copy()
        ema_slow = prices_array.copy()

        for i in range(1, len(prices_array)):
            ema_fast[i] = (
                2.0 / (fast + 1) * prices_array[i]
                + (1 - 2.0 / (fast + 1)) * ema_fast[i - 1]
            )
            ema_slow[i] = (
                2.0 / (slow + 1) * prices_array[i]
                + (1 - 2.0 / (slow + 1)) * ema_slow[i - 1]
            )

        macd = ema_fast - ema_slow
        signal = np.convolve(macd, [1 / 9] * 9, mode="same")

        return list(macd), list(signal)

    @staticmethod
    def calculate_bollinger_bands(
        prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands.

        Args:
            prices: List of closing prices.
            period: SMA period (default 20).
            std_dev: Standard deviation multiplier (default 2).

        Returns:
            Tuple of (Upper band, Middle band, Lower band).
        """
        prices_array = np.array(prices, dtype=float)
        middle = np.convolve(prices_array, [1 / period] * period, mode="same")
        std = (
            np.std(prices_array[-period:])
            if len(prices_array) >= period
            else 0
        )
        upper = middle + std * std_dev
        lower = middle - std * std_dev

        return list(upper), list(middle), list(lower)

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

        Args:
            prices: List of closing prices.
            period: SMA period (default 20).

        Returns:
            List of SMA values.
        """
        prices_array = np.array(prices, dtype=float)
        sma_values = []

        for i in range(len(prices_array)):
            if i < period - 1:
                # For initial values, use what we have
                sma_values.append(np.mean(prices_array[: i + 1]))
            else:
                # Use full period
                sma_values.append(
                    np.mean(prices_array[i - period + 1 : i + 1])
                )

        return sma_values


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

        # Return feature dict
        return {
            "f0": float(rsi) / 100.0,  # Normalize RSI (0-1)
            "f1": float(macd_val),
            "f2": float(signal_val),
            "f3": float(upper_val - close[-1]),  # Distance from upper band
            "f4": float(close[-1] - lower_val),  # Distance from lower band
            "f5": float(atr),
            "f6": float(close[-1] - sma),  # Price vs SMA
            "f7": float(
                (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            ),  # Daily return
            "f8": float(
                (close[-1] - close[-5]) / close[-5] if len(close) > 5 else 0
            ),  # 5-day return
            "f9": float(
                (close[-1] - close[-10]) / close[-10] if len(close) > 10 else 0
            ),  # 10-day return
        }

    @staticmethod
    def _default_features() -> Dict[str, float]:
        """Return default zero features."""
        return {f"f{i}": 0.0 for i in range(10)}


# Global pipeline instance (singleton)
_pipeline: Optional[DataPipeline] = None


def get_pipeline() -> DataPipeline:
    """Get global data pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
    return _pipeline
