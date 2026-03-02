"""
Vectorized Technical Indicator Computation for NEO.

Replaces Python-loop implementations with pure-numpy
vectorized calculations for RSI, MACD, SMA, EMA, and
Bollinger Bands.  Typically 10-50× faster than scalar
equivalents on large datasets.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def sma(prices: List[float], window: int = 14) -> np.ndarray:
    """Simple Moving Average (vectorized).

    Args:
        prices: Price series.
        window: Lookback window.

    Returns:
        Array with ``NaN`` for the warm-up period.
    """
    arr = np.array(prices, dtype=np.float64)
    if len(arr) < window:
        return np.full(len(arr), np.nan)
    kernel = np.ones(window) / window
    result = np.convolve(arr, kernel, mode="full")[: len(arr)]
    result[: window - 1] = np.nan
    return result


def ema(prices: List[float], span: int = 14) -> np.ndarray:
    """Exponential Moving Average (vectorized via pandas EWM).

    Uses the standard ``alpha = 2 / (span + 1)`` decay.

    Args:
        prices: Price series.
        span: EMA span.

    Returns:
        Array of EMA values.
    """
    import pandas as pd

    s = pd.Series(prices, dtype=np.float64)
    result: np.ndarray = np.asarray(
        s.ewm(span=span, adjust=False).mean(),
    )
    return result


def rsi(prices: List[float], window: int = 14) -> np.ndarray:
    """Relative Strength Index (vectorized via pandas EWM).

    Uses Wilder's smoothing (``com = window - 1``) which matches
    the industry-standard RSI used by TradingView / StockCharts.

    Args:
        prices: Price series.
        window: RSI lookback period.

    Returns:
        Array of RSI values (0-100), ``NaN`` during warm-up.
    """
    import pandas as pd

    s = pd.Series(prices, dtype=np.float64)
    if len(s) < window + 1:
        return np.full(len(s), np.nan)

    delta = s.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Wilder's smoothing: com = window - 1  →  alpha = 1/window
    avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = losses.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))

    # Keep NaN during warm-up
    result.iloc[:window] = np.nan

    out: np.ndarray = np.asarray(result)
    return out


def macd(
    prices: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, np.ndarray]:
    """MACD indicator (vectorized).

    Args:
        prices: Price series.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal line EMA span.

    Returns:
        Dict with ``"macd"``, ``"signal"``,
        ``"histogram"`` arrays.
    """
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line.tolist(), signal)
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def bollinger_bands(
    prices: List[float],
    window: int = 20,
    num_std: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Bollinger Bands (vectorized via pandas rolling).

    Args:
        prices: Price series.
        window: SMA window.
        num_std: Width in standard deviations.

    Returns:
        Dict with ``"upper"``, ``"middle"``,
        ``"lower"`` arrays.
    """
    import pandas as pd

    s = pd.Series(prices, dtype=np.float64)
    middle = s.rolling(window=window, min_periods=window).mean()
    std_arr = s.rolling(window=window, min_periods=window).std(ddof=1)

    upper = middle + num_std * std_arr
    lower = middle - num_std * std_arr
    return {
        "upper": upper.to_numpy(),
        "middle": middle.to_numpy(),
        "lower": lower.to_numpy(),
    }


def compute_all(
    prices: List[float],
) -> Dict[str, np.ndarray]:
    """Compute all indicators in a single pass.

    Args:
        prices: Price series.

    Returns:
        Dict keyed by indicator name.
    """
    result: Dict[str, np.ndarray] = {}
    result["sma_14"] = sma(prices, 14)
    result["sma_50"] = sma(prices, 50)
    result["ema_12"] = ema(prices, 12)
    result["ema_26"] = ema(prices, 26)
    result["rsi_14"] = rsi(prices, 14)

    m = macd(prices)
    result["macd"] = m["macd"]
    result["macd_signal"] = m["signal"]
    result["macd_histogram"] = m["histogram"]

    bb = bollinger_bands(prices)
    result["bb_upper"] = bb["upper"]
    result["bb_middle"] = bb["middle"]
    result["bb_lower"] = bb["lower"]

    return result
