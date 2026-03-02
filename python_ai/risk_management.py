"""
Risk Management utilities for NEO Hybrid AI.

Provides:
- Kelly Criterion optimal position sizing
- ATR-based dynamic stop-loss / take-profit
- Maximum drawdown guard-rails
"""

import logging
from typing import List, Literal

import numpy as np

logger = logging.getLogger(__name__)


# ── Kelly Criterion ───────────────────────────────────────────


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    cap: float = 0.25,
) -> float:
    """Compute the Kelly Criterion fraction.

    The Kelly formula maximises long-run geometric growth:

    .. math::
        f^* = \\frac{p}{1} - \\frac{q}{b}

    where *p* = probability of winning, *q = 1 - p*,
    *b = avg_win / avg_loss*.

    Args:
        win_rate: Historical win probability (0-1).
        avg_win: Average winning trade return (positive).
        avg_loss: Average losing trade return (positive magnitude).
        cap: Hard upper bound on the fraction (safety clamp).

    Returns:
        Optimal bet fraction clamped to ``[0, cap]``.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    b = avg_win / avg_loss

    if b == 0:
        return 0.0

    f_star = p - q / b
    return float(np.clip(f_star, 0.0, cap))


def kelly_position_size(
    equity: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    price: float,
    cap: float = 0.25,
) -> float:
    """Compute position size in base currency units via Kelly.

    Args:
        equity: Current portfolio equity (quote currency).
        win_rate: Historical win probability (0-1).
        avg_win: Average win magnitude.
        avg_loss: Average loss magnitude.
        price: Current asset price.
        cap: Kelly fraction cap.

    Returns:
        Quantity to buy in base currency.

    Raises:
        ValueError: If equity is non-positive.
    """
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    fraction = kelly_fraction(win_rate, avg_win, avg_loss, cap)
    notional = equity * fraction
    if price <= 0:
        return 0.0
    return notional / price


# ── ATR-based Dynamic Stop-Loss ──────────────────────────────


def atr_stop_loss(
    entry_price: float,
    atr_value: float,
    multiplier: float = 2.0,
    side: Literal["long", "short"] = "long",
) -> float:
    """Calculate a dynamic stop-loss based on ATR.

    The stop is placed ``multiplier * ATR`` away from entry.

    Args:
        entry_price: Position entry price.
        atr_value: Current Average True Range value (must be > 0).
        multiplier: ATR multiplier (default 2.0).
        side: ``long`` or ``short``.

    Returns:
        Stop-loss price level.

    Raises:
        ValueError: If atr_value is non-positive.
    """
    if atr_value <= 0:
        raise ValueError(f"atr_value must be positive, got {atr_value}")
    offset = atr_value * multiplier
    if side == "long":
        return entry_price - offset
    return entry_price + offset


def atr_take_profit(
    entry_price: float,
    atr_value: float,
    multiplier: float = 3.0,
    side: Literal["long", "short"] = "long",
) -> float:
    """Calculate a dynamic take-profit based on ATR.

    Args:
        entry_price: Position entry price.
        atr_value: Current ATR value (must be > 0).
        multiplier: ATR multiplier (default 3.0).
        side: ``long`` or ``short``.

    Returns:
        Take-profit price level.

    Raises:
        ValueError: If atr_value is non-positive.
    """
    if atr_value <= 0:
        raise ValueError(f"atr_value must be positive, got {atr_value}")
    offset = atr_value * multiplier
    if side == "long":
        return entry_price + offset
    return entry_price - offset


def atr_trailing_stop(
    highest_since_entry: float,
    atr_value: float,
    multiplier: float = 2.0,
    side: Literal["long", "short"] = "long",
) -> float:
    """Trailing stop that follows the price using ATR.

    For longs, trail below the highest high since entry.
    For shorts, trail above the lowest low since entry.

    Args:
        highest_since_entry: Peak price since entry (for longs)
                            or trough (for shorts).
        atr_value: Current ATR.
        multiplier: ATR multiplier.
        side: ``long`` or ``short``.

    Returns:
        Updated trailing stop price.
    """
    offset = atr_value * multiplier
    if side == "long":
        return highest_since_entry - offset
    return highest_since_entry + offset


# ── Max Drawdown Guard ────────────────────────────────────────


def max_drawdown_from_equity(equity_curve: List[float]) -> float:
    """Compute the maximum drawdown from an equity curve.

    Args:
        equity_curve: List of equity values over time.

    Returns:
        Maximum drawdown as a fraction (0-1).
    """
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / (peak + 1e-10)
    return float(np.max(drawdown))


def should_halt_trading(
    equity_curve: List[float],
    max_allowed_drawdown: float = 0.20,
) -> bool:
    """Return True if max drawdown exceeds the allowed limit.

    Use this as a circuit-breaker for the trading loop.

    Args:
        equity_curve: Recent equity values.
        max_allowed_drawdown: Fraction (e.g. 0.20 = 20 %).

    Returns:
        True if trading should be paused.
    """
    dd = max_drawdown_from_equity(equity_curve)
    if dd >= max_allowed_drawdown:
        logger.warning(
            "Max drawdown %.1f%% exceeds limit %.1f%% — halt trading",
            dd * 100,
            max_allowed_drawdown * 100,
        )
        return True
    return False


# ── Daily Loss Limit ──────────────────────────────────────────


def check_daily_loss_limit(
    starting_equity: float,
    current_equity: float,
    max_daily_loss_pct: float = 0.05,
) -> bool:
    """Check whether today's losses exceed the daily limit.

    Args:
        starting_equity: Equity at the start of the trading day.
        current_equity: Current equity.
        max_daily_loss_pct: Maximum allowed daily loss as a
            fraction (e.g. 0.05 = 5 %).

    Returns:
        True if the daily loss limit has been breached.

    Raises:
        ValueError: If starting_equity is non-positive.
    """
    if starting_equity <= 0:
        raise ValueError(
            f"starting_equity must be positive, got {starting_equity}"
        )
    loss_pct = (starting_equity - current_equity) / starting_equity
    if loss_pct >= max_daily_loss_pct:
        logger.warning(
            "Daily loss %.2f%% exceeds limit %.2f%% \u2014 freeze trading",
            loss_pct * 100,
            max_daily_loss_pct * 100,
        )
        return True
    return False


# ── Position Size Cap ─────────────────────────────────────────


def max_position_size(
    equity: float,
    price: float,
    max_pct_of_equity: float = 0.10,
) -> float:
    """Compute the maximum position size as a fraction of equity.

    Ensures no single position exceeds a configurable
    percentage of total equity.

    Args:
        equity: Current portfolio equity.
        price: Current asset price.
        max_pct_of_equity: Maximum fraction of equity per
            position (e.g. 0.10 = 10 %).

    Returns:
        Maximum quantity in base currency.

    Raises:
        ValueError: If equity or price is non-positive.
    """
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    return (equity * max_pct_of_equity) / price
