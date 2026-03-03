"""Chart data transforms for candlestick, indicators, and P&L visualization."""

from __future__ import annotations

from typing import Dict, Iterable, List


def candlestick_series(
    rows: Iterable[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Normalize candle rows for GUI chart rendering."""
    return [
        {
            "open": float(item["open"]),
            "high": float(item["high"]),
            "low": float(item["low"]),
            "close": float(item["close"]),
        }
        for item in rows
    ]


def pnl_series(
    trades: Iterable[Dict[str, float]],
) -> List[float]:
    """Build cumulative P&L curve from trade outcomes."""
    cumulative = 0.0
    out: List[float] = []
    for trade in trades:
        cumulative += float(trade.get("pnl", 0.0))
        out.append(cumulative)
    return out


def indicator_overlay(
    base_prices: Iterable[float],
    indicators: Dict[str, Iterable[float]],
) -> Dict[str, List[float]]:
    """Merge base price series with indicator overlays."""
    output = {"price": [float(v) for v in base_prices]}
    for name, values in indicators.items():
        output[name] = [float(v) for v in values]
    return output
