"""Synthetic data generators for edge-case and stress testing."""

from __future__ import annotations

from typing import Dict, List, Optional


import numpy as np

_faker: Optional["Faker"] = None
try:  # pragma: no cover - optional dependency path
    from faker import Faker

    _faker = Faker()
except Exception:  # pragma: no cover - optional dependency path
    _faker = None


def generate_price_series(length: int = 100, seed: int = 42) -> List[float]:
    """Generate synthetic price series with controlled randomness."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=max(1, length))
    prices = 100.0 * np.cumprod(1.0 + returns)
    return [float(v) for v in prices]


def generate_trade_record(seed: int = 1) -> Dict[str, object]:
    """Generate one synthetic trade-like record."""
    rng = np.random.default_rng(seed)
    symbol = "BTC/USD"
    if _faker is not None:
        trader_id = _faker.uuid4()
    else:
        trader_id = f"user-{int(rng.integers(1, 9999))}"
    return {
        "trader_id": trader_id,
        "symbol": symbol,
        "side": "BUY" if rng.random() > 0.5 else "SELL",
        "qty": float(abs(rng.normal(1.0, 0.2))),
        "price": float(abs(rng.normal(30000.0, 5000.0))),
    }
