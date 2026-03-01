"""
Historical data fetcher for NEO Hybrid AI.

Downloads OHLCV history from a cryptocurrency exchange, stores it
via HistoricalDataStore, and (optionally) trains the ML model on
the freshly ingested data.

Usage::

    python -m python_ai.historical_data_fetcher \\
        --exchange binance --symbol BTC/USDT --days 90

Or programmatically::

    from python_ai.historical_data_fetcher import fetch_and_train
    metrics = fetch_and_train("BTC/USDT", days=90)
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from python_ai.data_ingestion_api import (
    DataValidator,
    HistoricalDataStore,
)
from python_ai.exchange_feed import LiveExchangeDataFeed
from python_ai.ml_model import MLModel, get_model
from python_ai.training_data_builder import TrainingDataBuilder

logger = logging.getLogger(__name__)

# ccxt usually caps a single request at 500–1000 bars.
_MAX_BARS_PER_REQUEST = 500

# Milliseconds per timeframe string
_TF_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def fetch_candles(
    feed: LiveExchangeDataFeed,
    symbol: str,
    days: int = 30,
    timeframe: str = "1h",
) -> List[Dict[str, float]]:
    """Download historical candles with automatic pagination.

    Args:
        feed: An initialised LiveExchangeDataFeed.
        symbol: Trading pair, e.g. 'BTC/USDT'.
        days: How many days of history to fetch.
        timeframe: Bar period (must be in _TF_MS).

    Returns:
        List of candle dicts sorted oldest-first.
    """
    tf_ms = _TF_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported: {list(_TF_MS.keys())}"
        )

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - days * 86_400_000
    all_candles: List[Dict[str, float]] = []

    logger.info(
        "Fetching %s %s candles since %s …",
        symbol,
        timeframe,
        datetime.fromtimestamp(
            since_ms / 1000,
            tz=timezone.utc,
        ).isoformat(),
    )

    while since_ms < now_ms:
        batch = feed.fetch_historical_candles(
            symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=_MAX_BARS_PER_REQUEST,
        )
        if not batch:
            break

        all_candles.extend(batch)
        # Move cursor past last received bar
        last_ts = int(batch[-1]["timestamp"])
        if last_ts <= since_ms:
            break
        since_ms = last_ts + tf_ms

        # Respect rate limits
        time.sleep(feed.exchange.rateLimit / 1000)

    logger.info("Fetched %d candles for %s", len(all_candles), symbol)
    return all_candles


def store_candles(
    candles: List[Dict[str, float]],
    symbol: str,
    store: Optional[HistoricalDataStore] = None,
) -> bool:
    """Validate & persist candles to the HistoricalDataStore.

    Args:
        candles: Raw candle dicts from the exchange.
        symbol: Trading pair.
        store: Optional custom store; default creates a new one.

    Returns:
        True when at least one candle was saved.
    """
    if store is None:
        store = HistoricalDataStore()

    validator = DataValidator()
    valid: List[Dict[str, float]] = [
        c for c in candles if validator.validate_candle(c)
    ]

    if not valid:
        logger.warning("No valid candles to store for %s", symbol)
        return False

    ok = store.save_candles(symbol, valid, append=False)
    logger.info(
        "Stored %d / %d valid candles for %s (ok=%s)",
        len(valid),
        len(candles),
        symbol,
        ok,
    )
    return ok


def train_on_candles(
    symbol: str,
    candles: List[Dict[str, float]],
    model: Optional[MLModel] = None,
) -> Dict[str, Any]:
    """Build features from candles and train the model.

    Args:
        symbol: Trading pair.
        candles: List of OHLCV candle dicts.
        model: Optional model instance; uses singleton otherwise.

    Returns:
        Training metrics dict from ``MLModel.train()``.
    """
    builder = TrainingDataBuilder()
    X, y = builder.build_from_candles(symbol, candles)

    if model is None:
        model = get_model()
    return model.train(X, y)


def fetch_and_train(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    days: int = 30,
    timeframe: str = "1h",
) -> Dict[str, Any]:
    """End-to-end: download → store → train.

    Args:
        symbol: Trading pair.
        exchange_id: Exchange name.
        days: Days of history.
        timeframe: Bar period.

    Returns:
        Training metrics dict.
    """
    feed = LiveExchangeDataFeed(
        exchange_id=exchange_id,
        timeframe=timeframe,
    )
    candles = fetch_candles(feed, symbol, days, timeframe)
    store_candles(candles, symbol)
    metrics = train_on_candles(symbol, candles)
    logger.info("Training metrics: %s", metrics)
    return metrics


# ── CLI entry point ───────────────────────────────────────────


def _cli() -> None:  # pragma: no cover
    """Parse args and run fetch_and_train."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="NEO: fetch historical data & train model",
    )
    parser.add_argument(
        "--exchange",
        default="binance",
        choices=["binance", "coinbase", "kraken"],
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=list(_TF_MS.keys()),
    )
    args = parser.parse_args()

    metrics = fetch_and_train(
        symbol=args.symbol,
        exchange_id=args.exchange,
        days=args.days,
        timeframe=args.timeframe,
    )
    print(f"\nTraining complete.  Metrics:\n{metrics}")


if __name__ == "__main__":
    _cli()
