"""Tests for Phase 4 data cache, storage, and async IO modules."""

from __future__ import annotations

import pytest

from data.feature_cache import FeatureCache
from data.io import AsyncBatchCandleWriter
from data.storage import CandleStorage


def _rows() -> list[dict[str, float | int]]:
    return [
        {
            "timestamp": 1,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
        },
        {
            "timestamp": 2,
            "open": 101.0,
            "high": 103.0,
            "low": 100.0,
            "close": 102.0,
            "volume": 12.0,
        },
    ]


def test_feature_cache_hit_miss_ratio() -> None:
    cache = FeatureCache()
    cache.set("BTC/USD", 1, {"rsi": 55.0})
    assert cache.get("BTC/USD", 1) == {"rsi": 55.0}
    assert cache.get("BTC/USD", 2) is None
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert 0.0 < cache.hit_ratio() < 1.0


def test_storage_csv_and_parquet_roundtrip(tmp_path) -> None:
    storage = CandleStorage(tmp_path)
    rows = _rows()

    csv_path = storage.save("BTC/USD", rows, fmt="csv")
    loaded_csv = storage.load("BTC/USD", fmt="csv")
    assert csv_path.exists()
    assert len(loaded_csv) == 2

    parquet_path = storage.save("ETH/USD", rows, fmt="parquet")
    loaded_parquet = storage.load("ETH/USD", fmt="parquet")
    assert parquet_path.exists()
    assert len(loaded_parquet) == 2


@pytest.mark.asyncio
async def test_async_batch_writer_flushes(tmp_path) -> None:
    storage = CandleStorage(tmp_path)
    writer = AsyncBatchCandleWriter(
        storage,
        "BTC/USD",
        batch_size=2,
        fmt="csv",
    )
    rows = _rows()
    await writer.add(rows[0])  # no flush yet
    await writer.add(rows[1])  # triggers flush
    loaded = storage.load("BTC/USD", fmt="csv")
    assert len(loaded) == 2
