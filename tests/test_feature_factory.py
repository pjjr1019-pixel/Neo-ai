"""Tests for feature factory rolling indicators and optimizations."""

from python_ai.feature_factory import (
    clear_feature_cache,
    compute_rolling_features,
    ema_last_cached,
    rsi_last_cached,
    sma_last_cached,
)


def test_compute_rolling_features_basic() -> None:
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    out = compute_rolling_features(prices, sma_window=3, ema_span=3, rsi_window=3)
    assert set(out.keys()) == {"sma", "ema", "rsi", "returns_1"}
    assert isinstance(out["sma"], float)
    assert isinstance(out["ema"], float)
    assert isinstance(out["rsi"], float)


def test_compute_rolling_features_empty_series() -> None:
    out = compute_rolling_features([])
    assert out == {"sma": 0.0, "ema": 0.0, "rsi": 0.0, "returns_1": 0.0}


def test_feature_cache_hit_paths() -> None:
    clear_feature_cache()
    series = (1.0, 2.0, 3.0, 4.0, 5.0)
    _ = sma_last_cached(series, 3)
    _ = sma_last_cached(series, 3)
    assert sma_last_cached.cache_info().hits >= 1

    _ = ema_last_cached(series, 3)
    _ = ema_last_cached(series, 3)
    assert ema_last_cached.cache_info().hits >= 1

    _ = rsi_last_cached(series, 3)
    _ = rsi_last_cached(series, 3)
    assert rsi_last_cached.cache_info().hits >= 1
