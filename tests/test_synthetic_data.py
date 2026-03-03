"""Tests for synthetic test data generators."""

from python_ai.synthetic_data import generate_price_series, generate_trade_record


def test_generate_price_series_length_and_type() -> None:
    series = generate_price_series(length=32, seed=7)
    assert len(series) == 32
    assert all(isinstance(v, float) for v in series)


def test_generate_trade_record_shape() -> None:
    record = generate_trade_record(seed=3)
    assert set(record.keys()) == {"trader_id", "symbol", "side", "qty", "price"}
    assert record["side"] in {"BUY", "SELL"}
