"""
Tests for new NEO modules:
- training_data_builder (candles -> ML features)
- ml_model.train() (real training path)
- exchange_feed (LiveExchangeDataFeed with mocked ccxt)
- historical_data_fetcher (fetch -> store -> train pipeline)
- fastapi /learn buffering + retrain
- fastapi /metrics real counters
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from python_ai.data_ingestion_api import HistoricalDataStore
from python_ai.exchange_feed import LiveExchangeDataFeed
from python_ai.fastapi_service.fastapi_service import (
    _learn_buffer,
    _request_counts,
    app,
)
from python_ai.historical_data_fetcher import (
    fetch_candles,
    store_candles,
    train_on_candles,
)
from python_ai.ml_model import MLModel
from python_ai.training_data_builder import (
    TrainingDataBuilder,
    candles_to_ohlcv,
)


def _make_candles(n: int = 100) -> List[Dict[str, float]]:
    """Generate n synthetic candle dicts for testing."""
    candles: List[Dict[str, float]] = []
    base = 100.0
    for i in range(n):
        c = base + np.sin(i * 0.1) * 5
        candles.append(
            {
                "open": c - 0.5,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": 1000.0 + i * 10,
            }
        )
    return candles


class TestCandlesToOhlcv:
    """Tests for candles_to_ohlcv converter."""

    def test_basic_conversion(self) -> None:
        """Convert list-of-dicts to columnar dict."""
        candles = _make_candles(5)
        ohlcv = candles_to_ohlcv(candles)
        assert set(ohlcv.keys()) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert len(ohlcv["close"]) == 5

    def test_empty_list(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            candles_to_ohlcv([])

    def test_values_match(self) -> None:
        """Spot-check that values round-trip correctly."""
        candles = [
            {
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100.0,
            },
        ]
        ohlcv = candles_to_ohlcv(candles)
        assert ohlcv["open"] == [1.0]
        assert ohlcv["close"] == [1.5]


class TestTrainingDataBuilder:
    """Tests for TrainingDataBuilder."""

    def test_build_from_candles(self) -> None:
        """Build X, y matrices from candle list."""
        builder = TrainingDataBuilder(window_size=20)
        candles = _make_candles(100)
        X, y = builder.build_from_candles("TEST", candles)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0
        assert X.shape[1] == 10  # f0 - f9

    def test_too_few_candles(self) -> None:
        """Raise when not enough candles for the window."""
        builder = TrainingDataBuilder(window_size=50)
        candles = _make_candles(10)
        with pytest.raises(ValueError, match="Need at least"):
            builder.build_from_candles("TEST", candles)

    def test_y_values_reasonable(self) -> None:
        """Target values should be forward returns in [-1, 1]."""
        builder = TrainingDataBuilder(window_size=20)
        candles = _make_candles(200)
        _, y = builder.build_from_candles("TEST", candles)
        assert np.all(np.abs(y) < 1.0)


# ──────────────────────────────────────────────────────────────
# MLModel.train() tests
# ──────────────────────────────────────────────────────────────


class TestMLModelTrain:
    """Tests for the real training path."""

    def test_train_returns_metrics(self, tmp_path: Any) -> None:
        """train(X, y) returns a metrics dict with R² and MSE."""
        model = MLModel(model_path=str(tmp_path / "model.pkl"))
        X = np.random.randn(50, 10)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        metrics = model.train(X, y)
        assert "r2_ensemble" in metrics
        assert "mse_ensemble" in metrics
        assert metrics["train_samples"] + metrics["test_samples"] == 50

    def test_train_persists(self, tmp_path: Any) -> None:
        """Model saved after train() can be loaded back."""
        path = str(tmp_path / "model.pkl")
        m1 = MLModel(model_path=path)
        X = np.random.randn(50, 10)
        y = X[:, 0] * 2
        m1.train(X, y)
        assert m1.train_count == 1

        m2 = MLModel(model_path=path)
        assert m2.is_trained
        assert m2.train_count == 1
        assert m2.training_metrics["r2_ensemble"] is not None

    def test_train_too_few_samples(self, tmp_path: Any) -> None:
        """Raise ValueError when < 10 samples."""
        model = MLModel(model_path=str(tmp_path / "model.pkl"))
        with pytest.raises(ValueError, match="Need >= 10"):
            model.train(np.zeros((5, 10)), np.zeros(5))


# ──────────────────────────────────────────────────────────────
# LiveExchangeDataFeed tests (ccxt mocked)
# ──────────────────────────────────────────────────────────────


class TestLiveExchangeDataFeed:
    """Tests for exchange_feed with mocked ccxt."""

    @patch("python_ai.exchange_feed.ccxt")
    def test_init_loads_markets(self, mock_ccxt: Any) -> None:
        """Constructor calls load_markets on the exchange."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {"BTC/USDT": {}}
        mock_ccxt.binance.return_value = mock_exchange
        feed = LiveExchangeDataFeed(exchange_id="binance")
        mock_exchange.load_markets.assert_called_once()
        assert feed.is_connected()

    @patch("python_ai.exchange_feed.ccxt")
    def test_get_latest_candle(self, mock_ccxt: Any) -> None:
        """get_latest_candle returns the second-to-last bar."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {}
        mock_exchange.fetch_ohlcv.return_value = [
            [1000, 100.0, 105.0, 99.0, 103.0, 500.0],
            [2000, 103.0, 108.0, 102.0, 107.0, 600.0],
        ]
        mock_ccxt.binance.return_value = mock_exchange
        feed = LiveExchangeDataFeed(exchange_id="binance")
        candle = feed.get_latest_candle("BTC/USDT")
        assert candle is not None
        assert candle["close"] == 103.0  # second-to-last

    @patch("python_ai.exchange_feed.ccxt")
    def test_fetch_historical_candles(self, mock_ccxt: Any) -> None:
        """fetch_historical_candles returns list of dicts."""
        mock_exchange = MagicMock()
        mock_exchange.markets = {}
        mock_exchange.fetch_ohlcv.return_value = [
            [1000, 1.0, 2.0, 0.5, 1.5, 100.0],
            [2000, 1.5, 2.5, 1.0, 2.0, 200.0],
        ]
        mock_ccxt.binance.return_value = mock_exchange
        feed = LiveExchangeDataFeed(exchange_id="binance")
        candles = feed.fetch_historical_candles(
            "BTC/USDT",
            limit=2,
        )
        assert len(candles) == 2
        assert candles[0]["open"] == 1.0

    def test_unsupported_exchange(self) -> None:
        """Raise ValueError for unknown exchange id."""
        with pytest.raises(ValueError, match="Unsupported"):
            LiveExchangeDataFeed(exchange_id="fake_exchange")


# ──────────────────────────────────────────────────────────────
# Historical data fetcher tests (exchange mocked)
# ──────────────────────────────────────────────────────────────


class TestHistoricalDataFetcher:
    """Tests for historical_data_fetcher functions."""

    def test_store_candles_valid(self, tmp_path: Any) -> None:
        """store_candles writes to HistoricalDataStore."""
        store = HistoricalDataStore(data_dir=str(tmp_path))
        candles = _make_candles(10)
        ok = store_candles(candles, "TEST", store=store)
        assert ok is True

    def test_store_candles_empty(self, tmp_path: Any) -> None:
        """store_candles with empty list returns False."""
        store = HistoricalDataStore(data_dir=str(tmp_path))
        ok = store_candles([], "TEST", store=store)
        assert ok is False

    def test_train_on_candles(self, tmp_path: Any) -> None:
        """train_on_candles produces metrics from candle list."""
        model = MLModel(model_path=str(tmp_path / "model.pkl"))
        candles = _make_candles(200)
        metrics = train_on_candles("TEST", candles, model=model)
        assert "r2_ensemble" in metrics
        assert metrics["train_samples"] > 0

    @patch("python_ai.historical_data_fetcher.LiveExchangeDataFeed")
    def test_fetch_candles_pagination(
        self,
        MockFeed: Any,
    ) -> None:
        """fetch_candles paginates until no more data."""
        feed = MockFeed.return_value
        feed.exchange = MagicMock()
        feed.exchange.rateLimit = 100
        # First call returns 2 bars, second call returns empty
        feed.fetch_historical_candles.side_effect = [
            [
                {
                    "timestamp": 1000.0,
                    "open": 1,
                    "high": 2,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100,
                },
                {
                    "timestamp": 2000.0,
                    "open": 1.5,
                    "high": 2.5,
                    "low": 1.0,
                    "close": 2.0,
                    "volume": 200,
                },
            ],
            [],
        ]
        result = fetch_candles(
            feed,
            "BTC/USDT",
            days=1,
            timeframe="1h",
        )
        assert len(result) == 2


# ──────────────────────────────────────────────────────────────
# FastAPI /learn buffer + /metrics counters tests
# ──────────────────────────────────────────────────────────────


class TestLearnBuffer:
    """Tests for the /learn incremental learning endpoint."""

    def setup_method(self) -> None:
        """Clear buffer and counters before each test."""
        _learn_buffer.clear()
        for k in _request_counts:
            _request_counts[k] = 0

    def test_learn_buffers_sample(self) -> None:
        """POST /learn adds sample to buffer."""
        client = TestClient(app)
        resp = client.post(
            "/learn",
            json={"features": [1, 2, 3], "target": 0.5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "buffered"
        assert body["buffer_size"] >= 1

    def test_learn_retrain_at_threshold(self) -> None:
        """Buffer triggers retrain when threshold is reached."""
        client = TestClient(app)
        # Fill buffer to threshold - 1
        for i in range(49):
            _learn_buffer.append(
                {"features": list(np.random.randn(10)), "target": 0.1},
            )
        # The 50th sample should trigger retrain
        resp = client.post(
            "/learn",
            json={
                "features": list(np.random.randn(10)),
                "target": 0.2,
            },
        )
        body = resp.json()
        assert body["status"] == "retrained"
        assert "metrics" in body
        assert len(_learn_buffer) == 0  # buffer cleared


class TestMetricsEndpoint:
    """Tests for /metrics real counters."""

    def setup_method(self) -> None:
        """Clear counters before each test."""
        for k in _request_counts:
            _request_counts[k] = 0

    def test_metrics_structure(self) -> None:
        """GET /metrics returns expected keys."""
        client = TestClient(app)
        resp = client.get("/metrics")
        body = resp.json()
        assert "request_counts" in body
        assert "total_requests" in body
        assert "uptime_seconds" in body
        assert "model_trained" in body

    def test_metrics_counts_increment(self) -> None:
        """Request counts increment with each call."""
        client = TestClient(app)
        client.post(
            "/predict",
            json={"features": {"f0": 1, "f1": 2, "f2": 3}},
        )
        resp = client.get("/metrics")
        counts = resp.json()["request_counts"]
        assert counts["predict"] >= 1
