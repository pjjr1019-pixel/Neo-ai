"""
Tests for Data Ingestion API.

Covers data validation, historical storage, and ingestion workflows.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from python_ai.data_ingestion_api import (
    DataValidator,
    HistoricalDataStore,
    DataIngestionAPI,
    get_data_ingestion_api,
)


class TestDataValidator:
    """Test DataValidator class."""

    def test_validate_valid_candle(self) -> None:
        """Test validation of a valid OHLCV candle."""
        candle = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        }
        assert DataValidator.validate_candle(candle) is True

    def test_validate_missing_keys(self) -> None:
        """Test validation fails with missing keys."""
        candle = {
            "open": 100.0,
            "high": 105.0,
            "close": 102.0,
        }
        assert DataValidator.validate_candle(candle) is False

    def test_validate_high_less_than_low(self) -> None:
        """Test validation fails when high < low."""
        candle = {
            "open": 100.0,
            "high": 95.0,
            "low": 105.0,
            "close": 102.0,
            "volume": 1000.0,
        }
        assert DataValidator.validate_candle(candle) is False

    def test_validate_high_less_than_prices(self) -> None:
        """Test validation fails when high < open/close."""
        candle = {
            "open": 105.0,
            "high": 100.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        }
        assert DataValidator.validate_candle(candle) is False

    def test_validate_low_greater_than_prices(self) -> None:
        """Test validation fails when low > open/close."""
        candle = {
            "open": 100.0,
            "high": 110.0,
            "low": 105.0,
            "close": 102.0,
            "volume": 1000.0,
        }
        assert DataValidator.validate_candle(candle) is False

    def test_validate_negative_volume(self) -> None:
        """Test validation fails with negative volume."""
        candle = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": -1000.0,
        }
        assert DataValidator.validate_candle(candle) is False

    def test_validate_price_series_valid(self) -> None:
        """Test valid price series passes validation."""
        prices = [100.0, 101.0, 101.5, 101.2, 102.0]
        assert DataValidator.validate_price_series(prices) is True

    def test_validate_price_series_large_jump(self) -> None:
        """Test price series with large jump fails."""
        prices = [100.0, 120.0, 121.0]
        assert DataValidator.validate_price_series(prices) is False

    def test_validate_price_series_single_price(self) -> None:
        """Test single price always passes."""
        prices = [100.0]
        assert DataValidator.validate_price_series(prices) is True


class TestHistoricalDataStore:
    """Test HistoricalDataStore class."""

    def setup_method(self) -> None:
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = HistoricalDataStore(self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_init_creates_directory(self) -> None:
        """Test initialization creates data directory."""
        assert self.store.data_dir.exists()

    def test_save_candles(self) -> None:
        """Test saving candles to disk."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ]
        result = self.store.save_candles("BTC/USD", candles)
        assert result is True

        file_path = self.store.data_dir / "BTC/USD.csv"
        assert file_path.exists()

    def test_load_candles(self) -> None:
        """Test loading candles from disk."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            },
            {
                "open": 102.0,
                "high": 107.0,
                "low": 97.0,
                "close": 104.0,
                "volume": 1100.0,
            },
        ]
        self.store.save_candles("ETH/USD", candles)

        loaded = self.store.load_candles("ETH/USD")
        assert len(loaded) == 2
        assert loaded[0]["close"] == 102.0
        assert loaded[1]["close"] == 104.0

    def test_load_candles_with_limit(self) -> None:
        """Test loading candles with limit."""
        candles = [
            {
                "open": float(i),
                "high": float(i + 5),
                "low": float(i - 5),
                "close": float(i + 2),
                "volume": 1000.0,
            }
            for i in range(100, 110)
        ]
        self.store.save_candles("XRP/USD", candles)

        loaded = self.store.load_candles("XRP/USD", limit=5)
        assert len(loaded) == 5

    def test_load_nonexistent_symbol(self) -> None:
        """Test loading candles for nonexistent symbol."""
        loaded = self.store.load_candles("NONEXISTENT/USD")
        assert loaded == []

    def test_get_latest_candle(self) -> None:
        """Test getting the latest candle."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ]
        self.store.save_candles("ADA/USD", candles)

        latest = self.store.get_latest_candle("ADA/USD")
        assert latest is not None
        assert latest["close"] == 102.0

    def test_append_candles(self) -> None:
        """Test appending candles to existing file."""
        candles1 = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ]
        self.store.save_candles("SOL/USD", candles1, append=False)

        candles2 = [
            {
                "open": 102.0,
                "high": 107.0,
                "low": 97.0,
                "close": 104.0,
                "volume": 1100.0,
            }
        ]
        self.store.save_candles("SOL/USD", candles2, append=True)

        loaded = self.store.load_candles("SOL/USD")
        assert len(loaded) == 2


class TestDataIngestionAPI:
    """Test DataIngestionAPI class."""

    def setup_method(self) -> None:
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.api = DataIngestionAPI(self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_ingest_valid_candles(self) -> None:
        """Test ingesting valid candles."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ]
        result = self.api.ingest_candles("BTC/USD", candles)

        assert result["success"] is True
        assert result["valid"] == 1
        assert result["invalid"] == 0

    def test_ingest_mixed_valid_invalid(self) -> None:
        """Test ingesting mix of valid and invalid candles."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            },
            {
                "open": 100.0,
                "high": 95.0,
                "low": 105.0,
                "close": 102.0,
                "volume": 1000.0,
            },
        ]
        result = self.api.ingest_candles("ETH/USD", candles)

        assert result["valid"] == 1
        assert result["invalid"] == 1

    def test_ingest_price_jump_detected(self) -> None:
        """Test ingestion fails when price jump detected."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            },
            {
                "open": 150.0,
                "high": 155.0,
                "low": 145.0,
                "close": 152.0,
                "volume": 1000.0,
            },
        ]
        result = self.api.ingest_candles("XRP/USD", candles)

        assert result["success"] is False
        assert "Price series validation failed" in result["message"]

    def test_get_historical_data(self) -> None:
        """Test retrieving historical data."""
        candles = [
            {
                "open": float(i),
                "high": float(i + 5),
                "low": float(i - 5),
                "close": float(i + 2),
                "volume": 1000.0,
            }
            for i in range(100, 110)
        ]
        self.api.ingest_candles("ADA/USD", candles)

        data = self.api.get_historical_data("ADA/USD")
        assert len(data) == 10

    def test_get_historical_data_with_limit(self) -> None:
        """Test retrieving historical data with limit."""
        candles = [
            {
                "open": float(i),
                "high": float(i + 5),
                "low": float(i - 5),
                "close": float(i + 2),
                "volume": 1000.0,
            }
            for i in range(100, 110)
        ]
        self.api.ingest_candles("SOL/USD", candles)

        data = self.api.get_historical_data("SOL/USD", limit=5)
        assert len(data) == 5

    def test_get_data_statistics(self) -> None:
        """Test getting data statistics."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            },
            {
                "open": 102.0,
                "high": 107.0,
                "low": 97.0,
                "close": 104.0,
                "volume": 2000.0,
            },
        ]
        self.api.ingest_candles("DOT/USD", candles)

        stats = self.api.get_data_statistics("DOT/USD")
        assert stats["total_candles"] == 2
        assert stats["price_min"] == 102.0
        assert stats["price_max"] == 104.0
        assert stats["volume_min"] == 1000.0
        assert stats["volume_max"] == 2000.0

    def test_get_data_statistics_empty(self) -> None:
        """Test getting statistics for nonexistent symbol."""
        stats = self.api.get_data_statistics("EMPTY/USD")
        assert stats["total_candles"] == 0

    def test_get_ingestion_summary(self) -> None:
        """Test getting ingestion summary."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ]
        self.api.ingest_candles("BTC/USD", candles)
        self.api.ingest_candles("ETH/USD", candles)

        summary = self.api.get_ingestion_summary()
        assert summary["total_candles_ingested"] == 2
        assert len(summary["symbols_processed"]) == 2

    def test_ingest_updates_stats(self) -> None:
        """Test that ingestion updates stats correctly."""
        candles = [
            {
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ] * 5
        self.api.ingest_candles("XRP/USD", candles)

        summary = self.api.get_ingestion_summary()
        assert summary["ingestion_stats"]["XRP/USD"] == 5


class TestGlobalFactory:
    """Test global factory function."""

    def test_get_factory_returns_api(self) -> None:
        """Test factory returns DataIngestionAPI instance."""
        api = get_data_ingestion_api()
        assert isinstance(api, DataIngestionAPI)

    def test_factory_returns_singleton(self) -> None:
        """Test factory returns same instance."""
        api1 = get_data_ingestion_api()
        api2 = get_data_ingestion_api()
        assert api1 is api2
