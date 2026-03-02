"""Tests for Data Pipeline."""

from python_ai.data_pipeline import (
    DataPipeline,
    TechnicalIndicators,
    get_pipeline,
)


class TestTechnicalIndicators:
    """Test suite for Technical Indicators."""

    def test_rsi_calculation(self) -> None:
        """Test RSI calculation."""
        ti = TechnicalIndicators()
        prices = [100.0 + i * 0.5 for i in range(30)]
        rsi = ti.calculate_rsi(prices)
        assert len(rsi) == 30
        assert all(0 <= r <= 100 for r in rsi)

    def test_macd_calculation(self) -> None:
        """Test MACD calculation."""
        ti = TechnicalIndicators()
        prices = [100.0 + i * 0.5 for i in range(50)]
        macd, signal = ti.calculate_macd(prices)
        assert len(macd) == 50
        assert len(signal) == 50

    def test_bollinger_bands(self) -> None:
        """Test Bollinger Bands calculation."""
        ti = TechnicalIndicators()
        prices = [100.0 + i * 0.5 for i in range(30)]
        upper, middle, lower = ti.calculate_bollinger_bands(prices)
        assert len(upper) == 30
        assert len(middle) == 30
        assert len(lower) == 30
        # Upper should be >= middle >= lower
        for u, m, l in zip(upper, middle, lower):
            assert u >= m >= l

    def test_atr_calculation(self) -> None:
        """Test ATR calculation."""
        ti = TechnicalIndicators()
        high = [100.0 + i for i in range(30)]
        low = [99.0 + i for i in range(30)]
        close = [99.5 + i for i in range(30)]
        atr = ti.calculate_atr(high, low, close)
        assert len(atr) == 30
        assert all(val >= 0 for val in atr)

    def test_sma_calculation(self) -> None:
        """Test SMA calculation."""
        ti = TechnicalIndicators()
        prices = [100.0] * 30
        sma = ti.calculate_sma(prices, period=10)
        assert len(sma) == 30
        # SMA of constant values should be close to that value
        assert all(95.0 <= val <= 105.0 for val in sma)


class TestDataPipeline:
    """Test suite for Data Pipeline."""

    def test_pipeline_initialization(self) -> None:
        """Test pipeline can be initialized."""
        pipeline = DataPipeline()
        assert pipeline is not None
        assert len(pipeline.price_history) == 0

    def test_update_price_data(self) -> None:
        """Test updating price data."""
        pipeline = DataPipeline()
        ohlcv = {
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
        }
        pipeline.update_price_data("BTC/USD", ohlcv)
        assert "BTC/USD" in pipeline.price_history

    def test_compute_features(self) -> None:
        """Test feature computation."""
        pipeline = DataPipeline()
        ohlcv = {
            "open": [100.0 + i * 0.1 for i in range(20)],
            "high": [101.0 + i * 0.1 for i in range(20)],
            "low": [99.0 + i * 0.1 for i in range(20)],
            "close": [100.5 + i * 0.1 for i in range(20)],
            "volume": [1000.0] * 20,
        }
        pipeline.update_price_data("BTC/USD", ohlcv)
        features = pipeline.compute_features("BTC/USD")

        assert isinstance(features, dict)
        assert len(features) == 10
        assert all(f"f{i}" in features for i in range(10))
        assert all(isinstance(v, float) for v in features.values())

    def test_default_features(self) -> None:
        """Test default features for missing symbol."""
        pipeline = DataPipeline()
        features = pipeline.compute_features("NONEXISTENT")
        assert len(features) == 10
        assert all(val == 0.0 for val in features.values())

    def test_get_pipeline_singleton(self) -> None:
        """Test get_pipeline returns same instance."""
        p1 = get_pipeline()
        p2 = get_pipeline()
        assert p1 is p2


class TestFeatureNormalization:
    """Test feature normalization and scaling."""

    def test_rsi_normalization(self) -> None:
        """Test RSI is normalized to 0-1 range."""
        ti = TechnicalIndicators()
        prices = [100.0 + i * 1.0 for i in range(50)]
        rsi = ti.calculate_rsi(prices)
        normalized = [r / 100.0 for r in rsi]
        assert all(0 <= val <= 1 for val in normalized)

    def test_multiple_symbols(self) -> None:
        """Test pipeline handles multiple symbols."""
        pipeline = DataPipeline()
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]
        for symbol in symbols:
            ohlcv = {
                "open": [100.0 + i * 0.1 for i in range(20)],
                "high": [101.0 + i * 0.1 for i in range(20)],
                "low": [99.0 + i * 0.1 for i in range(20)],
                "close": [100.5 + i * 0.1 for i in range(20)],
                "volume": [1000.0] * 20,
            }
            pipeline.update_price_data(symbol, ohlcv)

        for symbol in symbols:
            features = pipeline.compute_features(symbol)
            assert len(features) == 10
