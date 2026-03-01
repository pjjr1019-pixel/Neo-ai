"""
Tests for Autonomous Trading Loop and Market Data Feeds.

Covers simulated data generation, trading cycle execution,
statistics tracking, and loop control.
"""

from unittest.mock import MagicMock

from python_ai.autonomous_trading_loop import (
    AutonomousTradingLoop,
    SimulatedMarketDataFeed,
    get_autonomous_trading_loop,
)


class TestSimulatedMarketDataFeed:
    """Test simulated market data feed."""

    def test_feed_initialization(self) -> None:
        """Test market feed initializes correctly."""
        feed = SimulatedMarketDataFeed(initial_price=100.0, volatility=0.02)
        assert feed.initial_price == 100.0
        assert feed.volatility == 0.02
        assert feed.current_price == 100.0
        assert feed.is_connected()

    def test_generate_candle(self) -> None:
        """Test candle generation with realistic OHLCV."""
        feed = SimulatedMarketDataFeed(initial_price=100.0)
        candle = feed.get_latest_candle("BTC/USD")
        assert candle is not None
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle
        assert "volume" in candle
        assert candle["high"] >= candle["close"]
        assert candle["high"] >= candle["open"]
        assert candle["low"] <= candle["close"]
        assert candle["low"] <= candle["open"]

    def test_price_random_walk(self) -> None:
        """Test that generated prices follow random walk."""
        feed = SimulatedMarketDataFeed(initial_price=100.0, volatility=0.01)
        prices = []
        for _ in range(10):
            candle = feed.get_latest_candle("BTC/USD")
            assert candle is not None
            prices.append(candle["close"])

        assert len(prices) == 10
        assert all(isinstance(p, float) for p in prices)

    def test_subscribe_callback(self) -> None:
        """Test subscription callbacks are triggered."""
        feed = SimulatedMarketDataFeed()
        callback = MagicMock()
        feed.subscribe("BTC/USD", callback)

        candle = feed.get_latest_candle("BTC/USD")
        callback.assert_called_once_with(candle)

    def test_feed_stop(self) -> None:
        """Test stopping the data feed."""
        feed = SimulatedMarketDataFeed()
        assert feed.is_connected()
        feed.stop()
        assert not feed.is_connected()
        assert feed.get_latest_candle("BTC/USD") is None


class TestAutonomousTradingLoop:
    """Test autonomous trading loop."""

    def test_loop_initialization(self) -> None:
        """Test trading loop initializes correctly."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        symbols = ["BTC/USD", "ETH/USD"]

        loop = AutonomousTradingLoop(feed, orchestrator, symbols)
        assert loop.data_feed is feed
        assert loop.orchestrator is orchestrator
        assert loop.symbols == symbols
        assert loop.check_interval_sec == 60.0
        assert not loop.is_running
        assert len(loop.trades_executed) == 0

    def test_price_history_update(self) -> None:
        """Test price history is maintained."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        loop = AutonomousTradingLoop(feed, orchestrator, ["BTC/USD"])

        candle = {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
        loop._update_price_history("BTC/USD", candle)

        prices = loop.last_prices["BTC/USD"]
        assert candle["close"] in prices

    def test_volatility_estimation(self) -> None:
        """Test volatility calculation from prices."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        loop = AutonomousTradingLoop(feed, orchestrator, ["BTC/USD"])

        loop.last_prices["BTC/USD"] = [100.0, 101.0, 99.0, 101.5, 100.5]
        volatility = loop._estimate_volatility("BTC/USD")
        assert 0.0 <= volatility <= 1.0

    def test_execute_trading_cycle(self) -> None:
        """Test trading cycle execution."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        orchestrator.execute_autonomous_cycle.return_value = {
            "symbol": "BTC/USD",
            "signal": "BUY",
            "confidence": 0.75,
            "prediction": 101.5,
        }

        loop = AutonomousTradingLoop(feed, orchestrator, ["BTC/USD"])
        loop.last_prices["BTC/USD"] = [100.0 + i for i in range(20)]

        result = loop._execute_trading_cycle("BTC/USD")
        assert result is not None
        assert result["symbol"] == "BTC/USD"
        assert result["signal"] == "BUY"
        assert len(loop.trades_executed) == 1

    def test_execute_trading_cycle_insufficient_data(self) -> None:
        """Test trading cycle with insufficient price history."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        loop = AutonomousTradingLoop(feed, orchestrator, ["BTC/USD"])

        loop.last_prices["BTC/USD"] = [100.0]
        result = loop._execute_trading_cycle("BTC/USD")
        assert result is None

    def test_get_trades(self) -> None:
        """Test retrieving executed trades."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        orchestrator.execute_autonomous_cycle.return_value = {
            "symbol": "BTC/USD",
            "signal": "BUY",
            "confidence": 0.75,
            "prediction": 101.5,
        }

        loop = AutonomousTradingLoop(feed, orchestrator, ["BTC/USD"])
        loop.last_prices["BTC/USD"] = [100.0 + i for i in range(20)]
        loop._execute_trading_cycle("BTC/USD")

        trades = loop.get_trades()
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC/USD"

    def test_get_statistics(self) -> None:
        """Test trading statistics calculation."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()

        def side_effect(*args, **kwargs):
            return {
                "symbol": "BTC/USD",
                "signal": "BUY",
                "confidence": 0.75,
                "prediction": 101.5,
            }

        orchestrator.execute_autonomous_cycle.side_effect = side_effect

        loop = AutonomousTradingLoop(feed, orchestrator, ["BTC/USD"])
        loop.last_prices["BTC/USD"] = [100.0 + i for i in range(20)]
        loop._execute_trading_cycle("BTC/USD")

        stats = loop.get_statistics()
        assert stats["total_trades"] == 1
        assert stats["buy_signals"] == 1
        assert "avg_confidence" in stats
        assert "BTC/USD" in stats["symbols_traded"]

    def test_empty_statistics(self) -> None:
        """Test statistics with no trades."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        loop = AutonomousTradingLoop(feed, orchestrator, ["BTC/USD"])

        stats = loop.get_statistics()
        assert stats["total_trades"] == 0
        assert stats["buy_signals"] == 0


class TestGlobalFactory:
    """Test global factory function."""

    def test_get_autonomous_trading_loop_defaults(self) -> None:
        """Test factory with default parameters."""
        loop = get_autonomous_trading_loop()
        assert loop is not None
        assert isinstance(loop, AutonomousTradingLoop)
        assert loop.data_feed is not None
        assert loop.orchestrator is not None
        assert len(loop.symbols) > 0

    def test_get_autonomous_trading_loop_custom(self) -> None:
        """Test factory with custom parameters."""
        feed = SimulatedMarketDataFeed()
        orchestrator = MagicMock()
        symbols = ["BTC/USD"]

        loop = get_autonomous_trading_loop(
            data_feed=feed,
            orchestrator_integration=orchestrator,
            symbols=symbols,
        )
        assert loop.data_feed is feed
        assert loop.orchestrator is orchestrator
        assert loop.symbols == symbols
