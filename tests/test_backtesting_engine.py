"""
Tests for Backtesting Engine.

Covers walk-forward simulation, performance metrics calculation,
Sharpe ratio, max drawdown, win rate, and fitness scoring.
"""

from python_ai.backtesting_engine import (
    BacktestingEngine,
    BacktestMetrics,
    get_backtesting_engine,
)


class TestBacktestMetrics:
    """Test BacktestMetrics fitness scoring."""

    def test_metrics_initialization(self) -> None:
        """Test metrics can be initialized with performance values."""
        metrics = BacktestMetrics(
            total_return=15.0,
            sharpe_ratio=1.5,
            max_drawdown=5.0,
            win_rate=60.0,
            num_trades=10,
        )
        assert metrics.total_return == 15.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown == 5.0
        assert metrics.win_rate == 60.0
        assert metrics.num_trades == 10

    def test_metrics_fitness_score_calculation(self) -> None:
        """Test fitness score combines return, Sharpe, and win rate."""
        metrics = BacktestMetrics(
            total_return=100.0,
            sharpe_ratio=3.0,
            max_drawdown=10.0,
            win_rate=100.0,
            num_trades=5,
        )
        assert 0.0 <= metrics.fitness_score <= 1.0

    def test_metrics_to_dict(self) -> None:
        """Test metrics can be converted to dictionary."""
        metrics = BacktestMetrics(
            total_return=20.0,
            sharpe_ratio=2.0,
            max_drawdown=8.0,
            win_rate=70.0,
            num_trades=15,
        )
        data = metrics.to_dict()
        assert data["total_return"] == 20.0
        assert data["sharpe_ratio"] == 2.0
        assert data["max_drawdown"] == 8.0
        assert data["win_rate"] == 70.0
        assert data["num_trades"] == 15
        assert "fitness_score" in data


class TestBacktestingEngine:
    """Test BacktestingEngine simulation logic."""

    def test_engine_initialization(self) -> None:
        """Test backtesting engine initializes with defaults."""
        engine = BacktestingEngine()
        assert engine.initial_capital == 10000.0
        assert engine.transaction_cost_pct == 0.001

    def test_engine_custom_parameters(self) -> None:
        """Test backtesting engine accepts custom parameters."""
        engine = BacktestingEngine(
            initial_capital=50000.0,
            transaction_cost_pct=0.0005,
        )
        assert engine.initial_capital == 50000.0
        assert engine.transaction_cost_pct == 0.0005

    def test_backtest_empty_data(self) -> None:
        """Test backtest with empty OHLCV data returns zero metrics."""
        engine = BacktestingEngine()
        metrics = engine.run_backtest({}, [])
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.num_trades == 0

    def test_backtest_buy_and_hold(self) -> None:
        """Test buy-and-hold strategy with uptrend."""
        engine = BacktestingEngine(initial_capital=1000.0)
        ohlcv = {
            "open": [100.0, 105.0, 110.0, 115.0, 120.0],
            "high": [101.0, 106.0, 111.0, 116.0, 121.0],
            "low": [99.0, 104.0, 109.0, 114.0, 119.0],
            "close": [100.5, 105.5, 110.5, 115.5, 120.5],
            "volume": [1000.0] * 5,
        }
        signals = ["BUY", "HOLD", "HOLD", "HOLD", "SELL"]
        metrics = engine.run_backtest(ohlcv, signals)
        assert metrics.total_return > 15.0
        assert metrics.num_trades == 2

    def test_backtest_downtrend_protection(self) -> None:
        """Test that sell signals limit losses in downtrend."""
        engine = BacktestingEngine(initial_capital=1000.0)
        ohlcv = {
            "open": [100.0, 95.0, 90.0, 85.0, 80.0],
            "high": [101.0, 96.0, 91.0, 86.0, 81.0],
            "low": [99.0, 94.0, 89.0, 84.0, 79.0],
            "close": [99.5, 94.5, 89.5, 84.5, 79.5],
            "volume": [1000.0] * 5,
        }
        signals = ["BUY", "SELL", "HOLD", "HOLD", "HOLD"]
        metrics = engine.run_backtest(ohlcv, signals)
        assert metrics.total_return < 0.0

    def test_backtest_multiple_trades(self) -> None:
        """Test multiple buy/sell cycles."""
        engine = BacktestingEngine(initial_capital=1000.0)
        ohlcv = {
            "open": [100.0, 105.0, 110.0, 105.0, 110.0],
            "high": [101.0, 106.0, 111.0, 106.0, 111.0],
            "low": [99.0, 104.0, 109.0, 104.0, 109.0],
            "close": [100.5, 105.5, 110.5, 105.5, 110.5],
            "volume": [1000.0] * 5,
        }
        signals = ["BUY", "SELL", "BUY", "SELL", "HOLD"]
        metrics = engine.run_backtest(ohlcv, signals)
        assert metrics.num_trades >= 4

    def test_calculate_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        import numpy as np

        engine = BacktestingEngine()
        returns = np.array([0.01, 0.02, -0.01, 0.02, 0.01])
        sharpe = engine._calculate_sharpe(returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0.0

    def test_calculate_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        import numpy as np

        engine = BacktestingEngine()
        values = np.array([1000.0, 1100.0, 950.0, 1050.0, 1200.0])
        max_dd = engine._calculate_max_drawdown(values)
        assert 0.0 <= max_dd <= 100.0

    def test_calculate_win_rate(self) -> None:
        """Test win rate calculation from trades."""
        engine = BacktestingEngine()
        trades = [100.0, -50.0, 200.0, -25.0, 150.0]
        win_rate = engine._calculate_win_rate(trades)
        assert 0.0 <= win_rate <= 100.0
        assert win_rate == 60.0

    def test_calculate_win_rate_no_trades(self) -> None:
        """Test win rate with no trades."""
        engine = BacktestingEngine()
        win_rate = engine._calculate_win_rate([])
        assert win_rate == 0.0


class TestGlobalSingleton:
    """Test global backtesting engine singleton."""

    def test_get_backtesting_engine(self) -> None:
        """Test get_backtesting_engine returns singleton."""
        engine1 = get_backtesting_engine()
        engine2 = get_backtesting_engine()
        assert engine1 is engine2
        assert isinstance(engine1, BacktestingEngine)


class TestBacktestMetricsIntegration:
    """Integration tests for metrics with real backtest scenarios."""

    def test_metrics_with_real_backtest(self) -> None:
        """Test metrics calculation with real backtest results."""
        engine = BacktestingEngine(initial_capital=5000.0)
        ohlcv = {
            "open": [100.0 + i for i in range(20)],
            "high": [101.0 + i for i in range(20)],
            "low": [99.0 + i for i in range(20)],
            "close": [100.5 + i for i in range(20)],
            "volume": [1000.0] * 20,
        }
        signals = ["BUY", "HOLD"] * 5 + ["SELL", "HOLD"] * 5
        metrics = engine.run_backtest(ohlcv, signals)
        metrics_dict = metrics.to_dict()
        assert all(
            key in metrics_dict
            for key in [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "num_trades",
                "fitness_score",
            ]
        )
