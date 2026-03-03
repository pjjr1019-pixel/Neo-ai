"""Tests for Phase 5 backtest optimization paths."""

from python_ai.backtesting_engine import BacktestingEngine


def _sample_data():
    ohlcv = {"close": [100.0, 102.0, 101.0, 104.0, 106.0]}
    signals = ["BUY", "HOLD", "HOLD", "SELL", "HOLD"]
    return ohlcv, signals


def test_vectorized_backtest_parity_with_standard() -> None:
    engine = BacktestingEngine(initial_capital=1000.0)
    ohlcv, signals = _sample_data()
    standard = engine.run_backtest(ohlcv, signals)
    vectorized = engine.run_backtest_vectorized(ohlcv, signals)
    assert abs(standard.total_return - vectorized.total_return) < 1e-9
    assert standard.num_trades == vectorized.num_trades


def test_parallel_backtests_returns_all_results() -> None:
    engine = BacktestingEngine(initial_capital=1000.0)
    ohlcv, signals = _sample_data()
    jobs = [(ohlcv, signals), (ohlcv, signals)]
    results = engine.run_parallel_backtests(jobs, max_workers=2)
    assert len(results) == 2
    assert all(hasattr(item, "fitness_score") for item in results)


def test_multi_timeframe_backtest() -> None:
    engine = BacktestingEngine(initial_capital=1000.0)
    ohlcv, signals = _sample_data()
    output = engine.run_multi_timeframe_backtest(
        frames={"1m": ohlcv, "5m": ohlcv},
        signals_by_frame={"1m": signals, "5m": signals},
    )
    assert set(output.keys()) == {"1m", "5m"}
    assert output["1m"].num_trades >= 0
