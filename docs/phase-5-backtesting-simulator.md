# Phase 5 Backtesting Simulator

## Implemented
- Standard backtest mode in `python_ai/backtesting_engine.py`.
- Vectorized-parity API: `run_backtest_vectorized`.
- Parallel strategy sweep API: `run_parallel_backtests`.
- Multi-timeframe evaluation API: `run_multi_timeframe_backtest`.
- Transaction cost overrides supported per run.

## Validation
- `tests/test_backtest.py`
- Existing suite: `tests/test_backtesting_engine.py`
