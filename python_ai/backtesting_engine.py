"""Backtesting engine with single, vectorized, and parallel run modes."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class BacktestMetrics:
    """Container for backtest performance metrics."""

    __slots__ = (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "num_trades",
        "fitness_score",
    )

    def __init__(
        self,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        num_trades: int,
    ) -> None:
        """Initialize backtest metrics.

        Args:
            total_return: Total return percentage (0-100).
            sharpe_ratio: Annualized Sharpe ratio.
            max_drawdown: Maximum drawdown percentage (0-100).
            win_rate: Winning trades percentage (0-100).
            num_trades: Total number of trades executed.
        """
        self.total_return = total_return
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.num_trades = num_trades
        self.fitness_score = self._calculate_fitness()

    def _calculate_fitness(self) -> float:
        """Calculate composite fitness score for evolution.

        Fitness combines:
        - Return (40% weight, sigmoid-scaled so negatives are
          preserved: -100% → 0.0, 0% → 0.5, +100% → 1.0)
        - Sharpe ratio (40% weight, normalized)
        - Win rate (20% weight)

        Returns:
            Fitness score (0-1 scale).
        """
        # Sigmoid-like linear mapping preserving gradient for
        # negative returns: -100% → 0, 0% → 0.5, +100% → 1.0
        return_score = float(
            np.clip((self.total_return + 100.0) / 200.0, 0.0, 1.0)
        )
        sharpe_score = min(abs(self.sharpe_ratio) / 3.0, 1.0)
        win_score = self.win_rate / 100.0

        fitness = 0.4 * return_score + 0.4 * sharpe_score + 0.2 * win_score
        return float(np.clip(fitness, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "total_return": float(self.total_return),
            "sharpe_ratio": float(self.sharpe_ratio),
            "max_drawdown": float(self.max_drawdown),
            "win_rate": float(self.win_rate),
            "num_trades": int(self.num_trades),
            "fitness_score": float(self.fitness_score),
        }


class BacktestingEngine:
    """Backtest trading strategies against historical OHLCV data."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost_pct: float = 0.001,
    ) -> None:
        """Initialize backtesting engine.

        Args:
            initial_capital: Starting capital in USD.
            transaction_cost_pct: Transaction cost as percentage
                                 (0.001 = 0.1%).
        """
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct

    def run_backtest(
        self,
        ohlcv_data: Dict[str, List[float]],
        signals: List[str],
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        position_size_pct: float = 1.0,
        transaction_cost_pct: Optional[float] = None,
    ) -> BacktestMetrics:
        """Run backtest simulation with given signals and risk controls.

        Args:
            ohlcv_data: Dictionary with keys 'open', 'high', 'low',
                       'close', 'volume'. Each value is list of prices/volumes.
            signals: List of trading signals ('BUY', 'SELL', 'HOLD').
            stop_loss_pct: Stop-loss as a negative decimal
                (e.g., -0.05 for -5%).
            take_profit_pct: Take-profit as a positive decimal
                (e.g., 0.10 for +10%).
            position_size_pct: Fraction of capital to allocate per trade (0-1).

        Returns:
            BacktestMetrics with performance statistics.
        """
        if not ohlcv_data.get("close") or not signals:
            return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0)

        tx_cost = (
            self.transaction_cost_pct
            if transaction_cost_pct is None
            else transaction_cost_pct
        )

        close_prices = np.array(ohlcv_data["close"], dtype=float)
        num_bars = len(close_prices)

        if num_bars < 2:
            return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0)

        portfolio_values = np.empty(num_bars + 1, dtype=float)
        portfolio_values[0] = self.initial_capital
        pv_idx = 0
        position = 0  # 0 = no position, 1 = long
        entry_price = 0.0
        shares = 0.0
        cash = self.initial_capital
        trades: List[float] = []
        num_trades = 0

        for i in range(num_bars):
            if i >= len(signals):
                signal = "HOLD"
            else:
                signal = signals[i]

            current_price = close_prices[i]

            # Check for stop-loss/take-profit triggers if in position
            if position == 1:
                pnl_pct = (current_price - entry_price) / entry_price
                if stop_loss_pct is not None and pnl_pct <= stop_loss_pct:
                    # Trigger stop-loss
                    proceeds = shares * current_price
                    cost = proceeds * tx_cost
                    net_proceeds = proceeds - cost
                    gain = net_proceeds - (shares * entry_price)
                    trades.append(gain)
                    cash = net_proceeds
                    position = 0
                    shares = 0.0
                    num_trades += 1
                    pv_idx += 1
                    portfolio_values[pv_idx] = cash
                    continue
                if take_profit_pct is not None and pnl_pct >= take_profit_pct:
                    # Trigger take-profit
                    proceeds = shares * current_price
                    cost = proceeds * tx_cost
                    net_proceeds = proceeds - cost
                    gain = net_proceeds - (shares * entry_price)
                    trades.append(gain)
                    cash = net_proceeds
                    position = 0
                    shares = 0.0
                    num_trades += 1
                    pv_idx += 1
                    portfolio_values[pv_idx] = cash
                    continue

            if signal == "BUY" and position == 0:
                alloc_cash = cash * position_size_pct
                cost = alloc_cash * tx_cost
                net_value = alloc_cash - cost
                shares = net_value / current_price
                cash -= alloc_cash
                position = 1
                entry_price = current_price
                num_trades += 1
                pv_idx += 1
                portfolio_values[pv_idx] = cash + shares * current_price
            elif signal == "SELL" and position == 1:
                proceeds = shares * current_price
                cost = proceeds * tx_cost
                net_proceeds = proceeds - cost
                gain = net_proceeds - (shares * entry_price)
                trades.append(gain)
                cash += net_proceeds
                position = 0
                shares = 0.0
                num_trades += 1
                pv_idx += 1
                portfolio_values[pv_idx] = cash
            else:
                # HOLD or invalid signal — portfolio = cash + position
                pv_idx += 1
                portfolio_values[pv_idx] = cash + shares * current_price

        values_array = portfolio_values[: pv_idx + 1]

        total_return = (
            (values_array[-1] - self.initial_capital)
            / self.initial_capital
            * 100.0
        )
        daily_returns = np.diff(values_array) / (values_array[:-1] + 1e-10)
        sharpe = self._calculate_sharpe(daily_returns)
        max_dd = self._calculate_max_drawdown(values_array)
        win_rate = self._calculate_win_rate(trades) if trades else 0.0

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=(num_trades),
        )

    def run_backtest_vectorized(
        self,
        ohlcv_data: Dict[str, List[float]],
        signals: List[str],
        *,
        position_size_pct: float = 1.0,
    ) -> BacktestMetrics:
        """Vectorized-friendly wrapper for parity and benchmark checks.

        The core strategy-state transitions (position entry/exit) are
        stateful and handled by ``run_backtest``. This method provides a
        dedicated API entry for optimization and parity tests.
        """
        return self.run_backtest(
            ohlcv_data=ohlcv_data,
            signals=signals,
            position_size_pct=position_size_pct,
        )

    def run_parallel_backtests(
        self,
        jobs: Sequence[Tuple[Dict[str, List[float]], List[str]]],
        *,
        max_workers: int = 4,
    ) -> List[BacktestMetrics]:
        """Run multiple backtests concurrently for strategy sweeps."""
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [
                pool.submit(self.run_backtest, ohlcv_data, signals)
                for ohlcv_data, signals in jobs
            ]
            return [future.result() for future in futures]

    def run_multi_timeframe_backtest(
        self,
        frames: Dict[str, Dict[str, List[float]]],
        signals_by_frame: Dict[str, List[str]],
    ) -> Dict[str, BacktestMetrics]:
        """Evaluate strategy across multiple timeframes."""
        output: Dict[str, BacktestMetrics] = {}
        for frame, ohlcv_data in frames.items():
            signals = signals_by_frame.get(frame, [])
            output[frame] = self.run_backtest(ohlcv_data, signals)
        return output

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from daily returns.

        Args:
            returns: Array of daily returns.

        Returns:
            Annualized Sharpe ratio (assuming 252 trading days).
        """
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        sharpe = mean_ret / std_ret * np.sqrt(252)
        return float(sharpe)

    def _calculate_max_drawdown(
        self,
        values: np.ndarray,
    ) -> float:
        """Calculate maximum drawdown from portfolio values.

        Args:
            values: Array of portfolio values.

        Returns:
            Maximum drawdown as percentage (0-100).
        """
        if len(values) < 2:
            return 0.0

        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax * 100.0
        max_dd = -np.min(drawdown)
        return float(max_dd)

    def _calculate_win_rate(self, trades: List[float]) -> float:
        """Calculate win rate from trade P&L.

        Args:
            trades: List of trade profits/losses.

        Returns:
            Win rate as percentage (0-100).
        """
        if not trades:
            return 0.0

        wins = sum(1 for trade in trades if trade > 0)
        win_rate = (wins / len(trades)) * 100.0
        return float(win_rate)


def get_backtesting_engine() -> BacktestingEngine:
    """Get global backtesting engine singleton.

    Returns:
        BacktestingEngine instance.
    """
    global _backtesting_engine
    if _backtesting_engine is None:
        _backtesting_engine = BacktestingEngine()
    return _backtesting_engine


def reset_backtesting_engine() -> None:
    """Reset the global backtesting engine singleton.

    Use in tests to prevent state leaking between test cases.
    """
    global _backtesting_engine
    _backtesting_engine = None


_backtesting_engine: Optional[BacktestingEngine] = None
