"""
Backtesting Engine for NEO Hybrid AI.

Performs walk-forward historical simulation with transaction costs,
calculates performance metrics (Sharpe ratio, max drawdown, win rate),
and evaluates trading strategies for evolution engine feedback.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class BacktestMetrics:
    """Container for backtest performance metrics."""

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
        - Return (40% weight)
        - Sharpe ratio (40% weight, normalized)
        - Win rate (20% weight)

        Returns:
            Fitness score (0-1 scale).
        """
        return_score = min(self.total_return / 100.0, 1.0)
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
    ) -> BacktestMetrics:
        """Run backtest simulation with given signals.

        Args:
            ohlcv_data: Dictionary with keys 'open', 'high', 'low',
                       'close', 'volume'. Each value is list of prices/volumes.
            signals: List of trading signals ('BUY', 'SELL', 'HOLD').

        Returns:
            BacktestMetrics with performance statistics.
        """
        if not ohlcv_data.get("close") or not signals:
            return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0)

        close_prices = np.array(ohlcv_data["close"], dtype=float)
        num_bars = len(close_prices)

        if num_bars < 2:
            return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0)

        portfolio_values = [self.initial_capital]
        position = 0  # 0 = no position, 1 = long
        entry_price = 0.0
        shares = 0.0
        trades = []
        num_trades = 0

        for i in range(num_bars):
            if i >= len(signals):
                signal = "HOLD"
            else:
                signal = signals[i]

            current_price = close_prices[i]
            portfolio_value = portfolio_values[-1]

            if signal == "BUY" and position == 0:
                cost = portfolio_value * self.transaction_cost_pct
                net_value = portfolio_value - cost
                shares = net_value / current_price
                position = 1
                entry_price = current_price
                num_trades += 1
                portfolio_values.append(net_value)
            elif signal == "SELL" and position == 1:
                proceeds = shares * current_price
                cost = proceeds * self.transaction_cost_pct
                net_proceeds = proceeds - cost
                gain = net_proceeds - (shares * entry_price)
                trades.append(gain)
                position = 0
                shares = 0.0
                num_trades += 1
                portfolio_values.append(portfolio_values[-1] + gain)
            else:
                if position == 1:
                    current_value = shares * current_price
                    portfolio_values.append(current_value)
                else:
                    portfolio_values.append(portfolio_values[-1])

        values_array = np.array(portfolio_values, dtype=float)

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


_backtesting_engine: Optional[BacktestingEngine] = None
