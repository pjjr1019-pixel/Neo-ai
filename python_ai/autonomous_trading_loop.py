"""
Real-time Market Data Feed for NEO Hybrid AI.

Provides streaming OHLCV data from market sources or simulations.
Supports live trading data feeds and historical data replay.
"""

from typing import Any, Callable, Dict, List, Optional
import time
from datetime import datetime

import numpy as np


class MarketDataFeed:
    """Abstract base class for market data feeds."""

    def get_latest_candle(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get the latest OHLCV candle for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD').

        Returns:
            Dict with open, high, low, close, volume or None.
        """
        raise NotImplementedError

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[Dict[str, float]], None],
    ) -> None:
        """Subscribe to price updates for a symbol.

        Args:
            symbol: Trading symbol.
            callback: Function called with each new candle.
        """
        raise NotImplementedError

    def is_connected(self) -> bool:
        """Check if data feed is connected and healthy."""
        raise NotImplementedError


class SimulatedMarketDataFeed(MarketDataFeed):
    """Simulated market data feed for testing and backtesting.

    Generates realistic OHLCV data with random walk behavior.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        volatility: float = 0.02,
    ) -> None:
        """Initialize simulated data feed.

        Args:
            initial_price: Starting price level.
            volatility: Daily volatility (std dev of returns).
        """
        self.initial_price = initial_price
        self.volatility = volatility
        self.current_price = initial_price
        self.last_candle: Optional[Dict[str, float]] = None
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.is_running = True

    def get_latest_candle(self, symbol: str) -> Optional[Dict[str, float]]:
        """Generate a simulated candle with random walk.

        Args:
            symbol: Trading symbol (unused in simulation).

        Returns:
            Simulated OHLCV candle.
        """
        if not self.is_running:
            return None

        return_pct = np.random.normal(0, self.volatility)
        new_price = self.current_price * (1 + return_pct)

        open_price = self.current_price
        close_price = new_price
        high_price = max(open_price, close_price) * (
            1 + abs(np.random.normal(0, self.volatility / 2))
        )
        low_price = min(open_price, close_price) * (
            1 - abs(np.random.normal(0, self.volatility / 2))
        )
        volume = np.random.uniform(1000, 10000)

        self.current_price = new_price
        candle = {
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
        }
        self.last_candle = candle

        if symbol in self.subscriptions:
            for callback in self.subscriptions[symbol]:
                callback(candle)

        return candle

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[Dict[str, float]], None],
    ) -> None:
        """Subscribe to simulated price updates.

        Args:
            symbol: Trading symbol.
            callback: Called with each new candle.
        """
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        self.subscriptions[symbol].append(callback)

    def is_connected(self) -> bool:
        """Check if feed is running."""
        return self.is_running

    def stop(self) -> None:
        """Stop the simulated feed."""
        self.is_running = False


class AutonomousTradingLoop:
    """Real-time autonomous trading execution loop."""

    def __init__(
        self,
        data_feed: MarketDataFeed,
        orchestrator_integration: Any,
        symbols: List[str],
        check_interval_sec: float = 60.0,
    ) -> None:
        """Initialize autonomous trading loop.

        Args:
            data_feed: MarketDataFeed for price data.
            orchestrator_integration: OrchestratorIntegration instance.
            symbols: List of trading symbols.
            check_interval_sec: Seconds between trading checks.
        """
        self.data_feed = data_feed
        self.orchestrator = orchestrator_integration
        self.symbols = symbols
        self.check_interval_sec = check_interval_sec
        self.is_running = False
        self.trades_executed: List[Dict[str, Any]] = []
        self.last_prices: Dict[str, List[float]] = {
            symbol: [100.0] * 20 for symbol in symbols
        }
        self.volatility_estimates: Dict[str, float] = {
            symbol: 0.02 for symbol in symbols
        }

    def _update_price_history(self, symbol: str, candle: Dict[str, float]) -> None:
        """Update rolling price history for feature computation.

        Args:
            symbol: Trading symbol.
            candle: Latest OHLCV candle.
        """
        if symbol not in self.last_prices:
            self.last_prices[symbol] = []

        prices = self.last_prices[symbol]
        prices.append(candle["close"])

        if len(prices) > 100:
            prices.pop(0)

    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate current market volatility from recent prices.

        Args:
            symbol: Trading symbol.

        Returns:
            Volatility estimate (0-1 scale).
        """
        prices = self.last_prices.get(symbol, [100.0])
        if len(prices) < 2:
            return 0.02

        returns = np.diff(prices) / prices[:-1]
        volatility = float(np.std(returns))
        self.volatility_estimates[symbol] = volatility

        return volatility

    def _execute_trading_cycle(self, symbol: str) -> Optional[Dict]:
        """Execute one trading cycle for a symbol.

        Args:
            symbol: Trading symbol to trade.

        Returns:
            Trade execution record or None.
        """
        try:
            prices = self.last_prices.get(symbol, [])
            if len(prices) < 20:
                return None

            ohlcv_data = {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000.0] * len(prices),
            }

            volatility = self._estimate_volatility(symbol)

            cycle_result = self.orchestrator.execute_autonomous_cycle(
                symbol,
                ohlcv_data,
                volatility,
            )

            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "signal": cycle_result.get("signal"),
                "confidence": cycle_result.get("confidence"),
                "prediction": cycle_result.get("prediction"),
                "volatility": volatility,
            }
            self.trades_executed.append(trade_record)

            return trade_record
        except Exception as e:
            print(f"Trading cycle failed for {symbol}: {e}")
            return None

    def run(self, duration_seconds: Optional[float] = None) -> None:
        """Run the autonomous trading loop.

        Args:
            duration_seconds: Maximum duration to run (None = infinite).
        """
        if not self.data_feed.is_connected():
            raise RuntimeError("Data feed not connected")

        self.is_running = True
        start_time = time.time()

        print(
            f"Starting autonomous trading loop for "
            f"{self.symbols} ({duration_seconds or 'infinite'} sec)"
        )

        try:
            while self.is_running:
                if duration_seconds and (time.time() - start_time > duration_seconds):
                    break

                for symbol in self.symbols:
                    candle = self.data_feed.get_latest_candle(symbol)
                    if candle:
                        self._update_price_history(symbol, candle)
                        self._execute_trading_cycle(symbol)

                time.sleep(self.check_interval_sec)
        except KeyboardInterrupt:
            print("Trading loop interrupted by user")
        finally:
            self.is_running = False
            print(
                f"Trading loop stopped. Executed {len(self.trades_executed)}" f" trades"
            )

    def stop(self) -> None:
        """Stop the trading loop."""
        self.is_running = False

    def get_trades(self) -> List[Dict[str, Any]]:
        """Get all executed trades."""
        return self.trades_executed.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        if not self.trades_executed:
            return {
                "total_trades": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "avg_confidence": 0.0,
            }

        trades = self.trades_executed
        buy_trades = [t for t in trades if t["signal"] == "BUY"]
        sell_trades = [t for t in trades if t["signal"] == "SELL"]
        avg_conf = np.mean([t["confidence"] for t in trades])

        return {
            "total_trades": len(trades),
            "buy_signals": len(buy_trades),
            "sell_signals": len(sell_trades),
            "hold_signals": len([t for t in trades if t["signal"] == "HOLD"]),
            "avg_confidence": float(avg_conf),
            "symbols_traded": list(set(t["symbol"] for t in trades)),
        }


def get_autonomous_trading_loop(
    data_feed: Optional[MarketDataFeed] = None,
    orchestrator_integration: Optional[Any] = None,
    symbols: Optional[List[str]] = None,
) -> AutonomousTradingLoop:
    """Factory function for autonomous trading loop.

    Args:
        data_feed: Market data feed (creates simulated if None).
        orchestrator_integration: Integration layer.
        symbols: Trading symbols.

    Returns:
        AutonomousTradingLoop instance.
    """
    if data_feed is None:
        data_feed = SimulatedMarketDataFeed()

    if orchestrator_integration is None:
        from python_ai.orchestrator_integration import (
            get_orchestrator_integration,
        )

        orchestrator_integration = get_orchestrator_integration()

    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD"]

    return AutonomousTradingLoop(
        data_feed,
        orchestrator_integration,
        symbols,
    )
