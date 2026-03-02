"""
Paper Trading Engine for NEO Hybrid AI.

Simulates order execution with realistic fills, slippage, and
fees against live or simulated market data.  Uses the
PortfolioTracker for position management and the MLModel for
signal generation.

No real money is at risk — perfect for strategy validation.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from python_ai.position_tracker import PortfolioTracker, Position

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Simulated trading engine with realistic execution.

    Attributes:
        portfolio: Underlying PortfolioTracker.
        fee_pct: Simulated trading fee (e.g. 0.001 = 0.1 %).
        slippage_pct: Simulated slippage (e.g. 0.0005 = 0.05 %).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        max_position_pct: float = 0.25,
    ) -> None:
        """Initialise the paper trading engine.

        Args:
            initial_capital: Starting balance in quote currency.
            fee_pct: Trading fee as a fraction (0.001 = 0.1 %).
            slippage_pct: Simulated slippage as a fraction.
            max_position_pct: Max fraction of equity per position.
        """
        self.portfolio = PortfolioTracker(initial_capital)
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self._order_log: List[Dict[str, Any]] = []
        self._trade_count = 0

    # ── Order execution ───────────────────────────────────────

    def execute_buy(
        self,
        symbol: str,
        price: float,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Position]:
        """Simulate a market BUY.

        If *quantity* is ``None``, the engine sizes the position
        to ``max_position_pct`` of current equity.

        Args:
            symbol: Trading pair.
            price: Intended fill price.
            quantity: Desired quantity (None → auto-size).
            stop_loss: Optional stop-loss level.
            take_profit: Optional take-profit level.

        Returns:
            The opened Position, or None if rejected.
        """
        fill_price = price * (1.0 + self.slippage_pct)
        fee = fill_price * self.fee_pct

        if quantity is None:
            max_spend = self.portfolio.balance * self.max_position_pct
            quantity = max_spend / (fill_price + fee)

        if quantity <= 0:
            return None

        cost = (fill_price + fee) * quantity
        if cost > self.portfolio.balance:
            logger.warning(
                "BUY rejected — cost %.2f > balance %.2f",
                cost,
                self.portfolio.balance,
            )
            return None

        pos = self.portfolio.open_position(
            symbol=symbol,
            side="long",
            price=fill_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={"fee": fee * quantity},
        )
        self._record_order("BUY", symbol, fill_price, quantity, fee)
        return pos

    def execute_sell(
        self,
        symbol: str,
        price: float,
        position: Optional[Position] = None,
    ) -> Optional[float]:
        """Simulate a market SELL (close an existing long).

        If *position* is ``None``, the most recent open long for
        *symbol* is closed.

        Args:
            symbol: Trading pair.
            price: Intended fill price.
            position: Specific position to close.

        Returns:
            Realised P&L, or None if nothing to sell.
        """
        if position is None:
            longs = [
                p
                for p in self.portfolio.open_positions
                if p.symbol == symbol and p.side == "long"
            ]
            if not longs:
                return None
            position = longs[-1]

        fill_price = price * (1.0 - self.slippage_pct)
        fee = fill_price * self.fee_pct * position.quantity

        pnl = self.portfolio.close_position(position, fill_price)
        pnl -= fee  # deduct fee from realised P&L
        self.portfolio.balance -= fee

        self._record_order(
            "SELL",
            symbol,
            fill_price,
            position.quantity,
            fee / position.quantity,
        )
        return pnl

    # ── Signal-driven trading ─────────────────────────────────

    def act_on_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        confidence: float = 0.5,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute based on a BUY / SELL / HOLD signal.

        Args:
            symbol: Trading pair.
            signal: One of ``BUY``, ``SELL``, ``HOLD``.
            price: Current market price.
            confidence: Model confidence (0-1).
            stop_loss: Optional stop-loss price.
            take_profit: Optional take-profit price.

        Returns:
            Dict describing the action taken, or None for HOLD.
        """
        if signal == "BUY":
            pos = self.execute_buy(
                symbol,
                price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if pos:
                return {
                    "action": "BUY",
                    "symbol": symbol,
                    "price": pos.entry_price,
                    "quantity": pos.quantity,
                    "confidence": confidence,
                }
        elif signal == "SELL":
            pnl = self.execute_sell(symbol, price)
            if pnl is not None:
                return {
                    "action": "SELL",
                    "symbol": symbol,
                    "price": price,
                    "pnl": pnl,
                    "confidence": confidence,
                }
        return None

    # ── Stop-loss / take-profit sweep ─────────────────────────

    def check_exits(
        self,
        prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Close positions that hit stop-loss or take-profit.

        Args:
            prices: Current prices keyed by symbol.

        Returns:
            List of exit records.
        """
        exits: List[Dict[str, Any]] = []
        for pos in list(self.portfolio.open_positions):
            current = prices.get(pos.symbol)
            if current is None:
                continue

            if pos.should_stop_out(current):
                pnl = self.execute_sell(pos.symbol, current, pos)
                exits.append(
                    {
                        "reason": "stop_loss",
                        "symbol": pos.symbol,
                        "price": current,
                        "pnl": pnl,
                    }
                )
            elif pos.should_take_profit(current):
                pnl = self.execute_sell(pos.symbol, current, pos)
                exits.append(
                    {
                        "reason": "take_profit",
                        "symbol": pos.symbol,
                        "price": current,
                        "pnl": pnl,
                    }
                )
        return exits

    # ── Reporting ─────────────────────────────────────────────

    @property
    def order_log(self) -> List[Dict[str, Any]]:
        """Full order history."""
        return list(self._order_log)

    def summary(
        self,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Return paper-trading session summary.

        Args:
            prices: Current prices for equity calculation.
        """
        port = self.portfolio.summary(prices)
        port["total_orders"] = len(self._order_log)
        port["fee_pct"] = self.fee_pct
        port["slippage_pct"] = self.slippage_pct
        return port

    # ── Internal helpers ──────────────────────────────────────

    def _record_order(
        self,
        side: str,
        symbol: str,
        price: float,
        quantity: float,
        fee: float,
    ) -> None:
        """Append an order record to the internal log."""
        self._trade_count += 1
        record = {
            "id": self._trade_count,
            "timestamp": time.time(),
            "side": side,
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "fee_per_unit": fee,
        }
        self._order_log.append(record)
        logger.info(
            "Order #%d: %s %s  qty=%.4f @ %.4f",
            self._trade_count,
            side,
            symbol,
            quantity,
            price,
        )
