"""
Position & Portfolio Tracker for NEO Hybrid AI.

Tracks open positions, unrealised P&L, and portfolio-level
statistics.  Used by the paper-trading engine and (in future)
by the live trading loop.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single open or closed position.

    Attributes:
        symbol: Trading pair, e.g. ``BTC/USDT``.
        side: ``long`` or ``short``.
        entry_price: Average entry price.
        quantity: Position size in base currency.
        entry_time: Unix timestamp of entry.
        stop_loss: Optional stop-loss price.
        take_profit: Optional take-profit price.
        exit_price: Set when the position is closed.
        exit_time: Set when the position is closed.
        is_open: Whether the position is still active.
        metadata: Free-form tags (strategy name, etc.).
    """

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: float = field(default_factory=time.time)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    is_open: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Computed properties ───────────────────────────────────

    @property
    def notional(self) -> float:
        """Entry notional value (entry_price * quantity)."""
        return self.entry_price * self.quantity

    def unrealised_pnl(self, current_price: float) -> float:
        """Unrealised P&L at *current_price*.

        Args:
            current_price: Latest market price.

        Returns:
            Profit (positive) or loss (negative) in quote currency.
        """
        if not self.is_open:
            return 0.0
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (current_price - self.entry_price) * self.quantity

    def realised_pnl(self) -> float:
        """Realised P&L (valid only when closed)."""
        if self.is_open or self.exit_price is None:
            return 0.0
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (self.exit_price - self.entry_price) * self.quantity

    def should_stop_out(self, current_price: float) -> bool:
        """Return True if *current_price* hits the stop-loss.

        Args:
            current_price: Latest market price.
        """
        if self.stop_loss is None or not self.is_open:
            return False
        if self.side == "long":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        """Return True if *current_price* hits the take-profit.

        Args:
            current_price: Latest market price.
        """
        if self.take_profit is None or not self.is_open:
            return False
        if self.side == "long":
            return current_price >= self.take_profit
        return current_price <= self.take_profit

    def close(self, exit_price: float) -> float:
        """Close the position at *exit_price* and return realised P&L.

        Args:
            exit_price: Price at which the position is exited.

        Returns:
            Realised P&L in quote currency.
        """
        self.exit_price = exit_price
        self.exit_time = time.time()
        self.is_open = False
        pnl = self.realised_pnl()
        logger.info(
            "Closed %s %s @ %.4f  (entry %.4f)  P&L=%.4f",
            self.side,
            self.symbol,
            exit_price,
            self.entry_price,
            pnl,
        )
        return pnl

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the position."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "is_open": self.is_open,
            "realised_pnl": self.realised_pnl(),
        }


class PortfolioTracker:
    """Aggregate portfolio state across all positions.

    Tracks equity curve, open/closed positions, and overall P&L.

    Attributes:
        initial_capital: Starting balance.
        balance: Current cash balance.
    """

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        """Initialise the portfolio tracker.

        Args:
            initial_capital: Starting cash balance.
        """
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self._positions: List[Position] = []
        self._equity_snapshots: List[Dict[str, float]] = []

    # ── Position management ───────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Position:
        """Open a new position and deduct from balance.

        Args:
            symbol: Trading pair.
            side: ``long`` or ``short``.
            price: Entry price.
            quantity: Size in base currency.
            stop_loss: Optional stop-loss level.
            take_profit: Optional take-profit level.
            metadata: Optional tags.

        Returns:
            The newly created Position.
        """
        cost = price * quantity
        if cost > self.balance:
            raise ValueError(
                f"Insufficient balance: need {cost:.2f}, "
                f"have {self.balance:.2f}"
            )

        self.balance -= cost
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {},
        )
        self._positions.append(pos)
        logger.info(
            "Opened %s %s  qty=%.4f @ %.4f  (balance=%.2f)",
            side,
            symbol,
            quantity,
            price,
            self.balance,
        )
        return pos

    def close_position(self, position: Position, price: float) -> float:
        """Close *position* at *price*, crediting P&L to balance.

        Args:
            position: The Position to close.
            price: Exit price.

        Returns:
            Realised P&L.
        """
        pnl = position.close(price)
        self.balance += position.entry_price * position.quantity + pnl
        return pnl

    # ── Queries ───────────────────────────────────────────────

    @property
    def open_positions(self) -> List[Position]:
        """All currently open positions."""
        return [p for p in self._positions if p.is_open]

    @property
    def closed_positions(self) -> List[Position]:
        """All closed positions."""
        return [p for p in self._positions if not p.is_open]

    def equity(self, prices: Dict[str, float]) -> float:
        """Total equity (balance + unrealised P&L).

        Args:
            prices: Dict mapping symbol → current price.
        """
        unrealised = sum(
            p.unrealised_pnl(prices.get(p.symbol, p.entry_price))
            for p in self.open_positions
        )
        # Value of open positions at entry
        open_notional = sum(p.notional for p in self.open_positions)
        return self.balance + open_notional + unrealised

    def snapshot(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Record and return an equity snapshot.

        Args:
            prices: Current market prices by symbol.

        Returns:
            Snapshot dict with timestamp, balance, equity.
        """
        snap = {
            "timestamp": time.time(),
            "balance": self.balance,
            "equity": self.equity(prices),
            "open_count": float(len(self.open_positions)),
        }
        self._equity_snapshots.append(snap)
        return snap

    def total_realised_pnl(self) -> float:
        """Sum of all realised P&L from closed positions."""
        return sum(p.realised_pnl() for p in self.closed_positions)

    def win_rate(self) -> float:
        """Percentage of profitable closed trades (0-100)."""
        closed = self.closed_positions
        if not closed:
            return 0.0
        wins = sum(1 for p in closed if p.realised_pnl() > 0)
        return (wins / len(closed)) * 100.0

    def summary(
        self,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Portfolio summary dict.

        Args:
            prices: Current prices for equity calculation.
        """
        prices = prices or {}
        return {
            "initial_capital": self.initial_capital,
            "balance": self.balance,
            "equity": self.equity(prices),
            "total_realised_pnl": self.total_realised_pnl(),
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "win_rate": self.win_rate(),
        }
