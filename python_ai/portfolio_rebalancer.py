"""
Portfolio Rebalancer for NEO Hybrid AI.

Automatically rebalances portfolio allocations when drift
exceeds configurable thresholds.  Supports both threshold-
and calendar-based triggers.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RebalanceOrder:
    """Represents a single rebalance trade instruction.

    Attributes:
        symbol: Asset symbol.
        side: ``"buy"`` or ``"sell"``.
        amount: Absolute amount to trade.
        reason: Human-readable reason.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        amount: float,
        reason: str = "",
    ) -> None:
        """Initialise a rebalance order."""
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "amount": round(self.amount, 8),
            "reason": self.reason,
        }


class PortfolioRebalancer:
    """Compute rebalance trades based on target weights.

    Supports two rebalance triggers:

    * **Threshold**: fires when any asset drifts by more
      than *threshold* from its target weight.
    * **Calendar**: fires when *interval_seconds* have
      elapsed since the last rebalance.

    Args:
        target_weights: Desired allocation per asset
            (values should sum to ~1.0).
        threshold: Maximum allowed weight drift
            (default 0.05 = 5%).
        interval_seconds: Minimum seconds between
            calendar-based rebalances (0 = disable).
        min_trade_value: Minimum trade value to emit.
    """

    def __init__(
        self,
        target_weights: Dict[str, float],
        threshold: float = 0.05,
        interval_seconds: float = 0.0,
        min_trade_value: float = 10.0,
    ) -> None:
        """Initialise rebalancer with targets."""
        self._targets = dict(target_weights)
        self._threshold = threshold
        self._interval = interval_seconds
        self._min_trade = min_trade_value
        self._last_rebalance: Optional[float] = None
        self._history: List[Dict[str, Any]] = []

    # ── drift calculation ─────────────────────────────

    def compute_drift(
        self,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute per-asset drift from target.

        Args:
            current_weights: Actual allocation weights.

        Returns:
            Dict mapping symbol → signed drift value.
        """
        drift: Dict[str, float] = {}
        for symbol, target in self._targets.items():
            actual = current_weights.get(symbol, 0.0)
            drift[symbol] = actual - target
        return drift

    def needs_rebalance(
        self,
        current_weights: Dict[str, float],
    ) -> bool:
        """Check whether rebalancing is needed.

        Args:
            current_weights: Actual allocation weights.

        Returns:
            ``True`` if threshold or calendar trigger fired.
        """
        drift = self.compute_drift(current_weights)
        if any(abs(d) > self._threshold for d in drift.values()):
            return True
        if self._interval > 0 and self._last_rebalance is not None:
            elapsed = time.time() - self._last_rebalance
            if elapsed >= self._interval:
                return True
        return False

    # ── order generation ──────────────────────────────

    def generate_orders(
        self,
        current_weights: Dict[str, float],
        portfolio_value: float = 1.0,
    ) -> List[RebalanceOrder]:
        """Produce rebalance trade orders.

        Args:
            current_weights: Current allocation weights.
            portfolio_value: Total portfolio value for
                sizing orders in absolute terms.

        Returns:
            List of ``RebalanceOrder`` instances.
        """
        drift = self.compute_drift(current_weights)
        orders: List[RebalanceOrder] = []

        for symbol, d in drift.items():
            trade_value = abs(d) * portfolio_value
            if trade_value < self._min_trade:
                continue
            side = "sell" if d > 0 else "buy"
            orders.append(
                RebalanceOrder(
                    symbol=symbol,
                    side=side,
                    amount=trade_value,
                    reason=(
                        f"Drift {d:+.4f} exceeds "
                        f"threshold {self._threshold}"
                    ),
                )
            )

        if orders:
            self._last_rebalance = time.time()
            self._history.append(
                {
                    "time": self._last_rebalance,
                    "orders": [o.to_dict() for o in orders],
                }
            )
            logger.info("Generated %d rebalance orders", len(orders))
        return orders

    # ── history & config ──────────────────────────────

    def update_targets(self, new_targets: Dict[str, float]) -> None:
        """Replace target weights.

        Args:
            new_targets: New allocation targets.
        """
        self._targets = dict(new_targets)
        logger.info("Updated target weights: %s", new_targets)

    def history(self) -> List[Dict[str, Any]]:
        """Return rebalance history."""
        return list(self._history)

    def summary(self) -> Dict[str, Any]:
        """Return configuration and stats summary."""
        return {
            "target_weights": self._targets,
            "threshold": self._threshold,
            "interval_seconds": self._interval,
            "rebalance_count": len(self._history),
            "last_rebalance": self._last_rebalance,
        }
