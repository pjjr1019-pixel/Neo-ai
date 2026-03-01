"""
Trade Journal for NEO Hybrid AI.

Structured trade logging with ISO timestamps, strategy tags,
and P&L tracking.  Persists to JSONL for post-session analysis.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TradeRecord:
    """Single trade entry in the journal.

    Attributes:
        trade_id: Unique identifier.
        timestamp: ISO-8601 entry time.
        symbol: Trading pair.
        side: ``BUY`` or ``SELL``.
        price: Execution price.
        quantity: Size in base currency.
        pnl: Realised P&L (set on SELL).
        strategy: Strategy label.
        tags: Free-form tags.
    """

    def __init__(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
        strategy: str = "default",
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a trade record.

        Args:
            trade_id: Sequential ID.
            symbol: Trading pair.
            side: BUY or SELL.
            price: Fill price.
            quantity: Fill quantity.
            pnl: Realised P&L (optional).
            strategy: Strategy name.
            tags: Extra metadata.
        """
        self.trade_id = trade_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.unix_ts = time.time()
        self.symbol = symbol
        self.side = side
        self.price = price
        self.quantity = quantity
        self.pnl = pnl
        self.strategy = strategy
        self.tags = tags or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "strategy": self.strategy,
            "tags": self.tags,
        }

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict())


class TradeJournal:
    """Append-only trade journal with file persistence.

    Usage::

        journal = TradeJournal("trades.jsonl")
        journal.log_trade("BTC/USDT", "BUY", 50000, 0.01)
        journal.log_trade("BTC/USDT", "SELL", 51000, 0.01, pnl=10)
        print(journal.summary())
    """

    def __init__(
        self,
        file_path: str = "trade_journal.jsonl",
        auto_flush: bool = True,
    ) -> None:
        """Initialise the journal.

        Args:
            file_path: Path for the JSONL log.
            auto_flush: Write each trade to disk immediately.
        """
        self.file_path = file_path
        self.auto_flush = auto_flush
        self._records: List[TradeRecord] = []
        self._next_id = 1

    def log_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
        strategy: str = "default",
        tags: Optional[Dict[str, Any]] = None,
    ) -> TradeRecord:
        """Record a trade.

        Args:
            symbol: Trading pair.
            side: BUY or SELL.
            price: Fill price.
            quantity: Fill quantity.
            pnl: Realised P&L.
            strategy: Strategy label.
            tags: Metadata.

        Returns:
            The created TradeRecord.
        """
        record = TradeRecord(
            trade_id=self._next_id,
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            pnl=pnl,
            strategy=strategy,
            tags=tags,
        )
        self._records.append(record)
        self._next_id += 1

        if self.auto_flush:
            self._write_record(record)

        logger.info(
            "Trade #%d: %s %s %.4f @ %.2f  P&L=%s",
            record.trade_id,
            side,
            symbol,
            quantity,
            price,
            pnl,
        )
        return record

    # ── Queries ───────────────────────────────────────────────

    @property
    def trades(self) -> List[TradeRecord]:
        """All recorded trades."""
        return list(self._records)

    def trades_for_symbol(self, symbol: str) -> List[TradeRecord]:
        """Filter trades by symbol.

        Args:
            symbol: Trading pair to filter.
        """
        return [r for r in self._records if r.symbol == symbol]

    def total_pnl(self) -> float:
        """Sum of all realised P&L."""
        return sum(r.pnl for r in self._records if r.pnl is not None)

    def win_rate(self) -> float:
        """Percentage of winning trades (0-100).

        Only considers trades with ``pnl is not None``.
        """
        with_pnl = [r for r in self._records if r.pnl is not None]
        if not with_pnl:
            return 0.0
        winners = [r for r in with_pnl if r.pnl > 0]  # type: ignore[operator]
        wins = len(winners)
        return (wins / len(with_pnl)) * 100.0

    def summary(self) -> Dict[str, Any]:
        """Journal summary statistics."""
        return {
            "total_trades": len(self._records),
            "total_pnl": self.total_pnl(),
            "win_rate": self.win_rate(),
            "symbols": list(set(r.symbol for r in self._records)),
            "strategies": list(set(r.strategy for r in self._records)),
        }

    # ── Persistence ───────────────────────────────────────────

    def _write_record(self, record: TradeRecord) -> None:
        """Append a single record to the JSONL file."""
        try:
            with open(self.file_path, "a") as f:
                f.write(record.to_json() + "\n")
        except OSError as exc:
            logger.error("Failed to write trade journal: %s", exc)

    def flush(self) -> None:
        """Write all records to disk (for non-auto mode)."""
        try:
            with open(self.file_path, "w") as f:
                for record in self._records:
                    f.write(record.to_json() + "\n")
        except OSError as exc:
            logger.error("Failed to flush journal: %s", exc)

    def load(self) -> int:
        """Load existing records from the JSONL file.

        Returns:
            Number of records loaded.
        """
        if not os.path.isfile(self.file_path):
            return 0

        count = 0
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = TradeRecord(
                    trade_id=data["trade_id"],
                    symbol=data["symbol"],
                    side=data["side"],
                    price=data["price"],
                    quantity=data["quantity"],
                    pnl=data.get("pnl"),
                    strategy=data.get("strategy", "default"),
                    tags=data.get("tags", {}),
                )
                record.timestamp = data.get(
                    "timestamp",
                    record.timestamp,
                )
                self._records.append(record)
                count += 1

        if count:
            self._next_id = max(r.trade_id for r in self._records) + 1
        logger.info("Loaded %d trade records from %s", count, self.file_path)
        return count
