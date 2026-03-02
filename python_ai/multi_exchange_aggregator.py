"""
Multi-Exchange Price Aggregator for NEO Hybrid AI.

Aggregates bid/ask/last prices from multiple exchanges
and computes consensus pricing (VWAP, median, best).
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExchangeQuote:
    """A price quote from a single exchange.

    Attributes:
        exchange: Exchange identifier.
        symbol: Trading pair (e.g. ``BTC/USDT``).
        bid: Best bid price.
        ask: Best ask price.
        last: Last traded price.
        volume_24h: 24-hour volume.
        timestamp: Quote timestamp (Unix seconds).
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        bid: float = 0.0,
        ask: float = 0.0,
        last: float = 0.0,
        volume_24h: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Initialise a quote."""
        self.exchange = exchange
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.last = last
        self.volume_24h = volume_24h
        self.timestamp = timestamp or time.time()

    @property
    def mid(self) -> float:
        """Mid-point between bid and ask."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        """Bid-ask spread as a fraction of mid."""
        m = self.mid
        if m <= 0:
            return 0.0
        return (self.ask - self.bid) / m

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "spread": self.spread,
            "volume_24h": self.volume_24h,
            "timestamp": self.timestamp,
        }


class MultiExchangeAggregator:
    """Aggregate prices across exchange feeds.

    Stores the latest quote per exchange and computes
    consensus metrics on demand.

    Args:
        stale_seconds: Quotes older than this are
            excluded from aggregation.
    """

    def __init__(self, stale_seconds: float = 30.0) -> None:
        """Initialise the aggregator."""
        self._quotes: Dict[str, ExchangeQuote] = {}
        self._stale = stale_seconds

    def update(self, quote: ExchangeQuote) -> None:
        """Ingest a new quote.

        Args:
            quote: Latest quote from an exchange.
        """
        key = f"{quote.exchange}:{quote.symbol}"
        self._quotes[key] = quote

    def _fresh_quotes(self, symbol: str) -> List[ExchangeQuote]:
        """Return non-stale quotes for a symbol."""
        now = time.time()
        return [
            q
            for q in self._quotes.values()
            if q.symbol == symbol and (now - q.timestamp) < self._stale
        ]

    # ── aggregation methods ───────────────────────────

    def vwap(self, symbol: str) -> float:
        """Volume-weighted average price.

        Args:
            symbol: Trading pair.

        Returns:
            VWAP across exchanges, or 0.0 if no data.
        """
        quotes = self._fresh_quotes(symbol)
        if not quotes:
            return 0.0
        total_vol = sum(q.volume_24h for q in quotes)
        if total_vol <= 0:
            return self.median_price(symbol)
        return sum(q.last * q.volume_24h for q in quotes) / total_vol

    def median_price(self, symbol: str) -> float:
        """Median last price across exchanges.

        Args:
            symbol: Trading pair.

        Returns:
            Median price, or 0.0 if no data.
        """
        quotes = self._fresh_quotes(symbol)
        if not quotes:
            return 0.0
        prices = sorted(q.last for q in quotes)
        n = len(prices)
        if n % 2 == 1:
            return prices[n // 2]
        return (prices[n // 2 - 1] + prices[n // 2]) / 2.0

    def best_bid(self, symbol: str) -> Optional[float]:
        """Highest bid across exchanges.

        Args:
            symbol: Trading pair.

        Returns:
            Best bid price, or ``None``.
        """
        quotes = self._fresh_quotes(symbol)
        bids = [q.bid for q in quotes if q.bid > 0]
        return max(bids) if bids else None

    def best_ask(self, symbol: str) -> Optional[float]:
        """Lowest ask across exchanges.

        Args:
            symbol: Trading pair.

        Returns:
            Best ask price, or ``None``.
        """
        quotes = self._fresh_quotes(symbol)
        asks = [q.ask for q in quotes if q.ask > 0]
        return min(asks) if asks else None

    def spread_summary(self, symbol: str) -> Dict[str, Any]:
        """Spread statistics across exchanges.

        Args:
            symbol: Trading pair.

        Returns:
            Dict with min/max/mean spread and per-exchange
            detail.
        """
        quotes = self._fresh_quotes(symbol)
        if not quotes:
            return {"exchanges": 0}
        spreads = [q.spread for q in quotes]
        return {
            "exchanges": len(quotes),
            "min_spread": min(spreads),
            "max_spread": max(spreads),
            "mean_spread": sum(spreads) / len(spreads),
            "per_exchange": {q.exchange: q.spread for q in quotes},
        }

    def all_quotes(self, symbol: str) -> List[Dict[str, Any]]:
        """Return all fresh quotes for a symbol.

        Args:
            symbol: Trading pair.

        Returns:
            List of serialised quotes.
        """
        return [q.to_dict() for q in self._fresh_quotes(symbol)]
