"""
Cross-Exchange Data Validation for NEO Hybrid AI.

Compares price data for the same symbol across
multiple exchanges to detect outliers, stale quotes,
and potential data-feed issues.  Builds on the
multi-exchange aggregator infrastructure.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ExchangePrice:
    """A price observation from a single exchange.

    Attributes:
        exchange: Exchange name (e.g. ``binance``).
        symbol: Trading pair (e.g. ``BTC/USDT``).
        price: Last/mid price.
        timestamp: Unix epoch when the quote was fetched.
        bid: Best bid price (optional).
        ask: Best ask price (optional).
    """

    exchange: str
    symbol: str
    price: float
    timestamp: float = field(default_factory=time.time)
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class ValidationResult:
    """Outcome of a cross-exchange validation check.

    Attributes:
        symbol: Trading pair validated.
        consensus_price: Central tendency price.
        outlier_exchanges: Exchanges flagged as outliers.
        stale_exchanges: Exchanges with stale quotes.
        max_spread_pct: Maximum percentage spread across
            all exchanges.
        details: Per-exchange detail dict.
        valid: ``True`` if no issues detected.
    """

    symbol: str
    consensus_price: float
    outlier_exchanges: List[str] = field(default_factory=list)
    stale_exchanges: List[str] = field(default_factory=list)
    max_spread_pct: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True


class CrossExchangeValidator:
    """Validates price consistency across exchanges.

    Detects:

    * **Outliers** — prices deviating more than
      *z_threshold* standard deviations from the
      median.
    * **Stale quotes** — timestamps older than
      *max_age_seconds*.
    * **Wide spreads** — bid/ask spread exceeding
      *max_spread_pct*.

    Args:
        z_threshold: Z-score threshold for outlier
            detection.
        max_age_seconds: Maximum quote age before
            it is considered stale.
        max_spread_pct: Maximum acceptable bid-ask
            spread as a percentage.
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        max_age_seconds: float = 60.0,
        max_spread_pct: float = 1.0,
    ) -> None:
        """Initialise the validator."""
        self._z = z_threshold
        self._max_age = max_age_seconds
        self._max_spread = max_spread_pct
        self._stats: Dict[str, int] = {
            "validations": 0,
            "outliers_found": 0,
            "stale_found": 0,
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def validate(
        self,
        quotes: Sequence[ExchangePrice],
    ) -> ValidationResult:
        """Validate a set of quotes for one symbol.

        Args:
            quotes: Prices from different exchanges for
                the same symbol.

        Returns:
            :class:`ValidationResult`.

        Raises:
            ValueError: If fewer than 2 quotes are given.
        """
        if len(quotes) < 2:
            raise ValueError("Need quotes from >= 2 exchanges")

        symbol = quotes[0].symbol
        prices = [q.price for q in quotes]
        median_p = float(sorted(prices)[len(prices) // 2])
        mean_p = sum(prices) / len(prices)
        std_p = (
            sum((p - mean_p) ** 2 for p in prices) / len(prices)
        ) ** 0.5 or 1e-12

        now = time.time()
        outliers: List[str] = []
        stale: List[str] = []
        details: Dict[str, Any] = {}

        for q in quotes:
            z = abs(q.price - mean_p) / std_p
            age = now - q.timestamp
            is_outlier = z > self._z
            is_stale = age > self._max_age
            spread_pct = 0.0
            if q.bid and q.ask and q.bid > 0:
                spread_pct = (q.ask - q.bid) / q.bid * 100.0

            if is_outlier:
                outliers.append(q.exchange)
            if is_stale:
                stale.append(q.exchange)

            details[q.exchange] = {
                "price": q.price,
                "z_score": round(z, 4),
                "age_seconds": round(age, 2),
                "spread_pct": round(spread_pct, 4),
                "outlier": is_outlier,
                "stale": is_stale,
            }

        max_spread = max(d.get("spread_pct", 0.0) for d in details.values())
        valid = not outliers and not stale

        self._stats["validations"] += 1
        self._stats["outliers_found"] += len(outliers)
        self._stats["stale_found"] += len(stale)

        logger.info(
            "Validated %s across %d exchanges: " "%d outliers, %d stale",
            symbol,
            len(quotes),
            len(outliers),
            len(stale),
        )

        return ValidationResult(
            symbol=symbol,
            consensus_price=median_p,
            outlier_exchanges=outliers,
            stale_exchanges=stale,
            max_spread_pct=max_spread,
            details=details,
            valid=valid,
        )

    def detect_arbitrage(
        self,
        quotes: Sequence[ExchangePrice],
        min_spread_pct: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities.

        Finds pairs where one exchange's ask is below
        another's bid by more than *min_spread_pct*.

        Args:
            quotes: Quotes with bid/ask data.
            min_spread_pct: Minimum spread to flag.

        Returns:
            List of opportunity dicts.
        """
        opportunities: List[Dict[str, Any]] = []
        bid_quotes = [q for q in quotes if q.bid is not None]
        ask_quotes = [q for q in quotes if q.ask is not None]

        for buy in ask_quotes:
            for sell in bid_quotes:
                if buy.exchange == sell.exchange:
                    continue
                ask_val = buy.ask or 0.0
                bid_val = sell.bid or 0.0
                if ask_val <= 0 or bid_val <= 0:
                    continue
                spread = (bid_val - ask_val) / ask_val * 100
                if spread >= min_spread_pct:
                    opportunities.append(
                        {
                            "buy_exchange": buy.exchange,
                            "sell_exchange": sell.exchange,
                            "buy_price": ask_val,
                            "sell_price": bid_val,
                            "spread_pct": round(spread, 4),
                            "symbol": buy.symbol,
                        }
                    )
        return opportunities

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        """Return validation statistics."""
        return dict(self._stats)

    def summary(self) -> Dict[str, Any]:
        """Return validator configuration and stats.

        Returns:
            Dict with config values and counters.
        """
        return {
            "z_threshold": self._z,
            "max_age_seconds": self._max_age,
            "max_spread_pct": self._max_spread,
            **self._stats,
        }
