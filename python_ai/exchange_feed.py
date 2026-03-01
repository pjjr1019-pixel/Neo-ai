"""
Live Exchange Data Feed for NEO Hybrid AI.

Connects to cryptocurrency exchanges (Binance, Coinbase, Kraken)
via the ccxt library.  Implements the MarketDataFeed interface so
the AutonomousTradingLoop can consume real candles transparently.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

import ccxt

from python_ai.autonomous_trading_loop import MarketDataFeed

logger = logging.getLogger(__name__)

# Supported exchanges and their ccxt class names
SUPPORTED_EXCHANGES: Dict[str, str] = {
    "binance": "binance",
    "coinbase": "coinbasepro",
    "kraken": "kraken",
}


class LiveExchangeDataFeed(MarketDataFeed):
    """Live data feed backed by a ccxt exchange.

    Usage::

        feed = LiveExchangeDataFeed(exchange_id="binance")
        candle = feed.get_latest_candle("BTC/USDT")
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        timeframe: str = "1m",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        """Initialise exchange connection.

        Args:
            exchange_id: One of 'binance', 'coinbase', 'kraken'.
            timeframe: OHLCV bar period (ccxt format, e.g. '1m').
            api_key: Optional API key for authenticated endpoints.
            api_secret: Optional API secret.
        """
        if exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Unsupported exchange '{exchange_id}'. "
                f"Supported: {list(SUPPORTED_EXCHANGES.keys())}"
            )

        ccxt_cls_name = SUPPORTED_EXCHANGES[exchange_id]
        ccxt_cls = getattr(ccxt, ccxt_cls_name)

        config: Dict[str, Any] = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if api_secret:
            config["secret"] = api_secret

        self.exchange: ccxt.Exchange = ccxt_cls(config)
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self._connected = False
        self._subscriptions: Dict[
            str, List[Callable[[Dict[str, float]], None]]
        ] = {}
        self._poll_threads: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()

        # Verify connectivity
        try:
            self.exchange.load_markets()
            self._connected = True
            logger.info(
                "Connected to %s (%d markets)",
                exchange_id,
                len(self.exchange.markets),
            )
        except ccxt.BaseError as exc:
            logger.error(
                "Failed to connect to %s: %s",
                exchange_id,
                exc,
            )

    # ── MarketDataFeed interface ──────────────────────────────

    def get_latest_candle(
        self,
        symbol: str,
    ) -> Optional[Dict[str, float]]:
        """Fetch the most recent closed OHLCV candle.

        Args:
            symbol: Trading pair, e.g. 'BTC/USDT'.

        Returns:
            Dict with open/high/low/close/volume keys, or None.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                self.timeframe,
                limit=2,
            )
            if not ohlcv or len(ohlcv) < 2:
                return None

            # The *second to last* bar is the most recent *closed* bar;
            # the last bar may still be forming.
            bar = ohlcv[-2]
            return {
                "timestamp": float(bar[0]),
                "open": float(bar[1]),
                "high": float(bar[2]),
                "low": float(bar[3]),
                "close": float(bar[4]),
                "volume": float(bar[5]),
            }
        except ccxt.BaseError as exc:
            logger.warning(
                "fetch_ohlcv failed for %s: %s",
                symbol,
                exc,
            )
            return None

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[Dict[str, float]], None],
    ) -> None:
        """Subscribe to real-time candle updates via polling.

        Spawns a background thread that fetches the latest closed
        candle at the cadence of ``self.timeframe``.

        Args:
            symbol: Trading pair.
            callback: Invoked with each new candle dict.
        """
        if symbol not in self._subscriptions:
            self._subscriptions[symbol] = []
        self._subscriptions[symbol].append(callback)

        if symbol not in self._poll_threads:
            thread = threading.Thread(
                target=self._poll_loop,
                args=(symbol,),
                daemon=True,
                name=f"poll-{symbol}",
            )
            self._poll_threads[symbol] = thread
            thread.start()
            logger.info("Started polling thread for %s", symbol)

    def is_connected(self) -> bool:
        """Return True when exchange markets are loaded."""
        return self._connected

    # ── Bulk history (for back-fill) ──────────────────────────

    def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict[str, float]]:
        """Download a batch of historical OHLCV candles.

        Args:
            symbol: Trading pair.
            timeframe: Override the default timeframe.
            since: Start timestamp in milliseconds.
            limit: Max bars to return (exchange may cap this).

        Returns:
            List of candle dicts sorted oldest-first.
        """
        tf = timeframe or self.timeframe
        try:
            raw = self.exchange.fetch_ohlcv(
                symbol,
                tf,
                since=since,
                limit=limit,
            )
        except ccxt.BaseError as exc:
            logger.error(
                "fetch_historical_candles failed: %s",
                exc,
            )
            return []

        candles: List[Dict[str, float]] = []
        for bar in raw:
            candles.append(
                {
                    "timestamp": float(bar[0]),
                    "open": float(bar[1]),
                    "high": float(bar[2]),
                    "low": float(bar[3]),
                    "close": float(bar[4]),
                    "volume": float(bar[5]),
                }
            )
        return candles

    # ── Internal helpers ──────────────────────────────────────

    _TIMEFRAME_SECONDS: Dict[str, int] = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    def _poll_loop(self, symbol: str) -> None:
        """Background loop: fetch + dispatch candles."""
        interval = self._TIMEFRAME_SECONDS.get(self.timeframe, 60)
        last_ts: Optional[float] = None

        while not self._stop_event.is_set():
            candle = self.get_latest_candle(symbol)
            if candle and candle.get("timestamp") != last_ts:
                last_ts = candle["timestamp"]
                for cb in self._subscriptions.get(symbol, []):
                    try:
                        cb(candle)
                    except Exception:
                        logger.exception(
                            "Callback error for %s",
                            symbol,
                        )
            self._stop_event.wait(timeout=interval)

    def stop(self) -> None:
        """Signal all polling threads to stop."""
        self._stop_event.set()
        for t in self._poll_threads.values():
            t.join(timeout=5)
        logger.info("All polling threads stopped")


# ── Convenience singleton ─────────────────────────────────────

_feed: Optional[LiveExchangeDataFeed] = None


def get_exchange_feed(
    exchange_id: str = "binance",
    **kwargs: Any,
) -> LiveExchangeDataFeed:
    """Return (or create) a singleton LiveExchangeDataFeed."""
    global _feed
    if _feed is None:
        _feed = LiveExchangeDataFeed(
            exchange_id=exchange_id,
            **kwargs,
        )
    return _feed
