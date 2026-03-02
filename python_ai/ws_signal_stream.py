"""
WebSocket Signal Streaming Endpoint for NEO Hybrid AI.

Provides a FastAPI-compatible WebSocket endpoint that
broadcasts live trading signals to connected clients.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SignalBroadcaster:
    """Manages WebSocket connections and signal dispatch.

    Thread-safe async broadcaster that maintains a set
    of connected queues (one per client) and fans out
    new signals to all subscribers.
    """

    def __init__(self) -> None:
        """Initialise the broadcaster."""
        self._subscribers: Set[asyncio.Queue[str]] = set()
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 100

    async def subscribe(self) -> asyncio.Queue[str]:
        """Register a new subscriber.

        Returns:
            An ``asyncio.Queue`` that will receive
            JSON-encoded signal strings.
        """
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.add(queue)
        logger.info(
            "New subscriber — total: %d",
            len(self._subscribers),
        )
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber.

        Args:
            queue: The queue returned by ``subscribe()``.
        """
        self._subscribers.discard(queue)
        logger.info(
            "Subscriber removed — total: %d",
            len(self._subscribers),
        )

    async def broadcast(self, signal: Dict[str, Any]) -> int:
        """Send a signal to all subscribers.

        Args:
            signal: Signal payload dict.

        Returns:
            Number of subscribers that received it.
        """
        signal["timestamp"] = time.time()
        payload = json.dumps(signal)

        self._history.append(signal)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        delivered = 0
        dead: List[asyncio.Queue[str]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                dead.append(q)

        for q in dead:
            self._subscribers.discard(q)

        logger.info(
            "Broadcast signal to %d/%d subscribers",
            delivered,
            delivered + len(dead),
        )
        return delivered

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._subscribers)

    @property
    def recent_signals(self) -> List[Dict[str, Any]]:
        """Last N signals from history."""
        return list(self._history)


# ── global broadcaster singleton ──────────────────────

_broadcaster: Optional[SignalBroadcaster] = None


def get_signal_broadcaster() -> SignalBroadcaster:
    """Get or create the global broadcaster."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = SignalBroadcaster()
    return _broadcaster


async def websocket_signal_handler(
    websocket: Any,
) -> None:
    """Handle a WebSocket connection lifecycle.

    Suitable for use directly in a FastAPI WebSocket
    route::

        @app.websocket("/ws/signals")
        async def ws_signals(ws: WebSocket):
            await ws.accept()
            await websocket_signal_handler(ws)

    Args:
        websocket: A Starlette/FastAPI ``WebSocket``.
    """
    broadcaster = get_signal_broadcaster()
    queue = await broadcaster.subscribe()
    try:
        while True:
            payload = await queue.get()
            await websocket.send_text(payload)
    except Exception:
        logger.info("WebSocket client disconnected")
    finally:
        await broadcaster.unsubscribe(queue)
