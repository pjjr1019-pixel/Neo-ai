"""
Redis Streams Event Ingestion for NEO Hybrid AI.

Provides a consumer that reads events from Redis
Streams using consumer groups, processes them via a
pluggable handler, and manages acknowledgements.

Falls back to an in-memory stub when Redis is
unavailable, enabling testing without infrastructure.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """A single event read from a stream.

    Attributes:
        event_id: Stream entry ID (e.g. ``"1-0"``).
        stream: Stream name.
        data: Parsed event payload.
        timestamp: When the event was received.
    """

    event_id: str
    stream: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class InMemoryStream:
    """In-memory Redis Stream stand-in for testing.

    Mimics ``XADD`` / ``XREADGROUP`` / ``XACK``
    semantics without needing a real Redis instance.
    """

    def __init__(self) -> None:
        """Initialise the in-memory stream."""
        self._streams: Dict[str, List[Dict[str, Any]]] = {}
        self._groups: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def xadd(self, stream: str, data: Dict[str, Any]) -> str:
        """Append an event to a stream.

        Args:
            stream: Stream name.
            data: Event payload.

        Returns:
            Generated event ID.
        """
        with self._lock:
            self._counter += 1
            eid = f"{self._counter}-0"
            self._streams.setdefault(stream, []).append(
                {"id": eid, "data": data}
            )
            return eid

    def create_group(self, stream: str, group: str) -> None:
        """Create a consumer group.

        Args:
            stream: Stream to bind to.
            group: Consumer group name.
        """
        key = f"{stream}:{group}"
        with self._lock:
            self._groups.setdefault(key, {"cursor": 0})

    def xreadgroup(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Read new entries for a consumer group.

        Args:
            stream: Stream name.
            group: Consumer group.
            consumer: Consumer name (unused in stub).
            count: Max entries to return.

        Returns:
            List of ``{"id": ..., "data": ...}`` dicts.
        """
        key = f"{stream}:{group}"
        with self._lock:
            entries = self._streams.get(stream, [])
            grp = self._groups.get(key, {"cursor": 0})
            cursor = grp["cursor"]
            batch = entries[cursor : cursor + count]
            return batch

    def xack(self, stream: str, group: str, event_id: str) -> None:
        """Acknowledge an event.

        Advances the cursor past the acknowledged entry.

        Args:
            stream: Stream name.
            group: Consumer group.
            event_id: The event ID to acknowledge.
        """
        key = f"{stream}:{group}"
        with self._lock:
            grp = self._groups.get(key)
            if grp is None:
                return
            entries = self._streams.get(stream, [])
            for i, e in enumerate(entries):
                if e["id"] == event_id:
                    grp["cursor"] = max(grp["cursor"], i + 1)
                    break


# Type alias for event handlers.
EventHandler = Callable[[StreamEvent], None]


class StreamConsumer:
    """Consumes events from a stream with handlers.

    Runs a background polling loop that reads events,
    invokes the handler, and acknowledges them.

    Args:
        backend: Redis-like backend
            (``InMemoryStream`` or real Redis client).
        stream: Stream name to consume from.
        group: Consumer group name.
        consumer: This consumer's name.
        handler: Callback for processing events.
        poll_interval: Seconds between polls.
        batch_size: Max events per read.
    """

    def __init__(
        self,
        backend: Any,
        stream: str = "neo:events",
        group: str = "neo-group",
        consumer: str = "worker-1",
        handler: Optional[EventHandler] = None,
        poll_interval: float = 1.0,
        batch_size: int = 10,
    ) -> None:
        """Initialise the consumer."""
        self._backend = backend
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._handler = handler or _default_handler
        self._poll = poll_interval
        self._batch = batch_size

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._processed = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background consumer thread.

        Creates the consumer group if it doesn't exist.

        Raises:
            RuntimeError: If already running.
        """
        if self._running:
            raise RuntimeError("Consumer already running")
        try:
            self._backend.create_group(self._stream, self._group)
        except Exception:
            logger.debug(
                "Group %s may already exist",
                self._group,
            )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"stream-{self._stream}",
            daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info(
            "Stream consumer started: %s/%s",
            self._stream,
            self._group,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the consumer.

        Args:
            timeout: Seconds to wait for thread exit.
        """
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._running = False
        logger.info("Stream consumer stopped")

    @property
    def is_running(self) -> bool:
        """Whether the consumer loop is active."""
        return self._running

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Poll and process events until stopped."""
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("Stream consumer error")
            self._stop.wait(timeout=self._poll)
        self._running = False

    def _poll_once(self) -> None:
        """Read and process one batch of events."""
        entries = self._backend.xreadgroup(
            self._stream,
            self._group,
            self._consumer,
            count=self._batch,
        )
        for entry in entries:
            eid = entry.get("id", "")
            data = entry.get("data", {})
            event = StreamEvent(
                event_id=eid,
                stream=self._stream,
                data=data,
            )
            try:
                self._handler(event)
                self._backend.xack(self._stream, self._group, eid)
                self._processed += 1
            except Exception:
                logger.exception("Handler failed for %s", eid)
                self._errors += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def processed_count(self) -> int:
        """Total events successfully processed."""
        return self._processed

    @property
    def error_count(self) -> int:
        """Total handler errors."""
        return self._errors

    def summary(self) -> Dict[str, Any]:
        """Return consumer stats.

        Returns:
            Dict with stream info and counters.
        """
        return {
            "stream": self._stream,
            "group": self._group,
            "consumer": self._consumer,
            "running": self._running,
            "processed": self._processed,
            "errors": self._errors,
        }


def _default_handler(event: StreamEvent) -> None:
    """No-op default handler that logs the event.

    Args:
        event: Received stream event.
    """
    logger.debug("Event %s: %s", event.event_id, event.data)
