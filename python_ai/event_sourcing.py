"""
Event Sourcing for NEO Hybrid AI.

Provides an append-only event store that records
every state-changing action as an immutable event.
Supports replaying events to reconstruct state,
filtering by aggregate, and both in-memory and
file-backed persistence.
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """An immutable domain event.

    Attributes:
        event_type: Descriptive type (e.g.
            ``order.placed``, ``strategy.updated``).
        aggregate_id: Identifier for the aggregate this
            event belongs to.
        data: Arbitrary event payload.
        event_id: Unique event identifier.
        version: Monotonic sequence number within the
            store.
        timestamp: Unix epoch when the event was created.
    """

    event_type: str
    aggregate_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the event to a plain dict."""
        return asdict(self)


class EventStore:
    """Append-only event store with optional file persistence.

    Thread-safe.  Each appended event is assigned a
    monotonically-increasing version number.

    Args:
        persist_path: If given, events are also written
            to this JSONL file and loaded on init.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        """Initialise the store."""
        self._events: List[Event] = []
        self._lock = threading.Lock()
        self._version = 0
        self._persist_path = persist_path
        self._subscribers: List[Callable[[Event], None]] = []

        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, event: Event) -> Event:
        """Append an event to the store.

        The event's version is overwritten with the
        next sequence number.

        Args:
            event: The event to store.

        Returns:
            The stored event (with assigned version).
        """
        with self._lock:
            self._version += 1
            stored = Event(
                event_type=event.event_type,
                aggregate_id=event.aggregate_id,
                data=event.data,
                event_id=event.event_id,
                version=self._version,
                timestamp=event.timestamp,
            )
            self._events.append(stored)
            if self._persist_path:
                self._persist(stored)

        for sub in self._subscribers:
            try:
                sub(stored)
            except Exception:
                logger.exception("Subscriber error")

        logger.debug(
            "Event appended: %s v%d",
            stored.event_type,
            stored.version,
        )
        return stored

    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[Event]:
        """Retrieve events, optionally filtered.

        Args:
            aggregate_id: Filter by aggregate.
            event_type: Filter by event type.

        Returns:
            List of matching events in version order.
        """
        with self._lock:
            result = list(self._events)
        if aggregate_id is not None:
            result = [e for e in result if e.aggregate_id == aggregate_id]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        return result

    def replay(
        self,
        handler: Callable[[Event], None],
        from_version: int = 0,
        aggregate_id: Optional[str] = None,
    ) -> int:
        """Replay stored events through a handler.

        Args:
            handler: Called once per event in order.
            from_version: Skip events before this
                version.
            aggregate_id: Optional aggregate filter.

        Returns:
            Number of events replayed.
        """
        events = self.get_events(aggregate_id=aggregate_id)
        count = 0
        for ev in events:
            if ev.version >= from_version:
                handler(ev)
                count += 1
        return count

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Register a callback for new events.

        Args:
            callback: Invoked on each ``append``.
        """
        self._subscribers.append(callback)

    @property
    def latest_version(self) -> int:
        """The highest version number in the store."""
        with self._lock:
            return self._version

    def __len__(self) -> int:
        """Total number of events stored."""
        with self._lock:
            return len(self._events)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the store.

        Returns:
            Dict with total events, latest version,
            and event-type counts.
        """
        with self._lock:
            events = list(self._events)
        type_counts: Dict[str, int] = {}
        for ev in events:
            type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1
        return {
            "total_events": len(events),
            "latest_version": self._version,
            "event_types": type_counts,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self, event: Event) -> None:
        """Append a single event to the JSONL file."""
        try:
            with open(
                self._persist_path,  # type: ignore[arg-type]
                "a",
                encoding="utf-8",
            ) as fh:
                fh.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            logger.exception("Failed to persist event")

    def _load(self, path: str) -> None:
        """Load events from a JSONL file."""
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    ev = Event(
                        event_type=d["event_type"],
                        aggregate_id=d["aggregate_id"],
                        data=d.get("data", {}),
                        event_id=d.get("event_id", ""),
                        version=d.get("version", 0),
                        timestamp=d.get("timestamp", 0.0),
                    )
                    self._events.append(ev)
                    if ev.version > self._version:
                        self._version = ev.version
            logger.info(
                "Loaded %d events from %s",
                len(self._events),
                path,
            )
        except OSError:
            logger.exception("Failed to load events")
