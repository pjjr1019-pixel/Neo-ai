"""
Dead Letter Queue for NEO Hybrid AI.

Captures failed messages/events that could not be
processed after the configured retry limit.  Supports
inspection, manual retry, and automatic purging of
old entries.
"""

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeadLetter:
    """A message that exhausted its retries.

    Attributes:
        payload: Original message payload.
        error: String description of the failure.
        source: Originating component or queue name.
        retry_count: Number of times delivery was tried.
        max_retries: Retry limit that was in effect.
        letter_id: Unique identifier for this entry.
        created_at: Unix timestamp of first failure.
        last_attempt: Unix timestamp of the most-recent
            retry attempt.
    """

    payload: Any
    error: str
    source: str = ""
    retry_count: int = 0
    max_retries: int = 3
    letter_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_attempt: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return asdict(self)


class DeadLetterQueue:
    """Thread-safe dead-letter queue.

    Messages land here after exceeding the retry limit
    in the upstream processing pipeline.

    Args:
        max_size: Maximum number of entries to hold.
            When exceeded, the oldest entry is evicted.
        default_max_retries: Default retry cap used when
            creating new entries.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        default_max_retries: int = 3,
    ) -> None:
        """Initialise the queue."""
        self._queue: List[DeadLetter] = []
        self._lock = threading.Lock()
        self._max_size = max_size
        self._default_retries = default_max_retries
        self._stats: Dict[str, int] = {
            "enqueued": 0,
            "retried": 0,
            "purged": 0,
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        payload: Any,
        error: str,
        source: str = "",
        retry_count: int = 0,
    ) -> DeadLetter:
        """Add a failed message to the queue.

        Args:
            payload: The original message.
            error: Human-readable error description.
            source: Originating component.
            retry_count: How many times it was already tried.

        Returns:
            The created :class:`DeadLetter`.
        """
        dl = DeadLetter(
            payload=payload,
            error=error,
            source=source,
            retry_count=retry_count,
            max_retries=self._default_retries,
        )
        with self._lock:
            if len(self._queue) >= self._max_size:
                self._queue.pop(0)
            self._queue.append(dl)
            self._stats["enqueued"] += 1
        logger.info(
            "Dead letter enqueued from %s: %s",
            source,
            error,
        )
        return dl

    def peek(self, count: int = 10) -> List[DeadLetter]:
        """Return the oldest *count* entries without removal.

        Args:
            count: Maximum entries to return.

        Returns:
            List of dead letters.
        """
        with self._lock:
            return list(self._queue[:count])

    def retry(
        self,
        letter_id: str,
        handler: Callable[[Any], None],
    ) -> bool:
        """Attempt to re-process a dead letter.

        On success the letter is removed from the queue.
        On failure it is updated with the new error and
        remains in the queue.

        Args:
            letter_id: ID of the letter to retry.
            handler: Callable that processes the payload.

        Returns:
            ``True`` if the handler succeeded.
        """
        with self._lock:
            target: Optional[DeadLetter] = None
            for dl in self._queue:
                if dl.letter_id == letter_id:
                    target = dl
                    break
            if target is None:
                logger.warning("Dead letter %s not found", letter_id)
                return False

        try:
            handler(target.payload)
        except Exception as exc:
            target.retry_count += 1
            target.last_attempt = time.time()
            target.error = str(exc)
            logger.warning(
                "Retry failed for %s: %s",
                letter_id,
                exc,
            )
            return False

        # Success — remove from queue.
        with self._lock:
            self._queue = [d for d in self._queue if d.letter_id != letter_id]
            self._stats["retried"] += 1
        logger.info("Dead letter %s retried OK", letter_id)
        return True

    def retry_all(self, handler: Callable[[Any], None]) -> Dict[str, int]:
        """Attempt to retry every entry in the queue.

        Args:
            handler: Processing callable.

        Returns:
            Dict with ``succeeded`` and ``failed`` counts.
        """
        ids = [dl.letter_id for dl in self.peek(self._max_size)]
        succeeded = 0
        failed = 0
        for lid in ids:
            if self.retry(lid, handler):
                succeeded += 1
            else:
                failed += 1
        return {"succeeded": succeeded, "failed": failed}

    def purge(self, older_than: Optional[float] = None) -> int:
        """Remove entries from the queue.

        Args:
            older_than: If given, only remove entries
                created before this Unix timestamp.
                If ``None``, remove *all*.

        Returns:
            Number of entries purged.
        """
        with self._lock:
            before = len(self._queue)
            if older_than is None:
                self._queue.clear()
            else:
                self._queue = [
                    d for d in self._queue if d.created_at >= older_than
                ]
            removed = before - len(self._queue)
            self._stats["purged"] += removed
        logger.info("Purged %d dead letters", removed)
        return removed

    def __len__(self) -> int:
        """Number of entries currently in the queue."""
        with self._lock:
            return len(self._queue)

    @property
    def stats(self) -> Dict[str, int]:
        """Return a copy of queue statistics."""
        with self._lock:
            return dict(self._stats)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the queue state.

        Returns:
            Dict with size, stats, and source breakdown.
        """
        with self._lock:
            entries = list(self._queue)
        sources: Dict[str, int] = {}
        for dl in entries:
            sources[dl.source] = sources.get(dl.source, 0) + 1
        return {
            "size": len(entries),
            "sources": sources,
            **self._stats,
        }
