"""
Bulkhead Pattern for NEO Hybrid AI.

Isolates failures by partitioning resources into
independent pools.  Each pool has a bounded number
of concurrent executions; excess callers are rejected
immediately rather than queuing indefinitely.
"""

import functools
import logging
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BulkheadFullError(Exception):
    """Raised when a bulkhead pool has no capacity."""


class Bulkhead:
    """Thread-pool-style bulkhead with bounded concurrency.

    Limits the number of simultaneous executions to
    *max_concurrent*.  When the limit is reached,
    additional callers immediately receive a
    :class:`BulkheadFullError`.

    Args:
        name: Human-readable pool name (for logging).
        max_concurrent: Maximum parallel executions.
    """

    def __init__(
        self, name: str = "default", max_concurrent: int = 10
    ) -> None:
        """Initialise the bulkhead."""
        self.name = name
        self._max = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._active = 0
        self._stats: Dict[str, int] = {
            "accepted": 0,
            "rejected": 0,
            "completed": 0,
            "failed": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *func* within the bulkhead.

        Args:
            func: Callable to run.
            *args: Positional arguments for *func*.
            **kwargs: Keyword arguments for *func*.

        Returns:
            The return value of *func*.

        Raises:
            BulkheadFullError: If the pool is at capacity.
        """
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            with self._lock:
                self._stats["rejected"] += 1
            logger.warning(
                "Bulkhead '%s' full (%d/%d)",
                self.name,
                self._active,
                self._max,
            )
            raise BulkheadFullError(
                f"Bulkhead '{self.name}' at capacity " f"({self._max})"
            )

        with self._lock:
            self._active += 1
            self._stats["accepted"] += 1

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._stats["completed"] += 1
            return result  # type: ignore[return-value]
        except Exception:
            with self._lock:
                self._stats["failed"] += 1
            raise
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as a decorator.

        Example::

            bh = Bulkhead("exchange", max_concurrent=5)

            @bh
            def fetch_price(symbol: str) -> float:
                ...
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Execute within bulkhead."""
            return self.execute(func, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    @property
    def active(self) -> int:
        """Number of currently-running executions."""
        with self._lock:
            return self._active

    @property
    def available(self) -> int:
        """Remaining capacity."""
        with self._lock:
            return self._max - self._active

    @property
    def stats(self) -> Dict[str, int]:
        """Return a copy of bulkhead statistics."""
        with self._lock:
            return dict(self._stats)

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary.

        Returns:
            Dict with name, capacity, active threads,
            and statistics.
        """
        with self._lock:
            return {
                "name": self.name,
                "max_concurrent": self._max,
                "active": self._active,
                "available": self._max - self._active,
                **dict(self._stats),
            }


class BulkheadRegistry:
    """Registry of named bulkheads for service isolation.

    Provides a single point of access for all bulkheads
    in the system.

    Example::

        reg = BulkheadRegistry()
        reg.register("exchange", max_concurrent=5)
        reg.register("database", max_concurrent=20)
        reg.get("exchange").execute(fetch_price, "BTC")
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._bulkheads: Dict[str, Bulkhead] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        max_concurrent: int = 10,
    ) -> Bulkhead:
        """Create and register a new bulkhead.

        If a bulkhead with *name* already exists, it is
        returned unchanged.

        Args:
            name: Unique pool name.
            max_concurrent: Concurrency limit.

        Returns:
            The (possibly pre-existing) bulkhead.
        """
        with self._lock:
            if name not in self._bulkheads:
                self._bulkheads[name] = Bulkhead(name, max_concurrent)
            return self._bulkheads[name]

    def get(self, name: str) -> Optional[Bulkhead]:
        """Retrieve a registered bulkhead by name.

        Args:
            name: Pool name.

        Returns:
            The bulkhead, or ``None`` if not found.
        """
        with self._lock:
            return self._bulkheads.get(name)

    def summary(self) -> Dict[str, Any]:
        """Return summaries of all registered bulkheads.

        Returns:
            Dict mapping names to their summaries.
        """
        with self._lock:
            return {name: bh.summary() for name, bh in self._bulkheads.items()}
