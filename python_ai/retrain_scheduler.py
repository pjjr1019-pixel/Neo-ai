"""
Auto-Retrain Scheduler for NEO Hybrid AI.

Wires the drift detection system to the ML model
retraining pipeline.  Supports both time-based
(periodic interval) and drift-triggered retraining.
Runs in a background thread.
"""

import logging
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """Background scheduler for model retraining.

    Combines two trigger modes:

    * **Time-based** — retrain every *interval_seconds*
      regardless of performance.
    * **Drift-triggered** — retrain whenever the
      *drift_checker* callback returns ``True``.

    The scheduler polls at *poll_seconds* intervals and
    invokes *retrain_fn* when either trigger fires.

    Args:
        retrain_fn: Callable that executes retraining.
            Must accept no positional arguments.  May
            return a dict with training results.
        drift_checker: Optional callable returning
            ``True`` when drift is detected.
        interval_seconds: Fixed retrain interval.
            ``0`` disables time-based triggering.
        poll_seconds: How often to check triggers.
        max_history: Maximum retrain history entries
            to keep.
    """

    def __init__(
        self,
        retrain_fn: Callable[[], Optional[Dict[str, Any]]],
        drift_checker: Optional[Callable[[], bool]] = None,
        interval_seconds: float = 3600.0,
        poll_seconds: float = 60.0,
        max_history: int = 100,
    ) -> None:
        """Initialise the scheduler."""
        self._retrain_fn = retrain_fn
        self._drift_checker = drift_checker
        self._interval = interval_seconds
        self._poll = poll_seconds
        self._max_history = max_history

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_retrain: float = 0.0
        self._retrain_count: int = 0
        self._history: List[Dict[str, Any]] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler thread.

        Raises:
            RuntimeError: If already running.
        """
        if self._running:
            raise RuntimeError("Scheduler already running")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="retrain-scheduler",
            daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info("Retrain scheduler started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the scheduler and wait for the thread.

        Args:
            timeout: Seconds to wait for the thread.
        """
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._running = False
        logger.info("Retrain scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler thread is active."""
        return self._running

    # ------------------------------------------------------------------
    # Manual trigger
    # ------------------------------------------------------------------

    def trigger_retrain(
        self, reason: str = "manual"
    ) -> Optional[Dict[str, Any]]:
        """Manually trigger a retrain cycle.

        Args:
            reason: Human-readable trigger reason.

        Returns:
            The retrain function's result dict, or ``None``
            on failure.
        """
        return self._do_retrain(reason)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Poll for triggers until stopped."""
        while not self._stop_event.is_set():
            try:
                self._check_triggers()
            except Exception:
                logger.exception("Error in retrain scheduler loop")
            self._stop_event.wait(timeout=self._poll)
        self._running = False

    def _check_triggers(self) -> None:
        """Evaluate time and drift triggers."""
        now = time.time()

        # Time-based trigger.
        if self._interval > 0:
            elapsed = now - self._last_retrain
            if elapsed >= self._interval:
                self._do_retrain("interval")
                return

        # Drift-based trigger.
        if self._drift_checker is not None:
            try:
                if self._drift_checker():
                    self._do_retrain("drift")
            except Exception:
                logger.exception("Drift checker failed")

    def _do_retrain(self, reason: str) -> Optional[Dict[str, Any]]:
        """Execute the retrain function.

        Args:
            reason: Why this retrain was triggered.

        Returns:
            Result dict from *retrain_fn*, or ``None``.
        """
        logger.info("Retraining triggered: %s", reason)
        try:
            result = self._retrain_fn()
        except Exception:
            logger.exception("Retrain failed")
            result = None

        with self._lock:
            self._last_retrain = time.time()
            self._retrain_count += 1
            entry = {
                "reason": reason,
                "timestamp": self._last_retrain,
                "success": result is not None,
                "result": result,
            }
            self._history.append(entry)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def retrain_count(self) -> int:
        """Total number of retrain cycles executed."""
        with self._lock:
            return self._retrain_count

    @property
    def last_retrain_time(self) -> float:
        """Unix timestamp of the latest retrain."""
        with self._lock:
            return self._last_retrain

    def history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent retrain history entries.

        Args:
            limit: Max entries to return.

        Returns:
            List of history dicts (newest last).
        """
        with self._lock:
            return list(self._history[-limit:])

    def summary(self) -> Dict[str, Any]:
        """Return scheduler status summary.

        Returns:
            Dict with running state, counts, and config.
        """
        with self._lock:
            return {
                "running": self._running,
                "retrain_count": self._retrain_count,
                "last_retrain": self._last_retrain,
                "interval_seconds": self._interval,
                "poll_seconds": self._poll,
                "drift_checker": (self._drift_checker is not None),
            }
