"""
Signal Alert System for NEO Hybrid AI.

Fires structured log alerts when trading signals are generated.
Supports pluggable alert handlers (console, file, future webhook).
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel:
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert:
    """Immutable alert record.

    Attributes:
        level: Severity (INFO / WARNING / CRITICAL).
        category: Alert category (signal, risk, drift, etc.).
        message: Human-readable description.
        data: Structured payload.
        timestamp: Unix timestamp.
    """

    def __init__(
        self,
        level: str,
        category: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create an alert.

        Args:
            level: Severity level.
            category: Category tag.
            message: Description.
            data: Extra payload.
        """
        self.level = level
        self.category = category
        self.message = message
        self.data = data or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict."""
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict())


# Type alias for alert handler callbacks
AlertHandler = Callable[[Alert], None]


class AlertDispatcher:
    """Central alert router.

    Register handlers and fire alerts.  All alerts are also
    appended to an internal history buffer.

    Usage::

        dispatcher = AlertDispatcher()
        dispatcher.add_handler(console_handler)
        dispatcher.fire(
            AlertLevel.INFO,
            "signal",
            "BUY signal for BTC/USDT",
            {"confidence": 0.85},
        )
    """

    def __init__(self, max_history: int = 500) -> None:
        """Initialise dispatcher.

        Args:
            max_history: Max alerts retained in history.
        """
        self._handlers: List[AlertHandler] = []
        self._history: List[Alert] = []
        self._max_history = max_history

    def add_handler(self, handler: AlertHandler) -> None:
        """Register an alert handler.

        Args:
            handler: Callable that receives an Alert.
        """
        self._handlers.append(handler)

    def fire(
        self,
        level: str,
        category: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and dispatch an alert.

        Args:
            level: Severity.
            category: Category tag.
            message: Description.
            data: Extra payload.

        Returns:
            The created Alert.
        """
        alert = Alert(level, category, message, data)
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                logger.exception("Alert handler failed")

        return alert

    @property
    def history(self) -> List[Alert]:
        """Alert history (most recent last)."""
        return list(self._history)

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last *n* alerts as dicts.

        Args:
            n: Number of alerts to return.
        """
        return [a.to_dict() for a in self._history[-n:]]


# ── Built-in handlers ─────────────────────────────────────────


def console_handler(alert: Alert) -> None:
    """Print alert to console / logger."""
    log_fn = {
        AlertLevel.INFO: logger.info,
        AlertLevel.WARNING: logger.warning,
        AlertLevel.CRITICAL: logger.critical,
    }.get(alert.level, logger.info)

    log_fn(
        "[%s] %s: %s  %s",
        alert.level,
        alert.category,
        alert.message,
        alert.data,
    )


def file_handler_factory(
    path: str = "alerts.jsonl",
) -> AlertHandler:
    """Create a handler that appends alerts to a JSONL file.

    Args:
        path: File path for the alert log.

    Returns:
        AlertHandler function.
    """

    def _handler(alert: Alert) -> None:
        """Append alert JSON to the log file."""
        with open(path, "a") as f:
            f.write(alert.to_json() + "\n")

    return _handler


# ── Global dispatcher ─────────────────────────────────────────

_dispatcher: Optional[AlertDispatcher] = None


def get_alert_dispatcher() -> AlertDispatcher:
    """Return (or create) the global AlertDispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = AlertDispatcher()
        _dispatcher.add_handler(console_handler)
    return _dispatcher
