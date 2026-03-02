"""
Alert Notification Integration for NEO Hybrid AI.

Provides a pluggable notification framework supporting
multiple channels: Slack, Telegram, Discord, and
generic webhooks.  Channel backends are intentionally
simple so they work without extra SDK dependencies.
"""

import json
import logging
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _safe_urlopen(
    req: urllib.request.Request,
    *,
    timeout: float,
) -> "urllib.response.addinfourl":
    """Open *req* after validating its URL scheme.

    Only ``http`` and ``https`` schemes are permitted;
    ``file://`` and custom schemes are rejected to
    prevent path-traversal (CWE-22 / B310).

    Raises:
        ValueError: If the scheme is not allowed.
    """
    parsed = urlparse(req.full_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme {parsed.scheme!r} is not allowed; "
            f"only {sorted(_ALLOWED_SCHEMES)} are permitted"
        )
    return urllib.request.urlopen(  # type: ignore[no-any-return]
        req, timeout=timeout
    )  # nosec B310


logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """An alert to be dispatched.

    Attributes:
        title: Short headline.
        message: Alert body text.
        severity: ``info``, ``warning``, ``critical``.
        source: Module or subsystem that raised it.
        timestamp: Unix epoch (auto-populated).
        metadata: Arbitrary extra data.
    """

    title: str
    message: str
    severity: str = "info"
    source: str = "neo"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationChannel(ABC):
    """Base class for notification backends."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Dispatch an alert.

        Args:
            alert: The alert to send.

        Returns:
            ``True`` if successfully dispatched.
        """


class WebhookChannel(NotificationChannel):
    """Generic JSON webhook poster.

    Posts the alert as JSON to the configured URL.

    Args:
        url: Webhook endpoint.
        headers: Extra HTTP headers.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> None:
        """Initialise the channel."""
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Post the alert as JSON.

        Args:
            alert: Alert to send.

        Returns:
            ``True`` on HTTP 2xx.
        """
        payload = {
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity,
            "source": alert.source,
            "timestamp": alert.timestamp,
            "metadata": alert.metadata,
        }
        data = json.dumps(payload).encode()
        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }
        req = urllib.request.Request(self._url, data=data, headers=headers)
        try:
            with _safe_urlopen(req, timeout=self._timeout) as resp:
                status: int = resp.status or 0
                return 200 <= status < 300
        except Exception:
            logger.exception("Webhook send failed: %s", self._url)
            return False


class SlackChannel(NotificationChannel):
    """Slack incoming-webhook notification channel.

    Args:
        webhook_url: Slack incoming webhook URL.
        timeout: HTTP timeout.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 10.0,
    ) -> None:
        """Initialise the channel."""
        self._url = webhook_url
        self._timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Send to Slack via incoming webhook.

        Args:
            alert: Alert to send.

        Returns:
            ``True`` on success.
        """
        icon = _severity_emoji(alert.severity)
        payload = {
            "text": (
                f"{icon} *{alert.title}*\n"
                f"{alert.message}\n"
                f"_source: {alert.source}_"
            )
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _safe_urlopen(req, timeout=self._timeout) as resp:
                status: int = resp.status or 0
                return 200 <= status < 300
        except Exception:
            logger.exception("Slack send failed")
            return False


class TelegramChannel(NotificationChannel):
    """Telegram Bot API notification channel.

    Args:
        bot_token: Telegram Bot API token.
        chat_id: Target chat/group ID.
        timeout: HTTP timeout.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout: float = 10.0,
    ) -> None:
        """Initialise the channel."""
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Send to Telegram.

        Args:
            alert: Alert to send.

        Returns:
            ``True`` on success.
        """
        icon = _severity_emoji(alert.severity)
        text = (
            f"{icon} {alert.title}\n\n"
            f"{alert.message}\n"
            f"Source: {alert.source}"
        )
        url = f"https://api.telegram.org/" f"bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _safe_urlopen(req, timeout=self._timeout) as resp:
                status: int = resp.status or 0
                return 200 <= status < 300
        except Exception:
            logger.exception("Telegram send failed")
            return False


class DiscordChannel(NotificationChannel):
    """Discord webhook notification channel.

    Args:
        webhook_url: Discord webhook URL.
        timeout: HTTP timeout.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 10.0,
    ) -> None:
        """Initialise the channel."""
        self._url = webhook_url
        self._timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Send to Discord.

        Args:
            alert: Alert to send.

        Returns:
            ``True`` on success.
        """
        icon = _severity_emoji(alert.severity)
        payload = {
            "content": (
                f"{icon} **{alert.title}**\n"
                f"{alert.message}\n"
                f"*source: {alert.source}*"
            )
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _safe_urlopen(req, timeout=self._timeout) as resp:
                status: int = resp.status or 0
                return 200 <= status < 300
        except Exception:
            logger.exception("Discord send failed")
            return False


def _severity_emoji(severity: str) -> str:
    """Map severity to an emoji.

    Args:
        severity: ``info``, ``warning``, ``critical``.

    Returns:
        Unicode emoji string.
    """
    return {
        "info": "\u2139\ufe0f",
        "warning": "\u26a0\ufe0f",
        "critical": "\U0001f6a8",
    }.get(severity, "\u2139\ufe0f")


class AlertDispatcher:
    """Dispatches alerts to multiple channels.

    Args:
        channels: Optional initial list of channels.
    """

    def __init__(
        self,
        channels: Optional[Sequence[NotificationChannel]] = None,
    ) -> None:
        """Initialise the dispatcher."""
        self._channels: List[NotificationChannel] = list(channels or [])
        self._sent = 0
        self._failed = 0
        self._history: List[Dict[str, Any]] = []
        self._max_history = 200

    def add_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel.

        Args:
            channel: Backend to add.
        """
        self._channels.append(channel)

    def dispatch(self, alert: Alert) -> Dict[str, Any]:
        """Send an alert to all registered channels.

        Args:
            alert: Alert to dispatch.

        Returns:
            Dict with ``sent`` count, ``failed`` count,
            and per-channel results.
        """
        results: Dict[str, bool] = {}
        for ch in self._channels:
            name = type(ch).__name__
            try:
                ok = ch.send(alert)
            except Exception:
                logger.exception("Channel %s raised", name)
                ok = False
            results[name] = ok
            if ok:
                self._sent += 1
            else:
                self._failed += 1

        entry = {
            "title": alert.title,
            "severity": alert.severity,
            "timestamp": alert.timestamp,
            "results": results,
        }
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return {
            "sent": sum(1 for v in results.values() if v),
            "failed": sum(1 for v in results.values() if not v),
            "results": results,
        }

    def summary(self) -> Dict[str, Any]:
        """Return dispatcher stats.

        Returns:
            Dict with channel count, sent/failed totals.
        """
        return {
            "channels": len(self._channels),
            "total_sent": self._sent,
            "total_failed": self._failed,
            "history_size": len(self._history),
        }
