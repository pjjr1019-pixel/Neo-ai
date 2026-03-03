"""In-app notifications for errors, alerts, and drift warnings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Notification:
    """Notification payload."""

    level: str
    message: str


@dataclass
class NotificationCenter:
    """Store and query notification events."""

    items: List[Notification] = field(default_factory=list)

    def push(self, level: str, message: str) -> None:
        """Append a new notification item."""
        self.items.append(Notification(level=level, message=message))

    def unread_count(self) -> int:
        """Return number of currently stored notifications."""
        return len(self.items)
