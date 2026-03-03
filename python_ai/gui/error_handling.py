"""Centralized GUI error handling and persistent exportable logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ErrorEvent:
    """One GUI/runtime error event."""

    source: str
    message: str


@dataclass
class ErrorManager:
    """Error collector with export support."""

    events: List[ErrorEvent] = field(default_factory=list)

    def capture(self, source: str, message: str) -> None:
        """Capture one error event from a GUI subsystem."""
        self.events.append(ErrorEvent(source=source, message=message))

    def export(self, path: str | Path) -> Path:
        """Export captured errors to a plain-text log file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{event.source}: {event.message}" for event in self.events]
        target.write_text("\n".join(lines), encoding="utf-8")
        return target
