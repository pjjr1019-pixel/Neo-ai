"""User settings persistence for GUI configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class UserSettings:
    """Persisted GUI settings."""

    api_key: str = ""
    risk_limit: float = 0.02
    theme: str = "light"
    colorblind_mode: bool = False


def save_settings(path: str | Path, settings: UserSettings) -> Path:
    """Persist user settings as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return target


def load_settings(path: str | Path) -> UserSettings:
    """Load settings from disk, or defaults when file is missing."""
    target = Path(path)
    if not target.exists():
        return UserSettings()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return UserSettings(**payload)
