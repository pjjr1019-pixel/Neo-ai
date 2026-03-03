"""GUI-facing model and strategy file management helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, cast


def save_strategy(path: str | Path, payload: Dict[str, object]) -> Path:
    """Save strategy configuration as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_strategy(path: str | Path) -> Dict[str, object]:
    """Load strategy configuration from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return cast(Dict[str, object], payload)
