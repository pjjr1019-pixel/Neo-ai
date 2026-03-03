"""Monitor regulation metadata changes for compliance workflows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class RegulationSnapshot:
    """Content hash snapshot for one regulation source."""

    name: str
    digest: str


def regulation_digest(path: str | Path) -> str:
    """Compute deterministic SHA-256 digest for a regulation source file."""
    content = Path(path).read_bytes()
    return hashlib.sha256(content).hexdigest()


def detect_updates(
    previous: Dict[str, str],
    current_paths: Dict[str, str],
) -> Dict[str, bool]:
    """Detect which regulation sources changed since previous snapshot."""
    changed: Dict[str, bool] = {}
    for name, source_path in current_paths.items():
        digest = regulation_digest(source_path)
        changed[name] = previous.get(name) != digest
    return changed
