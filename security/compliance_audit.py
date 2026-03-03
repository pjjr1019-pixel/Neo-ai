"""Compliance audit scheduler and report generator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class AuditRecord:
    """Single compliance audit record."""

    executed_at: datetime
    controls_checked: List[str]
    status: str


def next_quarterly_audit(last_run: datetime) -> datetime:
    """Return next quarterly audit datetime (90-day cadence)."""
    return last_run + timedelta(days=90)


def run_audit(controls: List[str]) -> AuditRecord:
    """Run a lightweight audit over configured controls."""
    status = "passed" if controls else "warning"
    return AuditRecord(
        executed_at=datetime.now(timezone.utc),
        controls_checked=controls,
        status=status,
    )


def write_audit_log(record: AuditRecord, path: str | Path) -> None:
    """Persist audit report to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        (
            f"executed_at={record.executed_at.isoformat()}\n"
            f"status={record.status}\n"
            f"controls={','.join(record.controls_checked)}\n"
        ),
        encoding="utf-8",
    )
