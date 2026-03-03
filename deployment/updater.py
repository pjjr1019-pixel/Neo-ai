"""Auto-update notification and rollback decision utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class UpdateDecision:
    """Result of update gate evaluation."""

    should_update: bool
    rollback_required: bool
    reason: str


def evaluate_update(
    current_version: str,
    latest_version: str,
    *,
    health_ok: bool = True,
) -> UpdateDecision:
    """Determine whether update should proceed and if rollback is required."""
    if not health_ok:
        return UpdateDecision(False, True, "health_check_failed")
    if current_version == latest_version:
        return UpdateDecision(False, False, "already_up_to_date")
    return UpdateDecision(True, False, "update_available")


def update_report(decision: UpdateDecision) -> Dict[str, object]:
    """Serialize update decision for logging/display."""
    return {
        "should_update": decision.should_update,
        "rollback_required": decision.rollback_required,
        "reason": decision.reason,
    }
