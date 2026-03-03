"""Cross-platform installer planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class InstallerPlan:
    """Installer bundle specification."""

    tool: str
    targets: List[str]
    bundled_components: List[str]


def create_installer_plan() -> InstallerPlan:
    """Create default installer plan for Windows/macOS/Linux."""
    return InstallerPlan(
        tool="PyInstaller",
        targets=["windows", "macos", "linux"],
        bundled_components=["python", "gui", "models", "configs"],
    )


def installer_health_report() -> Dict[str, object]:
    """Return readiness report for installer generation."""
    plan = create_installer_plan()
    return {
        "tool": plan.tool,
        "targets": plan.targets,
        "bundle_count": len(plan.bundled_components),
        "ready": True,
    }
