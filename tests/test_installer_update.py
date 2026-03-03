"""Tests for installer planning and update/rollback helpers."""

from deployment.installer import create_installer_plan, installer_health_report
from deployment.updater import evaluate_update, update_report


def test_installer_plan_targets_all_platforms() -> None:
    plan = create_installer_plan()
    assert plan.tool == "PyInstaller"
    assert set(plan.targets) == {"windows", "macos", "linux"}
    assert "gui" in plan.bundled_components


def test_installer_health_report_ready() -> None:
    report = installer_health_report()
    assert report["ready"] is True
    assert report["bundle_count"] >= 1


def test_update_decision_flow() -> None:
    no_update = evaluate_update("1.0.0", "1.0.0")
    assert no_update.should_update is False
    assert no_update.rollback_required is False

    update = evaluate_update("1.0.0", "1.1.0")
    assert update.should_update is True

    rollback = evaluate_update("1.0.0", "1.1.0", health_ok=False)
    assert rollback.rollback_required is True
    payload = update_report(rollback)
    assert payload["reason"] == "health_check_failed"
