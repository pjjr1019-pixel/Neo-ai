"""Tests for CI compliance checks and related security controls."""

from __future__ import annotations

from pathlib import Path

from ci.compliance_checks import compliance_summary, run_compliance_checks
from security.compliance_audit import next_quarterly_audit, run_audit
from security.regulation_monitor import detect_updates, regulation_digest
from security.user_consent import ConsentManager


def test_run_compliance_checks_detects_required_artifacts() -> None:
    results = run_compliance_checks()
    assert results
    assert all(hasattr(item, "passed") for item in results)


def test_compliance_summary_has_boolean_pass_flag() -> None:
    summary = compliance_summary()
    assert "passed" in summary
    assert isinstance(summary["passed"], bool)
    assert isinstance(summary["checks"], list)


def test_regulation_monitor_detects_changes(tmp_path: Path) -> None:
    source = tmp_path / "gdpr.txt"
    source.write_text("v1", encoding="utf-8")
    first = regulation_digest(source)
    source.write_text("v2", encoding="utf-8")
    changed = detect_updates({"gdpr": first}, {"gdpr": str(source)})
    assert changed["gdpr"] is True


def test_consent_manager_set_and_revoke() -> None:
    manager = ConsentManager()
    manager.set_consent("user1", granted=True, scope="trading_data")
    assert manager.has_consent("user1")
    assert manager.has_consent("user1", scope="trading_data")
    manager.revoke("user1")
    assert not manager.has_consent("user1")


def test_compliance_audit_helpers() -> None:
    record = run_audit(["consent", "encryption"])
    assert record.status == "passed"
    next_run = next_quarterly_audit(record.executed_at)
    assert next_run > record.executed_at
