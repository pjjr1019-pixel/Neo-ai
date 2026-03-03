"""Compliance checks for GDPR/CCPA policy enforcement in CI/CD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ComplianceResult:
    """Result of one compliance check."""

    check: str
    passed: bool
    detail: str


def _required_artifacts() -> Dict[str, str]:
    """Return required policy/document artifacts for compliance."""
    return {
        "data_handling_policy": "security/data_handling_policy.md",
        "consent_controls": "security/user_consent.py",
        "regulation_monitor": "security/regulation_monitor.py",
        "compliance_audit": "security/compliance_audit.py",
    }


def run_compliance_checks(
    repo_root: str | Path = ".",
) -> List[ComplianceResult]:
    """Run lightweight compliance checks suitable for CI execution.

    Args:
        repo_root: Repository root where policy artifacts should exist.

    Returns:
        List of check outcomes.
    """
    root = Path(repo_root)
    results: List[ComplianceResult] = []

    for name, rel_path in _required_artifacts().items():
        target = root / rel_path
        passed = target.exists()
        results.append(
            ComplianceResult(
                check=name,
                passed=passed,
                detail=f"{rel_path} {'found' if passed else 'missing'}",
            )
        )

    return results


def compliance_summary(repo_root: str | Path = ".") -> Dict[str, object]:
    """Return aggregate summary for CI/CD consumption."""
    results = run_compliance_checks(repo_root=repo_root)
    passed = all(item.passed for item in results)
    return {
        "passed": passed,
        "checks": [item.__dict__ for item in results],
    }


if __name__ == "__main__":  # pragma: no cover
    summary = compliance_summary()
    print(summary)
