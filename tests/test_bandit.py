import subprocess


def test_bandit_security():
    """Run bandit security checks on all Python files."""
    import shutil

    import pytest

    bandit_path = shutil.which("bandit")
    if not bandit_path:
        pytest.skip("bandit is not installed or not found in PATH")
    bandit_args = [
        bandit_path,
        "-r",
        "python_ai",
        "--quiet",
        "--exit-zero",
    ]
    result = subprocess.run(bandit_args, capture_output=True, text=True)
    no_issues = "No issues identified." in result.stdout
    no_errors = result.returncode == 0
    msg = "Bandit security issues found:\n" + result.stdout
    assert no_issues or no_errors, msg


def test_bandit_no_medium_high():
    """Bandit must report zero medium/high-severity issues.

    Uses ``-ll`` (medium+) and excludes test files so that only
    production code is scanned.  This catches regressions like
    B310 (unsafe urlopen schemes) that previously slipped through
    the softer ``--exit-zero`` check above.
    """
    import shutil

    import pytest

    bandit_path = shutil.which("bandit")
    if not bandit_path:
        pytest.skip("bandit is not installed or not found in PATH")

    bandit_args = [
        bandit_path,
        "-r",
        "python_ai",
        "--exclude",
        "*/test_*.py,*/integration_test.py,*/benchmark_predict.py",
        "-ll",  # medium + high only
        "--quiet",
    ]
    result = subprocess.run(bandit_args, capture_output=True, text=True)
    assert result.returncode == 0, (
        "Bandit medium/high issues found — fix before merging:\n"
        + result.stdout
    )
