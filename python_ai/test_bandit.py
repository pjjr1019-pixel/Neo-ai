import os
import subprocess


def test_bandit_security():
    """Run bandit security checks on all Python files."""
    import shutil
    bandit_path = shutil.which("bandit") or "bandit"
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
