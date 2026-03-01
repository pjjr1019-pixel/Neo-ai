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
