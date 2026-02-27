import subprocess
import os


def test_bandit_security():
    """Run bandit security checks on all Python files."""
    venv_dir = os.environ.get("VIRTUAL_ENV")
    if venv_dir:
        bandit_path = os.path.join(venv_dir, "Scripts", "bandit.exe")
    else:
        bandit_path = os.path.join(os.getcwd(), ".venv", "Scripts", "bandit.exe")
    bandit_args = [bandit_path, "-r", "python_ai", "--quiet", "--exit-zero"]
    result = subprocess.run(bandit_args, capture_output=True, text=True)
    no_issues = "No issues identified." in result.stdout
    no_errors = result.returncode == 0
    msg = "Bandit security issues found:\n" + result.stdout
    assert no_issues or no_errors, msg
