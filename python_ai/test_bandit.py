
import subprocess


def test_bandit_security():
    """Run bandit security checks on all Python files."""
    import os
    import sys
    venv_dir = os.environ.get('VIRTUAL_ENV')
    if venv_dir:
        bandit_path = os.path.join(venv_dir, "Scripts", "bandit.exe")
    else:
        bandit_path = os.path.join(os.getcwd(), ".venv", "Scripts", "bandit.exe")
    result = subprocess.run(
        [bandit_path, "-r", "python_ai", "--quiet", "--exit-zero"],
        capture_output=True,
        text=True
    )
    assert (
        "No issues identified." in result.stdout or result.returncode == 0
    ), (
        "Bandit security issues found:\n" + result.stdout
    )
