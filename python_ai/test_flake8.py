"""
Test that all Python files in the project are flake8-compliant.
Fails if any flake8 errors or warnings are found.
"""

import subprocess
import sys


def test_flake8_compliance():
    """Run flake8 on the project and fail if any errors/warnings are found."""
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "python_ai/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
    assert result.returncode == 0, f"Flake8 errors found:\n{result.stdout}"


def test_flake8_error_print(monkeypatch, capsys):
    """Test print output when flake8 errors are found."""
    class FakeResult:
        returncode = 1
        stdout = 'E999 syntax error'
    monkeypatch.setattr('subprocess.run', lambda *a, **k: FakeResult())
    try:
        test_flake8_compliance()
    except AssertionError:
        pass
    out = capsys.readouterr().out
    assert 'E999 syntax error' in out
