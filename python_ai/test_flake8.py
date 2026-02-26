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
