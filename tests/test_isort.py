import subprocess
import sys

import pytest


def test_isort_imports():
    """Check import sorting with isort."""
    result = subprocess.run(
        [sys.executable, "-m", "isort", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "isort module not installed. "
            "Install with: pip install -r requirements-dev.txt"
        )

    result = subprocess.run(
        [sys.executable, "-m", "isort", "--check-only", "python_ai"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Isort import sorting errors found:\n" + result.stdout
    )
