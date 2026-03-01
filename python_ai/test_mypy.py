import subprocess
import sys

import pytest


def test_mypy_type_checking():
    """Check static typing with mypy."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "mypy module not installed. "
            "Install with: pip install -r requirements-dev.txt"
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "python_ai",
            "--ignore-missing-imports",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Mypy type errors found:\n" + result.stdout
