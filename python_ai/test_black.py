import shutil
import subprocess

import pytest


def test_black_formatting():
    """Check code formatting with black."""
    black_path = shutil.which("black")
    if black_path is None:
        pytest.skip(
            "black command not found. "
            "Install with: pip install -r requirements-dev.txt"
        )

    result = subprocess.run(
        ["black", "--check", "--line-length", "79", "python_ai"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Black formatting errors found:\n" + result.stdout
    )
