import subprocess
import sys
import os
import pytest


def test_coverage_threshold():
    """Require minimum test coverage for CI compliance.

    Skips if no coverage data file is present (e.g. mid-run or no --cov).
    """
    if not os.path.exists(".coverage"):
        pytest.skip("No .coverage data file found; skipping threshold check.")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            "--fail-under=90",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Coverage below threshold:\n" + result.stdout + result.stderr
    )
