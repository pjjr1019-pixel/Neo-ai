
import subprocess


def test_coverage_threshold():
    """Require minimum test coverage for CI compliance."""
    result = subprocess.run([
        "pytest", "--cov=python_ai", "--cov-report=term", "--cov-fail-under=90"
    ], capture_output=True, text=True)
    assert result.returncode == 0, (
        "Coverage below threshold:\n" + result.stdout
    )
