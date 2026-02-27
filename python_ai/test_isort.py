
import subprocess


def test_isort_imports():
    """Check import sorting with isort."""
    result = subprocess.run([
        "isort", "--check-only", "python_ai"
    ], capture_output=True, text=True)
    assert result.returncode == 0, (
        "Isort import sorting errors found:\n" + result.stdout
    )
