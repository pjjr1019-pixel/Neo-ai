import subprocess


def test_black_formatting():
    """Check code formatting with black."""
    result = subprocess.run(
        ["black", "--check", "python_ai"], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "Black formatting errors found:\n" + result.stdout
    )
