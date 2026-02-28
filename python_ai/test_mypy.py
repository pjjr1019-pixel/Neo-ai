import subprocess


def test_mypy_type_checking():
    """Check static typing with mypy."""
    result = subprocess.run(
        ["mypy", "python_ai"], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "Mypy type errors found:\n" + result.stdout
    )
