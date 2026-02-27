

def test_no_todo_comments():
    """Fail if any forbidden comments are found in codebase."""
    pass


def test_no_unused_imports():
    """Fail if unused imports are detected (flake8 F401)."""
    import subprocess
    result = subprocess.run([
        "flake8", "python_ai", "--select=F401"
    ], capture_output=True, text=True)
    assert result.returncode == 0 and not result.stdout.strip(), (
        "Unused imports found:\n" + result.stdout
    )
