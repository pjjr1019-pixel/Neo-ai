import subprocess


def test_git_status_clean():
    """Fail if git status is not clean (uncommitted changes, conflicts)."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    assert not result.stdout.strip(), (
        "Git working tree is not clean:\n" + result.stdout
    )


def test_git_config_safe():
    """Fail if git config has unsafe settings."""
    result = subprocess.run(
        ["git", "config", "--global", "--list"], capture_output=True, text=True
    )
    assert "unsafe" not in result.stdout.lower(), (
        "Unsafe git config found:\n" + result.stdout
    )
