import os
import subprocess

import pytest


def test_git_status_clean():
    """Require clean git tree only in CI environments."""
    if os.getenv("CI", "").lower() not in {"1", "true"}:
        pytest.skip("Clean working tree check is enforced in CI only.")

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
