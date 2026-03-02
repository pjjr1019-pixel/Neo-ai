"""Environment validation tests for CI/local development compatibility."""

import shutil
import subprocess
import sys

import pytest


class TestEnvironmentSetup:
    """Validate that required tools are available in the environment."""

    def test_pytest_available(self):
        """Verify pytest is installed."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "pytest not available"

    def test_optional_tools_warning(self):
        """Warn if optional dev tools are missing."""
        optional_tools = {
            "black": "Code formatter - pip install black",
            "flake8": "Linter - pip install flake8",
            "isort": "Import sorter - pip install isort",
            "mypy": "Type checker - pip install mypy",
        }

        missing = []
        for tool, install_cmd in optional_tools.items():
            path = shutil.which(tool)
            if path is None:
                result = subprocess.run(
                    [sys.executable, "-m", tool, "--version"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    missing.append(f"{tool}: {install_cmd}")

        if missing:
            pytest.skip(
                f"Optional dev tools not installed:\n"
                f"{chr(10).join(missing)}\n"
                f"Install with: pip install -r requirements-dev.txt"
            )

    def test_flake8_module_available(self):
        """Check if flake8 module is available."""
        result = subprocess.run(
            [sys.executable, "-m", "flake8", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(
                "flake8 module not installed. "
                "Install with: pip install flake8"
            )

    def test_isort_module_available(self):
        """Check if isort module is available."""
        result = subprocess.run(
            [sys.executable, "-m", "isort", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(
                "isort module not installed. "
                "Install with: pip install isort"
            )

    def test_black_command_available(self):
        """Check if black command is available."""
        result = shutil.which("black")
        if result is None:
            pytest.skip(
                "black command not found in PATH. "
                "Install with: pip install black"
            )

    def test_mypy_module_available(self):
        """Check if mypy module is available."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(
                "mypy module not installed. " "Install with: pip install mypy"
            )
