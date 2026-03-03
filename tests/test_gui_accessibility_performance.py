"""Tests for GUI accessibility, error handling, and performance utilities."""

from pathlib import Path

from python_ai.gui.error_handling import ErrorManager
from python_ai.gui.launch import launch_headless_summary
from python_ai.gui.performance import profile_operation, regression_check
from python_ai.gui.settings import UserSettings
from python_ai.gui.themes import get_theme


def test_colorblind_theme_available() -> None:
    theme = get_theme("colorblind")
    assert theme.colors["accent"] == "#0072B2"


def test_headless_launch_summary() -> None:
    summary = launch_headless_summary()
    assert "NEO GUI initialized" in summary


def test_error_manager_export(tmp_path: Path) -> None:
    manager = ErrorManager()
    manager.capture("trading_panel", "connection lost")
    path = manager.export(tmp_path / "errors.log")
    assert path.exists()
    assert "connection lost" in path.read_text(encoding="utf-8")


def test_performance_regression_helper() -> None:
    sample = profile_operation("noop", lambda: None)
    assert sample.duration_ms >= 0.0
    report = regression_check(100.0, 110.0, tolerance_ratio=1.2)
    assert report["regressed"] is False
    report2 = regression_check(100.0, 130.0, tolerance_ratio=1.2)
    assert report2["regressed"] is True


def test_settings_dataclass_for_accessibility_flags() -> None:
    settings = UserSettings(colorblind_mode=True)
    assert settings.colorblind_mode is True
