"""Unit tests for GUI module components and workflows."""

from pathlib import Path

from python_ai.gui.app import create_default_app_state
from python_ai.gui.charts import candlestick_series, indicator_overlay, pnl_series
from python_ai.gui.controls import validate_backtest_controls
from python_ai.gui.help_about import about_dialog, build_help_topics
from python_ai.gui.model_manager import load_strategy, save_strategy
from python_ai.gui.notifications import NotificationCenter
from python_ai.gui.settings import UserSettings, load_settings, save_settings
from python_ai.gui.themes import get_theme
from python_ai.gui.trading_panel import TradingState, register_error, start_trading, stop_trading


def test_create_default_app_state_has_layout() -> None:
    state = create_default_app_state()
    assert state.layout.navigation_items
    assert len(state.layout.dashboard_widgets) >= 1


def test_chart_transforms() -> None:
    candles = [{"open": 1, "high": 2, "low": 0.5, "close": 1.5}]
    series = candlestick_series(candles)
    assert series[0]["high"] == 2.0
    assert pnl_series([{"pnl": 1.0}, {"pnl": -0.5}]) == [1.0, 0.5]
    overlay = indicator_overlay([1, 2], {"ema": [1.1, 1.2]})
    assert "ema" in overlay


def test_control_validation() -> None:
    ok = validate_backtest_controls(
        symbol="BTC/USD",
        lookback_days=30,
        risk_limit=0.1,
        mode="paper",
    )
    assert ok.valid is True
    bad = validate_backtest_controls(
        symbol="",
        lookback_days=0,
        risk_limit=2.0,
        mode="invalid",
    )
    assert bad.valid is False
    assert len(bad.errors) >= 2


def test_model_manager_save_load(tmp_path: Path) -> None:
    path = tmp_path / "strategy.json"
    payload = {"name": "baseline", "threshold": 0.5}
    save_strategy(path, payload)
    restored = load_strategy(path)
    assert restored["name"] == "baseline"


def test_trading_state_transitions_and_notifications() -> None:
    state = TradingState()
    start_trading(state, mode="paper")
    assert state.running is True
    register_error(state, "network timeout")
    assert "timeout" in state.last_error
    stop_trading(state)
    assert state.running is False

    center = NotificationCenter()
    center.push("warning", "drift detected")
    assert center.unread_count() == 1


def test_settings_and_help_about(tmp_path: Path) -> None:
    settings = UserSettings(api_key="abc", risk_limit=0.03, theme="dark")
    path = save_settings(tmp_path / "settings.json", settings)
    loaded = load_settings(path)
    assert loaded.api_key == "abc"
    assert loaded.theme == "dark"
    assert "getting_started" in build_help_topics()
    assert "Version" in about_dialog("1.0.0")


def test_theme_switching() -> None:
    assert get_theme("light").name == "light"
    assert get_theme("dark").name == "dark"
    assert get_theme("colorblind").name == "colorblind"
