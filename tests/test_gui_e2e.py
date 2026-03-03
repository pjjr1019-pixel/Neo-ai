"""End-to-end style tests for major GUI workflows."""

from pathlib import Path

from python_ai.gui.app import create_default_app_state
from python_ai.gui.controls import validate_backtest_controls
from python_ai.gui.model_manager import load_strategy, save_strategy
from python_ai.gui.notifications import NotificationCenter
from python_ai.gui.settings import UserSettings, load_settings, save_settings
from python_ai.gui.trading_panel import TradingState, start_trading, stop_trading


def test_e2e_strategy_management_workflow(tmp_path: Path) -> None:
    strategy_path = tmp_path / "strategy.json"
    save_strategy(strategy_path, {"name": "s1", "risk": 0.02})
    restored = load_strategy(strategy_path)
    assert restored["name"] == "s1"


def test_e2e_trading_start_stop_workflow() -> None:
    state = TradingState()
    start_trading(state, mode="paper")
    assert state.running is True
    stop_trading(state)
    assert state.running is False


def test_e2e_settings_persistence_workflow(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    save_settings(
        settings_path,
        UserSettings(api_key="k", risk_limit=0.05, theme="light"),
    )
    loaded = load_settings(settings_path)
    assert loaded.risk_limit == 0.05


def test_e2e_dashboard_control_validation_workflow() -> None:
    app_state = create_default_app_state()
    assert len(app_state.layout.navigation_items) >= 3
    validation = validate_backtest_controls(
        symbol="ETH/USD",
        lookback_days=14,
        risk_limit=0.15,
        mode="paper",
    )
    assert validation.valid is True


def test_e2e_notification_workflow() -> None:
    center = NotificationCenter()
    center.push("error", "drift threshold breached")
    assert center.unread_count() == 1
