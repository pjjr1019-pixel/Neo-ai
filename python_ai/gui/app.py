"""Composable GUI application state for desktop workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from python_ai.gui.components import LayoutSpec, WidgetSpec
from python_ai.gui.notifications import NotificationCenter
from python_ai.gui.settings import UserSettings
from python_ai.gui.trading_panel import TradingState


@dataclass
class GuiAppState:
    """Container for GUI subsystems."""

    layout: LayoutSpec
    trading: TradingState = field(default_factory=TradingState)
    notifications: NotificationCenter = field(
        default_factory=NotificationCenter
    )
    settings: UserSettings = field(default_factory=UserSettings)


def create_default_app_state() -> GuiAppState:
    """Create default GUI state with dashboard/navigation/status widgets."""
    widgets = [
        WidgetSpec("chart_main", "chart", "Price Chart"),
        WidgetSpec("pnl_view", "chart", "P&L"),
        WidgetSpec("signal_panel", "table", "Signals"),
    ]
    layout = LayoutSpec(
        dashboard_widgets=widgets,
        navigation_items=[
            "Dashboard",
            "Strategy",
            "Models",
            "Trading",
            "Settings",
        ],
        status_bar_items=["connection", "mode", "latency_ms"],
    )
    return GuiAppState(layout=layout)
