"""Help and about dialog content generation."""

from __future__ import annotations

from typing import Dict


def build_help_topics() -> Dict[str, str]:
    """Return in-app help topics and quick guidance."""
    return {
        "getting_started": (
            "Connect API keys, choose paper mode, run a backtest."
        ),
        "risk_controls": (
            "Set risk_limit and max drawdown before live trading."
        ),
        "alerts": "Review drift and error notifications in the alerts panel.",
    }


def about_dialog(version: str) -> str:
    """Return About dialog content."""
    return f"NEO Hybrid AI Desktop\nVersion: {version}"
