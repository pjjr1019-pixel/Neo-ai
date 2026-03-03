"""Trading panel state transitions for live/simulated controls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingState:
    """Current GUI trading state."""

    running: bool = False
    mode: str = "paper"
    last_error: str = ""


def start_trading(state: TradingState, mode: str) -> TradingState:
    """Start trading in selected mode."""
    state.running = True
    state.mode = mode
    state.last_error = ""
    return state


def stop_trading(state: TradingState) -> TradingState:
    """Stop trading loop."""
    state.running = False
    return state


def register_error(state: TradingState, message: str) -> TradingState:
    """Store latest trade/runtime error for UI display."""
    state.last_error = message
    return state
