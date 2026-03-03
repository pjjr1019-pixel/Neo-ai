# Phase 8: Local Interactive GUI

## Framework Choice
- Selected framework: lightweight Python-native modular GUI stack with a
  headless launch path (`python_ai/gui/launch.py`) and composable state
  modules under `python_ai/gui/`.
- Rationale: low dependency risk, easy local deployment, direct integration
  with Python backend modules.

## Implemented Modules
- `python_ai/gui/app.py`: main application state and default layout.
- `python_ai/gui/components.py`: widget and layout specs.
- `python_ai/gui/charts.py`: candlestick, indicator, and P&L transforms.
- `python_ai/gui/controls.py`: control validation for backtest/live toggles.
- `python_ai/gui/model_manager.py`: load/save model/strategy payloads.
- `python_ai/gui/trading_panel.py`: start/stop/error state transitions.
- `python_ai/gui/notifications.py`: in-app notifications and alerts.
- `python_ai/gui/settings.py`: persisted settings (API key, risk/theme).
- `python_ai/gui/help_about.py`: help topics and About dialog content.

## Validation
- Unit tests:
  - `tests/test_gui_modules.py`
  - `tests/test_gui_e2e.py`
  - `tests/test_gui_accessibility_performance.py`

## Integration
- GUI state modules are backend-ready and designed for API adapter wiring in
  runtime launchers.
