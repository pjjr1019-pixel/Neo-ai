"""Launch script for local desktop GUI."""

from __future__ import annotations

from python_ai.gui.app import create_default_app_state


def launch_headless_summary() -> str:
    """Return startup summary for environments without display server."""
    state = create_default_app_state()
    widgets = len(state.layout.dashboard_widgets)
    return f"NEO GUI initialized with {widgets} widgets"


if __name__ == "__main__":  # pragma: no cover
    print(launch_headless_summary())
