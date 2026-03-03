"""Core GUI component definitions used by the local desktop interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class WidgetSpec:
    """Declarative description of one GUI widget."""

    widget_id: str
    widget_type: str
    title: str
    visible: bool = True
    props: Dict[str, object] = field(default_factory=dict)


@dataclass
class LayoutSpec:
    """Top-level layout definition for dashboard/navigation/status bar."""

    dashboard_widgets: List[WidgetSpec]
    navigation_items: List[str]
    status_bar_items: List[str]
