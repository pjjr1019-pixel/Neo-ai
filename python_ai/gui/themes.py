"""Theme and accessibility helpers for GUI rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Theme:
    """Theme definition."""

    name: str
    colors: Dict[str, str]


LIGHT = Theme(
    name="light",
    colors={"bg": "#F4F1EA", "fg": "#162027", "accent": "#C85A3A"},
)
DARK = Theme(
    name="dark",
    colors={"bg": "#1B1F22", "fg": "#F7F2E8", "accent": "#F2A65A"},
)
COLORBLIND = Theme(
    name="colorblind",
    colors={"bg": "#FFFFFF", "fg": "#111111", "accent": "#0072B2"},
)


def get_theme(name: str) -> Theme:
    """Return configured theme object."""
    if name == "dark":
        return DARK
    if name == "colorblind":
        return COLORBLIND
    return LIGHT
