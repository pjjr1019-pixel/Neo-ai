"""Configuration management for NEO AI Platform.

This module provides centralized, environment-based configuration with:
- Type-safe settings using Pydantic
- Environment variable loading from .env files
- Validation and defaults
- Hierarchical configuration structure
"""

from python_ai.config.settings import (
    Settings,
    get_settings,
    DatabaseSettings,
    AuthSettings,
    LoggingSettings,
    ModelSettings,
    APISettings,
)

__all__ = [
    "Settings",
    "get_settings",
    "DatabaseSettings",
    "AuthSettings",
    "LoggingSettings",
    "ModelSettings",
    "APISettings",
]
