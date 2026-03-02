"""
Database configuration for NEO Hybrid AI.

This module is a **compatibility shim** that delegates to the
canonical ``python_ai.config.settings.DatabaseSettings``.

All new code should import from ``python_ai.config.settings``
directly instead of using this module.
"""

from __future__ import annotations

from python_ai.config.settings import get_settings


class DatabaseConfig:
    """Thin wrapper around ``DatabaseSettings``.

    Preserved for backward compatibility with existing callers.
    All attributes are read from the central ``get_settings().database``
    instance.
    """

    def __init__(self) -> None:
        """Initialise from central settings."""
        db = get_settings().database
        self.host: str = db.host
        self.port: int = db.port
        self.user: str = db.user
        self.password: str = db.password
        self.database: str = db.name
        self.pool_size: int = db.pool_size
        self.max_overflow: int = db.max_overflow
        self.pool_timeout: int = db.pool_timeout
        self.pool_recycle: int = db.pool_recycle
        self.echo: bool = db.echo_sql

    @property
    def sync_url(self) -> str:
        """Generate synchronous database connection URL."""
        return get_settings().database.url

    @property
    def async_url(self) -> str:
        """Generate async database connection URL."""
        return get_settings().database.async_url

    @property
    def test_url(self) -> str:
        """Generate SQLite URL for testing."""
        return get_settings().database.test_url

    @property
    def async_test_url(self) -> str:
        """Generate async SQLite URL for testing."""
        return get_settings().database.async_test_url


# Singleton ────────────────────────────────────────────────────

_config: DatabaseConfig | None = None


def get_database_config() -> DatabaseConfig:
    """Get or create the global database configuration.

    Returns:
        DatabaseConfig backed by central settings.
    """
    global _config
    if _config is None:
        _config = DatabaseConfig()
    return _config


def get_database_url(async_mode: bool = False, testing: bool = False) -> str:
    """Get the appropriate database URL.

    Args:
        async_mode: If True, return async-compatible URL.
        testing: If True, use SQLite test database.

    Returns:
        str: Database connection URL.
    """
    config = get_database_config()
    if testing:
        return config.async_test_url if async_mode else config.test_url
    return config.async_url if async_mode else config.sync_url


def reset_config() -> None:
    """Reset configuration (useful for testing)."""
    global _config
    _config = None
