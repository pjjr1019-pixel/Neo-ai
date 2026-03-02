"""
Database configuration for NEO Hybrid AI.

Handles environment-based configuration for database connections.
Supports PostgreSQL with connection pooling settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatabaseConfig:
    """Configuration settings for database connection.

    Attributes:
        host: Database server hostname.
        port: Database server port.
        user: Database username.
        password: Database password.
        database: Database name.
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections beyond pool_size.
        pool_timeout: Seconds to wait for a connection from pool.
        pool_recycle: Seconds after which to recycle connections.
        echo: Whether to log SQL statements.
    """

    host: str = field(
        default_factory=lambda: os.getenv("DB_HOST", "localhost")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("DB_PORT", "5432"))
    )
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "neoai"))
    password: str = field(
        default_factory=lambda: os.getenv("DB_PASSWORD", "neoai123")
    )
    database: str = field(
        default_factory=lambda: os.getenv("DB_NAME", "neoai_db")
    )
    pool_size: int = field(
        default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5"))
    )
    max_overflow: int = field(
        default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10"))
    )
    pool_timeout: int = field(
        default_factory=lambda: int(os.getenv("DB_POOL_TIMEOUT", "30"))
    )
    pool_recycle: int = field(
        default_factory=lambda: int(os.getenv("DB_POOL_RECYCLE", "1800"))
    )
    echo: bool = field(
        default_factory=lambda: os.getenv("DB_ECHO", "false").lower() == "true"
    )

    @property
    def sync_url(self) -> str:
        """Generate synchronous PostgreSQL connection URL."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def async_url(self) -> str:
        """Generate async PostgreSQL connection URL for asyncpg."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def test_url(self) -> str:
        """Generate SQLite URL for testing."""
        return "sqlite:///./test.db"

    @property
    def async_test_url(self) -> str:
        """Generate async SQLite URL for testing."""
        return "sqlite+aiosqlite:///./test.db"


# Global configuration instance
_config: Optional[DatabaseConfig] = None


def get_database_config() -> DatabaseConfig:
    """Get or create the global database configuration.

    Returns:
        DatabaseConfig: The singleton configuration instance.
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
