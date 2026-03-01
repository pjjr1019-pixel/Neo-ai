"""Application settings using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings.

    Attributes:
        driver: Database driver (postgresql, sqlite, etc.).
        host: Database server host.
        port: Database server port.
        name: Database name.
        user: Database username.
        password: Database password.
        pool_size: Connection pool size.
        max_overflow: Max connections above pool_size.
        pool_timeout: Connection wait timeout in seconds.
        echo_sql: Whether to log SQL statements.
    """

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        extra="ignore",
    )

    driver: str = Field(default="postgresql", description="Database driver")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="neo_ai", description="Database name")
    user: str = Field(default="neo", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout")
    echo_sql: bool = Field(default=False, description="Log SQL")

    @property
    def url(self) -> str:
        """Get database connection URL."""
        if self.driver == "sqlite":
            return f"sqlite:///{self.name}.db"
        return (
            f"{self.driver}://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def async_url(self) -> str:
        """Get async database connection URL."""
        if self.driver == "sqlite":
            return f"sqlite+aiosqlite:///{self.name}.db"
        return (
            f"{self.driver}+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class AuthSettings(BaseSettings):
    """Authentication configuration settings.

    Attributes:
        secret_key: JWT signing secret key.
        algorithm: JWT signing algorithm.
        access_token_expire_minutes: Access token lifetime.
        refresh_token_expire_days: Refresh token lifetime.
        api_key_prefix: Prefix for API keys.
        password_min_length: Minimum password length.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUTH_",
        extra="ignore",
    )

    secret_key: str = Field(
        default="change-me-in-production",
        description="JWT secret key",
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiry",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiry",
    )
    api_key_prefix: str = Field(default="neo_", description="API key prefix")
    password_min_length: int = Field(
        default=8,
        description="Min password length",
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Warn if using default secret key."""
        if v == "change-me-in-production":
            import warnings

            warnings.warn(
                "Using default AUTH_SECRET_KEY. "
                "Set a secure key in production!",
                UserWarning,
                stacklevel=2,
            )
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration settings.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log format (json, console, plain).
        log_dir: Directory for log files.
        enable_file_logging: Whether to write to log files.
        max_file_size_mb: Max log file size in MB.
        backup_count: Number of backup log files.
    """

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        extra="ignore",
    )

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="console", description="Log format")
    log_dir: str = Field(default="logs", description="Log directory")
    enable_file_logging: bool = Field(
        default=False,
        description="Enable file logging",
    )
    max_file_size_mb: int = Field(default=50, description="Max file size")
    backup_count: int = Field(default=10, description="Backup count")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


class ModelSettings(BaseSettings):
    """ML model configuration settings.

    Attributes:
        models_dir: Directory for saved models.
        default_model_type: Default model architecture.
        max_training_epochs: Maximum training epochs.
        early_stopping_patience: Early stopping patience.
        validation_split: Validation data fraction.
        random_seed: Random seed for reproducibility.
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        extra="ignore",
    )

    models_dir: str = Field(
        default="models",
        description="Models directory",
    )
    default_model_type: str = Field(
        default="transformer",
        description="Default model type",
    )
    max_training_epochs: int = Field(
        default=100,
        description="Max epochs",
    )
    early_stopping_patience: int = Field(
        default=10,
        description="Early stopping patience",
    )
    validation_split: float = Field(
        default=0.2,
        description="Validation split",
    )
    random_seed: int = Field(default=42, description="Random seed")

    @field_validator("validation_split")
    @classmethod
    def validate_split(cls, v: float) -> float:
        """Validate validation split range."""
        if not 0 < v < 1:
            raise ValueError("validation_split must be between 0 and 1")
        return v


class APISettings(BaseSettings):
    """API server configuration settings.

    Attributes:
        host: Server bind host.
        port: Server bind port.
        debug: Enable debug mode.
        reload: Enable auto-reload.
        cors_origins: Allowed CORS origins.
        rate_limit_per_minute: Rate limit requests per minute.
        max_request_size_mb: Maximum request body size.
    """

    model_config = SettingsConfigDict(
        env_prefix="API_",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="CORS origins",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Rate limit",
    )
    max_request_size_mb: int = Field(
        default=10,
        description="Max request size",
    )


class Settings(BaseSettings):
    """Main application settings.

    Aggregates all configuration subsections and provides
    application-wide settings.

    Attributes:
        app_name: Application name.
        environment: Deployment environment (dev, staging, prod).
        version: Application version.
        database: Database settings.
        auth: Authentication settings.
        logging: Logging settings.
        model: Model settings.
        api: API settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    app_name: str = Field(default="neo-ai", description="Application name")
    environment: str = Field(
        default="development",
        description="Environment",
    )
    version: str = Field(default="0.1.0", description="Version")

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    api: APISettings = Field(default_factory=APISettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        valid = {"development", "staging", "production", "testing"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid environment: {v}")
        return v.lower()

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"


# Cached settings instance
_settings: Optional[Settings] = None


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings instance loaded from environment.
    """
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment.

    Clears the cache and reloads settings.

    Returns:
        Fresh Settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
