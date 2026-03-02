"""Tests for configuration management."""

import os
from unittest.mock import patch

from python_ai.config.settings import (
    APISettings,
    AuthSettings,
    DatabaseSettings,
    LoggingSettings,
    ModelSettings,
    Settings,
    get_settings,
    reload_settings,
)


class TestDatabaseSettings:
    """Tests for DatabaseSettings."""

    def test_default_values(self):
        """Test default database settings."""
        settings = DatabaseSettings()
        assert settings.driver == "postgresql"
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.pool_size == 5

    def test_url_property(self):
        """Test database URL generation."""
        settings = DatabaseSettings(
            driver="postgresql",
            user="testuser",
            password="testpass",
            host="dbhost",
            port=5433,
            name="testdb",
        )
        assert "testuser" in settings.url
        assert "testpass" in settings.url
        assert "dbhost:5433" in settings.url
        assert "testdb" in settings.url

    def test_sqlite_url(self):
        """Test SQLite URL generation."""
        settings = DatabaseSettings(driver="sqlite", name="test")
        assert settings.url == "sqlite:///test.db"
        assert "aiosqlite" in settings.async_url

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"DB_HOST": "envhost", "DB_PORT": "5555"},
        ):
            settings = DatabaseSettings()
            assert settings.host == "envhost"
            assert settings.port == 5555


class TestAuthSettings:
    """Tests for AuthSettings."""

    def test_default_values(self):
        """Test default auth settings."""
        settings = AuthSettings()
        assert settings.algorithm == "HS256"
        assert settings.access_token_expire_minutes == 30
        assert settings.refresh_token_expire_days == 7

    def test_secret_key_warning(self):
        """Test warning for default secret key."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AuthSettings(secret_key="change-me-in-production")
            assert len(w) == 1
            assert "AUTH_SECRET_KEY" in str(w[0].message)

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {"AUTH_SECRET_KEY": "my-secret", "AUTH_ALGORITHM": "HS512"},
        ):
            settings = AuthSettings()
            assert settings.secret_key == "my-secret"
            assert settings.algorithm == "HS512"


class TestLoggingSettings:
    """Tests for LoggingSettings."""

    def test_default_values(self):
        """Test default logging settings."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.format == "console"
        assert settings.enable_file_logging is False

    def test_level_validation(self):
        """Test log level validation."""
        settings = LoggingSettings(level="debug")
        assert settings.level == "DEBUG"

    def test_invalid_level(self):
        """Test invalid log level raises error."""
        import pytest

        with pytest.raises(ValueError):
            LoggingSettings(level="invalid")


class TestModelSettings:
    """Tests for ModelSettings."""

    def test_default_values(self):
        """Test default model settings."""
        settings = ModelSettings()
        assert settings.max_training_epochs == 100
        assert settings.validation_split == 0.2
        assert settings.random_seed == 42

    def test_validation_split_validation(self):
        """Test validation split must be 0-1."""
        import pytest

        with pytest.raises(ValueError):
            ModelSettings(validation_split=1.5)

        with pytest.raises(ValueError):
            ModelSettings(validation_split=0)


class TestAPISettings:
    """Tests for APISettings."""

    def test_default_values(self):
        """Test default API settings."""
        settings = APISettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.debug is False

    def test_cors_origins(self):
        """Test CORS origins default."""
        settings = APISettings()
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0


class TestSettings:
    """Tests for main Settings class."""

    def test_default_values(self):
        """Test default settings."""
        settings = Settings()
        assert settings.app_name == "neo-ai"
        assert settings.environment == "development"

    def test_nested_settings(self):
        """Test nested settings are initialized."""
        settings = Settings()
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.auth, AuthSettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.api, APISettings)

    def test_environment_validation(self):
        """Test environment validation."""
        import pytest

        with pytest.raises(ValueError):
            Settings(environment="invalid")

    def test_is_production(self):
        """Test is_production property."""
        settings = Settings(environment="production")
        assert settings.is_production is True
        assert settings.is_development is False

    def test_is_development(self):
        """Test is_development property."""
        settings = Settings(environment="development")
        assert settings.is_development is True
        assert settings.is_production is False

    def test_is_testing(self):
        """Test is_testing property."""
        settings = Settings(environment="testing")
        assert settings.is_testing is True


class TestGetSettings:
    """Tests for get_settings function."""

    def teardown_method(self):
        """Clear cache after each test."""
        get_settings.cache_clear()

    def test_returns_settings(self):
        """Test get_settings returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached(self):
        """Test settings are cached."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reload_clears_cache(self):
        """Test reload_settings clears cache."""
        settings1 = get_settings()
        settings2 = reload_settings()
        # New instance after reload
        assert settings1 is not settings2
