"""Tests for Phase-3 production config hardening.

Covers:
- DatabaseSettings.validate_password: trivial-password rejection
- Settings.validate_production_secrets: combined auth+db validation
"""

import os
from unittest.mock import patch

import pytest

from python_ai.config.settings import DatabaseSettings, Settings


class TestDatabasePasswordValidator:
    """Tests for DatabaseSettings.validate_password."""

    @pytest.mark.parametrize(
        "bad_pw",
        ["password", "postgres", "admin", ""],
    )
    def test_trivial_passwords_rejected_in_production(self, bad_pw):
        """Trivial passwords raise ValueError in production."""
        with patch.dict(os.environ, {"NEO_ENVIRONMENT": "production"}):
            with pytest.raises(ValueError, match="non-trivial"):
                DatabaseSettings(password=bad_pw)

    def test_strong_password_accepted_in_production(self):
        """Strong password passes validation in production."""
        with patch.dict(os.environ, {"NEO_ENVIRONMENT": "production"}):
            db = DatabaseSettings(password="Str0ng!Pr0d#Pass")
            assert db.password == "Str0ng!Pr0d#Pass"

    def test_trivial_passwords_allowed_in_dev(self):
        """Trivial passwords are accepted in development."""
        with patch.dict(
            os.environ,
            {"NEO_ENVIRONMENT": "development"},
            clear=False,
        ):
            db = DatabaseSettings(password="postgres")
            assert db.password == "postgres"


class TestProductionSecretsValidator:
    """Tests for Settings.validate_production_secrets."""

    def test_short_auth_key_rejected(self):
        """AUTH_SECRET_KEY < 16 chars is rejected in production."""
        with patch.dict(
            os.environ,
            {
                "AUTH_SECRET_KEY": "short",
                "DB_PASSWORD": "Str0ng!Pr0d#Pass",
            },
        ):
            with pytest.raises(ValueError, match="16 characters"):
                Settings(environment="production")

    def test_missing_db_password_rejected(self):
        """Empty DB_PASSWORD is rejected in production."""
        with patch.dict(
            os.environ,
            {"AUTH_SECRET_KEY": "a-very-long-secret-key-here"},
        ):
            with pytest.raises(ValueError, match="DB_PASSWORD"):
                Settings(environment="production")

    def test_valid_production_settings(self):
        """Valid secrets pass production validation."""
        with patch.dict(
            os.environ,
            {
                "AUTH_SECRET_KEY": "a-very-long-secret-key-here",
                "DB_PASSWORD": "Str0ng!Pr0d#Pass",
            },
        ):
            settings = Settings(environment="production")
            assert settings.is_production is True

    def test_dev_no_validation(self):
        """Development mode skips production checks."""
        settings = Settings(environment="development")
        assert settings.is_development is True
