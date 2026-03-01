"""
Tests for authentication module.

Covers JWT handling, API key management, and authentication dependencies.
"""

from datetime import datetime, timedelta

import pytest


class TestUserRole:
    """Tests for UserRole enum."""

    def test_user_roles_exist(self):
        """Test that all expected roles exist."""
        from python_ai.auth.models import UserRole

        assert UserRole.ADMIN == "admin"
        assert UserRole.DEVELOPER == "developer"
        assert UserRole.VIEWER == "viewer"
        assert UserRole.SERVICE == "service"


class TestUserModel:
    """Tests for User model."""

    def test_user_creation(self):
        """Test creating a user with defaults."""
        from python_ai.auth.models import User, UserRole

        user = User(username="testuser")
        assert user.username == "testuser"
        assert user.email is None
        assert user.disabled is False
        assert UserRole.VIEWER in user.roles

    def test_user_with_all_fields(self):
        """Test creating a user with all fields."""
        from python_ai.auth.models import User, UserRole

        user = User(
            username="admin",
            email="admin@test.com",
            full_name="Admin User",
            roles=[UserRole.ADMIN],
            disabled=False,
        )
        assert user.username == "admin"
        assert user.email == "admin@test.com"
        assert UserRole.ADMIN in user.roles


class TestTokenModels:
    """Tests for Token models."""

    def test_token_data_creation(self):
        """Test creating token data."""
        from python_ai.auth.models import TokenData

        data = TokenData(
            username="user1",
            roles=["admin"],
            token_type="access",
        )
        assert data.username == "user1"
        assert "admin" in data.roles
        assert data.token_type == "access"

    def test_token_response(self):
        """Test Token response model."""
        from python_ai.auth.models import Token

        token = Token(
            access_token="abc123",
            refresh_token="def456",
            expires_in=3600,
        )
        assert token.access_token == "abc123"
        assert token.token_type == "bearer"


class TestAPIKeyModel:
    """Tests for APIKey model."""

    def test_api_key_creation(self):
        """Test creating an API key model."""
        from python_ai.auth.models import APIKey, UserRole

        key = APIKey(
            key_id="test123",
            key_hash="hash123",
            name="Test Key",
        )
        assert key.key_id == "test123"
        assert key.enabled is True
        assert UserRole.SERVICE in key.roles


class TestJWTHandler:
    """Tests for JWT token handling."""

    def test_create_access_token(self):
        """Test creating an access token."""
        from python_ai.auth.jwt_handler import create_access_token

        token = create_access_token({"sub": "testuser", "roles": ["admin"]})
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test creating a refresh token."""
        from python_ai.auth.jwt_handler import create_refresh_token

        token = create_refresh_token({"sub": "testuser", "roles": ["viewer"]})
        assert token is not None
        assert isinstance(token, str)

    def test_decode_valid_token(self):
        """Test decoding a valid token."""
        from python_ai.auth.jwt_handler import (
            create_access_token,
            decode_token,
        )

        token = create_access_token({"sub": "testuser", "roles": ["admin"]})
        data = decode_token(token)
        assert data is not None
        assert data.username == "testuser"
        assert "admin" in data.roles
        assert data.token_type == "access"

    def test_decode_refresh_token(self):
        """Test decoding a refresh token."""
        from python_ai.auth.jwt_handler import (
            create_refresh_token,
            decode_token,
        )

        token = create_refresh_token({"sub": "testuser", "roles": []})
        data = decode_token(token)
        assert data is not None
        assert data.token_type == "refresh"

    def test_decode_invalid_token(self):
        """Test decoding an invalid token returns None."""
        from python_ai.auth.jwt_handler import decode_token

        data = decode_token("invalid.token.here")
        assert data is None

    def test_decode_expired_token(self):
        """Test that expired tokens are rejected."""
        from python_ai.auth.jwt_handler import (
            create_access_token,
            decode_token,
        )

        # Create token that's already expired
        token = create_access_token(
            {"sub": "testuser", "roles": []},
            expires_delta=timedelta(seconds=-10),
        )
        data = decode_token(token)
        assert data is None

    def test_password_hashing(self):
        """Test password hashing and verification."""
        from python_ai.auth.jwt_handler import (
            get_password_hash,
            verify_password,
        )

        password = "testpassword123"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False

    def test_jwt_handler_class(self):
        """Test JWTHandler class methods."""
        from python_ai.auth.jwt_handler import JWTHandler

        handler = JWTHandler()
        tokens = handler.create_tokens("user1", ["admin"])

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

    def test_refresh_access_token(self):
        """Test refreshing an access token."""
        from python_ai.auth.jwt_handler import JWTHandler

        handler = JWTHandler()
        tokens = handler.create_tokens("user1", ["admin"])

        new_access = handler.refresh_access_token(tokens["refresh_token"])
        assert new_access is not None

        # Verify new token is valid
        data = handler.validate_token(new_access)
        assert data is not None
        assert data.username == "user1"

    def test_refresh_with_access_token_fails(self):
        """Test that using access token as refresh fails."""
        from python_ai.auth.jwt_handler import JWTHandler

        handler = JWTHandler()
        tokens = handler.create_tokens("user1", ["admin"])

        # Should fail because access token has type "access", not "refresh"
        new_access = handler.refresh_access_token(tokens["access_token"])
        assert new_access is None


class TestAPIKeyManager:
    """Tests for API key management."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset API key manager before each test."""
        from python_ai.auth.api_key import APIKeyManager

        APIKeyManager.reset()
        yield
        APIKeyManager.reset()

    def test_create_api_key(self):
        """Test creating a new API key."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        full_key, api_key = manager.create_key("Test Key")

        assert full_key is not None
        assert "." in full_key  # Format: key_id.secret
        assert api_key.name == "Test Key"
        assert api_key.enabled is True

    def test_validate_api_key(self):
        """Test validating an API key."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        full_key, _ = manager.create_key("Test Key")

        validated = manager.validate_key(full_key)
        assert validated is not None
        assert validated.name == "Test Key"

    def test_validate_invalid_key(self):
        """Test validating an invalid key returns None."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        validated = manager.validate_key("invalid.key")
        assert validated is None

    def test_validate_malformed_key(self):
        """Test validating a key without separator."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        validated = manager.validate_key("nokeyidseparator")
        assert validated is None

    def test_revoke_api_key(self):
        """Test revoking an API key."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        full_key, api_key = manager.create_key("Test Key")

        assert manager.revoke_key(api_key.key_id) is True

        # Key should no longer validate
        validated = manager.validate_key(full_key)
        assert validated is None

    def test_delete_api_key(self):
        """Test deleting an API key."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        _, api_key = manager.create_key("Test Key")

        assert manager.delete_key(api_key.key_id) is True
        assert manager.get_key(api_key.key_id) is None

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        assert manager.delete_key("nonexistent") is False

    def test_list_keys(self):
        """Test listing all API keys."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        manager.create_key("Key 1")
        manager.create_key("Key 2")

        keys = manager.list_keys()
        assert len(keys) == 2

    def test_expired_key_validation(self):
        """Test that expired keys don't validate."""
        from python_ai.auth.api_key import get_api_key_manager

        manager = get_api_key_manager()
        past = datetime.utcnow() - timedelta(days=1)
        full_key, _ = manager.create_key("Expired Key", expires_at=past)

        validated = manager.validate_key(full_key)
        assert validated is None

    def test_validate_api_key_function(self):
        """Test the convenience validate_api_key function."""
        from python_ai.auth.api_key import (
            get_api_key_manager,
            validate_api_key,
        )

        manager = get_api_key_manager()
        full_key, _ = manager.create_key("Test Key")

        validated = validate_api_key(full_key)
        assert validated is not None


class TestAuthDependencies:
    """Tests for FastAPI authentication dependencies."""

    def test_get_user_exists(self):
        """Test getting an existing user."""
        from python_ai.auth.dependencies import get_user

        user = get_user("admin")
        assert user is not None
        assert user.username == "admin"

    def test_get_user_not_exists(self):
        """Test getting a non-existent user."""
        from python_ai.auth.dependencies import get_user

        user = get_user("nonexistent")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_current_user_no_auth(self):
        """Test that missing auth raises 401."""
        from fastapi import HTTPException

        from python_ai.auth.dependencies import get_current_user

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None, None, None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_with_token(self):
        """Test auth with valid JWT token."""
        from python_ai.auth.dependencies import get_current_user
        from python_ai.auth.jwt_handler import create_access_token

        token = create_access_token({"sub": "admin", "roles": ["admin"]})
        user = await get_current_user(token, None, None)

        assert user is not None
        assert user.username == "admin"

    @pytest.mark.asyncio
    async def test_get_current_active_user_disabled(self):
        """Test that disabled users are rejected."""
        from fastapi import HTTPException

        from python_ai.auth.dependencies import get_current_active_user
        from python_ai.auth.models import User, UserRole

        # Create a disabled user
        disabled_user = User(
            username="disabled",
            roles=[UserRole.VIEWER],
            disabled=True,
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(disabled_user)
        assert exc_info.value.status_code == 403

    def test_require_role_factory(self):
        """Test the require_role dependency factory."""
        from python_ai.auth.dependencies import require_role
        from python_ai.auth.models import UserRole

        checker = require_role([UserRole.ADMIN])
        assert callable(checker)

    @pytest.mark.asyncio
    async def test_admin_has_all_permissions(self):
        """Test that admin role bypasses role checks."""
        from python_ai.auth.dependencies import require_role
        from python_ai.auth.models import User, UserRole

        checker = require_role([UserRole.DEVELOPER])
        admin_user = User(username="admin", roles=[UserRole.ADMIN])

        result = await checker(admin_user)
        assert result.username == "admin"

    @pytest.mark.asyncio
    async def test_role_check_insufficient_permissions(self):
        """Test that users without role are rejected."""
        from fastapi import HTTPException

        from python_ai.auth.dependencies import require_role
        from python_ai.auth.models import User, UserRole

        checker = require_role([UserRole.ADMIN])
        viewer_user = User(username="viewer", roles=[UserRole.VIEWER])

        with pytest.raises(HTTPException) as exc_info:
            await checker(viewer_user)
        assert exc_info.value.status_code == 403


class TestPackageImports:
    """Tests for auth package imports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from python_ai.auth import (
            APIKey,
            APIKeyManager,
            JWTHandler,
            Token,
            TokenData,
            User,
            create_access_token,
            create_refresh_token,
            decode_token,
            get_current_active_user,
            get_current_user,
            oauth2_scheme,
            require_role,
            validate_api_key,
        )

        assert JWTHandler is not None
        assert create_access_token is not None
        assert create_refresh_token is not None
        assert decode_token is not None
        assert APIKeyManager is not None
        assert validate_api_key is not None
        assert get_current_user is not None
        assert get_current_active_user is not None
        assert require_role is not None
        assert oauth2_scheme is not None
        assert User is not None
        assert TokenData is not None
        assert Token is not None
        assert APIKey is not None
