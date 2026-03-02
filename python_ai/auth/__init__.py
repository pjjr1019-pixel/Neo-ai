"""
Authentication package for NEO Hybrid AI.

Provides JWT authentication, API key validation, and
role-based access control for FastAPI endpoints.
"""

from python_ai.auth.api_key import (
    APIKeyManager,
    validate_api_key,
)
from python_ai.auth.dependencies import (
    get_current_active_user,
    get_current_user,
    oauth2_scheme,
    require_role,
)
from python_ai.auth.jwt_handler import (
    JWTHandler,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from python_ai.auth.models import (
    APIKey,
    Token,
    TokenData,
    User,
)

__all__ = [
    "JWTHandler",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "APIKeyManager",
    "validate_api_key",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "oauth2_scheme",
    "User",
    "TokenData",
    "Token",
    "APIKey",
]
