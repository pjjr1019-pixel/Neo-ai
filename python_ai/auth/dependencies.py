"""
FastAPI authentication dependencies for NEO Hybrid AI.

Provides dependency injection functions for authentication
and authorization in FastAPI routes.
"""

from typing import Callable, List, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)

from python_ai.auth.api_key import validate_api_key
from python_ai.auth.jwt_handler import decode_token
from python_ai.auth.models import User, UserRole

# OAuth2 scheme for JWT authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    auto_error=False,
)

# HTTP Bearer scheme for direct token auth
bearer_scheme = HTTPBearer(auto_error=False)

# API Key header scheme
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


# In-memory user store (replace with database in production)
_users_db: dict = {
    "admin": {
        "username": "admin",
        "email": "admin@neo.ai",
        "full_name": "Admin User",
        "roles": [UserRole.ADMIN],
        "disabled": False,
        "hashed_password": (
            # Default password: "admin123" - CHANGE IN PRODUCTION
            "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4"
            ".FVttYxVE9KW3QG"
        ),
    },
}


def get_user(username: str) -> Optional[User]:
    """Get a user from the user store.

    Args:
        username: Username to look up.

    Returns:
        Optional[User]: User if found, None otherwise.
    """
    user_data = _users_db.get(username)
    if user_data:
        return User(**user_data)
    return None


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> User:
    """Get the current authenticated user.

    Supports multiple authentication methods:
    1. OAuth2 bearer token
    2. Direct JWT bearer token
    3. API key

    Args:
        token: OAuth2 bearer token.
        bearer: HTTP bearer credentials.
        api_key: API key from header.

    Returns:
        User: Authenticated user.

    Raises:
        HTTPException: If no valid authentication provided.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Try OAuth2 token first
    if token:
        token_data = decode_token(token)
        if token_data and token_data.username:
            user = get_user(token_data.username)
            if user:
                return user

    # Try direct bearer token
    if bearer and bearer.credentials:
        token_data = decode_token(bearer.credentials)
        if token_data and token_data.username:
            user = get_user(token_data.username)
            if user:
                return user

    # Try API key
    if api_key:
        api_key_obj = validate_api_key(api_key)
        if api_key_obj:
            # Create a service user for API key auth
            return User(
                username=f"service:{api_key_obj.name}",
                roles=api_key_obj.roles,
            )

    raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active (non-disabled) user.

    Args:
        current_user: Authenticated user from get_current_user.

    Returns:
        User: Active authenticated user.

    Raises:
        HTTPException: If user is disabled.
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return current_user


def require_role(required_roles: List[UserRole]) -> Callable:
    """Create a dependency that requires specific roles.

    Args:
        required_roles: List of roles, user must have at least one.

    Returns:
        Callable: Dependency function for FastAPI.
    """

    async def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        """Check if user has required role.

        Args:
            current_user: Authenticated active user.

        Returns:
            User: User if authorized.

        Raises:
            HTTPException: If user lacks required role.
        """
        user_roles = set(current_user.roles)
        required_set = set(required_roles)

        # Admin has all permissions
        if UserRole.ADMIN in user_roles:
            return current_user

        # Check if user has any required role
        if not user_roles.intersection(required_set):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )

        return current_user

    return role_checker


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[User]:
    """Get the current user if authenticated, None otherwise.

    This allows endpoints to be accessible both with and without
    authentication, with different behavior based on auth status.

    Args:
        token: OAuth2 bearer token.
        bearer: HTTP bearer credentials.
        api_key: API key from header.

    Returns:
        Optional[User]: User if authenticated, None otherwise.
    """
    try:
        return await get_current_user(token, bearer, api_key)
    except HTTPException:
        return None


# Role-specific dependencies for common use cases
require_admin = require_role([UserRole.ADMIN])
require_developer = require_role([UserRole.ADMIN, UserRole.DEVELOPER])
require_viewer = require_role(
    [UserRole.ADMIN, UserRole.DEVELOPER, UserRole.VIEWER]
)
