"""
Authentication data models for NEO Hybrid AI.

Defines Pydantic models for users, tokens, and API keys.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    SERVICE = "service"


class User(BaseModel):
    """User model for authentication.

    Attributes:
        username: Unique username.
        email: User email address.
        full_name: User's full name.
        roles: List of user roles.
        disabled: Whether user account is disabled.
        created_at: Account creation timestamp.
    """

    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    roles: List[UserRole] = Field(default_factory=lambda: [UserRole.VIEWER])
    disabled: bool = False
    created_at: Optional[datetime] = None


class UserInDB(User):
    """User model with hashed password for database storage."""

    hashed_password: str


class TokenData(BaseModel):
    """JWT token payload data.

    Attributes:
        username: Subject of the token.
        roles: User roles encoded in token.
        exp: Token expiration timestamp.
        token_type: Type of token (access or refresh).
    """

    username: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    exp: Optional[datetime] = None
    token_type: str = "access"


class Token(BaseModel):
    """Token response model.

    Attributes:
        access_token: JWT access token.
        refresh_token: JWT refresh token.
        token_type: Token type (bearer).
        expires_in: Seconds until token expiration.
    """

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 3600


class APIKey(BaseModel):
    """API key model.

    Attributes:
        key_id: Unique key identifier.
        key_hash: Hashed API key value.
        name: Human-readable name for the key.
        roles: Roles associated with this key.
        created_at: Key creation timestamp.
        expires_at: Optional expiration timestamp.
        last_used: Last usage timestamp.
        enabled: Whether the key is active.
    """

    key_id: str
    key_hash: str
    name: str
    roles: List[UserRole] = Field(default_factory=lambda: [UserRole.SERVICE])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    enabled: bool = True


class LoginRequest(BaseModel):
    """Login request model."""

    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str
