"""
JWT token handling for NEO Hybrid AI.

Provides JWT token creation, validation, and decoding
using PyJWT with RS256 or HS256 algorithms.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from python_ai.auth.models import TokenData


class JWTConfig:
    """Configuration for JWT operations.

    Attributes:
        secret_key: Secret key for HS256 signing.
        algorithm: JWT signing algorithm.
        access_token_expire_minutes: Access token lifetime.
        refresh_token_expire_days: Refresh token lifetime.
    """

    def __init__(self) -> None:
        """Initialize JWT configuration from environment."""
        self.secret_key = os.getenv(
            "JWT_SECRET_KEY", "your-secret-key-change-in-production"
        )
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(
            os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
        )
        self.refresh_token_expire_days = int(
            os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")
        )


# Password hashing context - use sha256_crypt for compatibility
# In production, consider using argon2 or bcrypt with compatible versions
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# Global config instance
_jwt_config: Optional[JWTConfig] = None


def get_jwt_config() -> JWTConfig:
    """Get JWT configuration singleton."""
    global _jwt_config
    if _jwt_config is None:
        _jwt_config = JWTConfig()
    return _jwt_config


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.

    Args:
        plain_password: Plain text password.
        hashed_password: Bcrypt hashed password.

    Returns:
        bool: True if password matches hash.
    """
    return bool(pwd_context.verify(plain_password, hashed_password))


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password.

    Returns:
        str: Bcrypt hashed password.
    """
    return str(pwd_context.hash(password))


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token.

    Args:
        data: Payload data to encode.
        expires_delta: Custom expiration time.

    Returns:
        str: Encoded JWT token.
    """
    config = get_jwt_config()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=config.access_token_expire_minutes
        )

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }
    )

    encoded = jwt.encode(
        to_encode, config.secret_key, algorithm=config.algorithm
    )
    return str(encoded)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT refresh token.

    Args:
        data: Payload data to encode.
        expires_delta: Custom expiration time.

    Returns:
        str: Encoded JWT refresh token.
    """
    config = get_jwt_config()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=config.refresh_token_expire_days
        )

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }
    )

    encoded = jwt.encode(
        to_encode, config.secret_key, algorithm=config.algorithm
    )
    return str(encoded)


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token.

    Args:
        token: JWT token string.

    Returns:
        Optional[TokenData]: Decoded token data or None if invalid.
    """
    config = get_jwt_config()

    try:
        payload = jwt.decode(
            token, config.secret_key, algorithms=[config.algorithm]
        )
        username: Optional[str] = payload.get("sub")
        if username is None:
            return None

        roles: List[str] = payload.get("roles", [])
        token_type: str = payload.get("type", "access")
        exp_timestamp = payload.get("exp")

        exp = None
        if exp_timestamp:
            exp = datetime.fromtimestamp(exp_timestamp)

        return TokenData(
            username=username,
            roles=roles,
            exp=exp,
            token_type=token_type,
        )
    except JWTError:
        return None


class JWTHandler:
    """High-level JWT handler for authentication operations.

    Provides methods for token creation, validation, and user
    authentication with password verification.
    """

    def __init__(self) -> None:
        """Initialize JWT handler."""
        self.config = get_jwt_config()

    def create_tokens(
        self,
        username: str,
        roles: List[str],
    ) -> Dict[str, str]:
        """Create both access and refresh tokens.

        Args:
            username: User's username.
            roles: User's role list.

        Returns:
            Dict with access_token and refresh_token.
        """
        data = {"sub": username, "roles": roles}

        return {
            "access_token": create_access_token(data),
            "refresh_token": create_refresh_token(data),
            "token_type": "bearer",
        }

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token.

        Args:
            refresh_token: Valid refresh token.

        Returns:
            Optional[str]: New access token or None if refresh invalid.
        """
        token_data = decode_token(refresh_token)
        if token_data is None:
            return None
        if token_data.token_type != "refresh":
            return None

        data = {"sub": token_data.username, "roles": token_data.roles}
        return create_access_token(data)

    def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate a token and return its data.

        Args:
            token: JWT token to validate.

        Returns:
            Optional[TokenData]: Token data if valid, None otherwise.
        """
        return decode_token(token)

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password.

        Args:
            password: Plain text password.

        Returns:
            str: Hashed password.
        """
        return get_password_hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password.
            hashed: Hashed password.

        Returns:
            bool: True if password is correct.
        """
        return verify_password(plain_password, hashed)
