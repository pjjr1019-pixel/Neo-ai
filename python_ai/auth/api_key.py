"""
API key authentication for NEO Hybrid AI.

Provides API key generation, validation, and management
for service-to-service authentication.
"""

import hashlib
import os
import secrets
from datetime import datetime
from typing import Dict, List, Optional

from python_ai.auth.models import APIKey, UserRole


class APIKeyManager:
    """Manager for API key operations.

    Handles API key generation, validation, and storage.
    Keys are stored in memory by default but can be extended
    to use database storage.

    Attributes:
        _keys: In-memory storage of API keys.
    """

    _instance: Optional["APIKeyManager"] = None
    _keys: Dict[str, APIKey] = {}

    def __new__(cls) -> "APIKeyManager":
        """Singleton pattern for API key manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._keys = {}
        return cls._instance

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for secure storage.

        Args:
            key: Plain text API key.

        Returns:
            str: SHA-256 hash of the key.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def _generate_key() -> str:
        """Generate a new secure API key.

        Returns:
            str: URL-safe random API key.
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def _generate_key_id() -> str:
        """Generate a unique key identifier.

        Returns:
            str: Short unique identifier.
        """
        return secrets.token_hex(8)

    def create_key(
        self,
        name: str,
        roles: Optional[List[UserRole]] = None,
        expires_at: Optional[datetime] = None,
    ) -> tuple:
        """Create a new API key.

        Args:
            name: Human-readable name for the key.
            roles: Roles to associate with the key.
            expires_at: Optional expiration timestamp.

        Returns:
            tuple: (plain_key, APIKey object)
        """
        plain_key = self._generate_key()
        key_id = self._generate_key_id()
        key_hash = self._hash_key(plain_key)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            roles=roles or [UserRole.SERVICE],
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key

        # Return full key as key_id.plain_key for easier identification
        full_key = f"{key_id}.{plain_key}"
        return full_key, api_key

    def validate_key(self, full_key: str) -> Optional[APIKey]:
        """Validate an API key.

        Args:
            full_key: Full API key in format key_id.secret.

        Returns:
            Optional[APIKey]: API key object if valid, None otherwise.
        """
        try:
            key_id, secret = full_key.split(".", 1)
        except ValueError:
            return None

        api_key = self._keys.get(key_id)
        if api_key is None:
            return None

        # Check if key is enabled
        if not api_key.enabled:
            return None

        # Check expiration
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return None

        # Verify hash
        if self._hash_key(secret) != api_key.key_hash:
            return None

        # Update last used timestamp
        api_key.last_used = datetime.utcnow()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: Key identifier to revoke.

        Returns:
            bool: True if key was revoked, False if not found.
        """
        api_key = self._keys.get(key_id)
        if api_key is None:
            return False

        api_key.enabled = False
        return True

    def delete_key(self, key_id: str) -> bool:
        """Delete an API key.

        Args:
            key_id: Key identifier to delete.

        Returns:
            bool: True if key was deleted, False if not found.
        """
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False

    def list_keys(self) -> List[APIKey]:
        """List all API keys.

        Returns:
            List[APIKey]: All registered API keys.
        """
        return list(self._keys.values())

    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get an API key by ID.

        Args:
            key_id: Key identifier.

        Returns:
            Optional[APIKey]: API key if found.
        """
        return self._keys.get(key_id)

    @classmethod
    def reset(cls) -> None:
        """Reset the manager (for testing)."""
        cls._instance = None
        cls._keys = {}


def get_api_key_manager() -> APIKeyManager:
    """Get the API key manager singleton.

    Returns:
        APIKeyManager: The singleton manager instance.
    """
    return APIKeyManager()


def validate_api_key(key: str) -> Optional[APIKey]:
    """Validate an API key.

    Args:
        key: Full API key string.

    Returns:
        Optional[APIKey]: API key if valid, None otherwise.
    """
    manager = get_api_key_manager()
    return manager.validate_key(key)


# Load API keys from environment on startup
def load_default_api_keys() -> None:
    """Load default API keys from environment variables.

    Looks for NEO_API_KEY_* environment variables and creates
    API keys for each one found.
    """
    manager = get_api_key_manager()

    # Check for default service API key
    default_key = os.getenv("NEO_DEFAULT_API_KEY")
    if default_key:
        key_id = "default"
        key_hash = APIKeyManager._hash_key(default_key)
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name="Default Service Key",
            roles=[UserRole.SERVICE],
        )
        manager._keys[key_id] = api_key
