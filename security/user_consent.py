"""User consent and data access control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class ConsentRecord:
    """Consent state for one user."""

    user_id: str
    granted: bool
    updated_at: datetime
    scope: str = "trading_data"


class ConsentManager:
    """In-memory consent manager for API-layer authorization checks."""

    def __init__(self) -> None:
        self._records: Dict[str, ConsentRecord] = {}

    def set_consent(self, user_id: str, granted: bool, scope: str) -> None:
        """Set or update user consent status."""
        self._records[user_id] = ConsentRecord(
            user_id=user_id,
            granted=granted,
            updated_at=datetime.now(timezone.utc),
            scope=scope,
        )

    def has_consent(self, user_id: str, scope: Optional[str] = None) -> bool:
        """Return whether user has active consent (optionally for scope)."""
        record = self._records.get(user_id)
        if record is None or not record.granted:
            return False
        if scope is not None and record.scope != scope:
            return False
        return True

    def revoke(self, user_id: str) -> None:
        """Revoke user consent."""
        if user_id in self._records:
            record = self._records[user_id]
            self._records[user_id] = ConsentRecord(
                user_id=record.user_id,
                granted=False,
                updated_at=datetime.now(timezone.utc),
                scope=record.scope,
            )
