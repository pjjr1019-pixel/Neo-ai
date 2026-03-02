"""
Password policy and account-lockout logic for NEO Hybrid AI.

Provides:
- ``PasswordPolicy``: configurable complexity rules.
- ``AccountLockout``: per-user failed-attempt tracking with
  automatic lockout and time-based reset.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

__all__ = [
    "AccountLockout",
    "LockoutStatus",
    "PasswordPolicy",
    "PasswordValidationResult",
]


# ── Password complexity ──────────────────────────────────────


@dataclass
class PasswordValidationResult:
    """Outcome of a password-policy check.

    Attributes:
        valid: ``True`` when all rules pass.
        errors: List of human-readable failure reasons.
    """

    valid: bool
    errors: List[str] = field(default_factory=list)


class PasswordPolicy:
    """Enforce configurable password complexity rules.

    Args:
        min_length: Minimum password length.
        require_uppercase: Require at least one uppercase letter.
        require_lowercase: Require at least one lowercase letter.
        require_digit: Require at least one digit.
        require_special: Require at least one special character.
        max_length: Maximum password length (prevent DoS).
    """

    def __init__(
        self,
        *,
        min_length: int = 8,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digit: bool = True,
        require_special: bool = True,
        max_length: int = 128,
    ) -> None:
        """Initialise with the given complexity constraints."""
        self.min_length = min_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digit = require_digit
        self.require_special = require_special
        self.max_length = max_length

    def validate(self, password: str) -> PasswordValidationResult:
        """Check *password* against all configured rules.

        Returns:
            PasswordValidationResult with ``valid`` flag and any
            ``errors``.
        """
        errors: List[str] = []

        if len(password) < self.min_length:
            errors.append(
                f"Password must be at least {self.min_length} characters"
            )
        if len(password) > self.max_length:
            errors.append(
                f"Password must be at most {self.max_length} characters"
            )
        if self.require_uppercase and not re.search(r"[A-Z]", password):
            errors.append(
                "Password must contain at least one uppercase letter"
            )
        if self.require_lowercase and not re.search(r"[a-z]", password):
            errors.append(
                "Password must contain at least one lowercase letter"
            )
        if self.require_digit and not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")
        if self.require_special and not re.search(
            r"[!@#$%^&*(),.?\":{}|<>\-_=+\[\]\\;'/`~]", password
        ):
            errors.append(
                "Password must contain at least one special character"
            )

        return PasswordValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )


# ── Account lockout ──────────────────────────────────────────


@dataclass
class LockoutStatus:
    """Snapshot of an account's lockout state.

    Attributes:
        locked: Whether the account is currently locked.
        attempts: Failed attempts since last reset.
        remaining: Attempts left before lockout.
        locked_until: Epoch timestamp when the lockout expires
                      (``None`` if not locked).
    """

    locked: bool
    attempts: int
    remaining: int
    locked_until: float | None = None


class AccountLockout:
    """Track failed login attempts and lock accounts.

    Thread-safe.  Uses an in-process dictionary keyed by
    username.  In production, back this with Redis or a DB
    for multi-worker support.

    Args:
        max_attempts: Failed attempts before lockout.
        lockout_seconds: Duration of lockout window.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        lockout_seconds: float = 300.0,
    ) -> None:
        """Initialise lockout tracker with the given limits."""
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        self._attempts: Dict[str, int] = {}
        self._locked_until: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────

    def record_failure(self, username: str) -> LockoutStatus:
        """Record a failed login attempt for *username*.

        Returns the resulting lockout status.
        """
        with self._lock:
            self._clear_if_expired(username)

            count = self._attempts.get(username, 0) + 1
            self._attempts[username] = count

            if count >= self.max_attempts:
                until = time.time() + self.lockout_seconds
                self._locked_until[username] = until
                logger.warning(
                    "Account '%s' locked for %.0fs after %d failures",
                    username,
                    self.lockout_seconds,
                    count,
                )
                return LockoutStatus(
                    locked=True,
                    attempts=count,
                    remaining=0,
                    locked_until=until,
                )

            return LockoutStatus(
                locked=False,
                attempts=count,
                remaining=self.max_attempts - count,
            )

    def record_success(self, username: str) -> None:
        """Clear failed attempts after a successful login."""
        with self._lock:
            self._attempts.pop(username, None)
            self._locked_until.pop(username, None)

    def is_locked(self, username: str) -> bool:
        """Return ``True`` if *username* is currently locked out."""
        with self._lock:
            self._clear_if_expired(username)
            return username in self._locked_until

    def status(self, username: str) -> LockoutStatus:
        """Return full lockout status for *username*."""
        with self._lock:
            self._clear_if_expired(username)
            attempts = self._attempts.get(username, 0)
            until = self._locked_until.get(username)
            locked = until is not None
            remaining = max(0, self.max_attempts - attempts)
            return LockoutStatus(
                locked=locked,
                attempts=attempts,
                remaining=remaining,
                locked_until=until,
            )

    def reset(self, username: str) -> None:
        """Admin reset — immediately unlock *username*."""
        with self._lock:
            self._attempts.pop(username, None)
            self._locked_until.pop(username, None)

    # ── Internal ──────────────────────────────────────────────

    def _clear_if_expired(self, username: str) -> None:
        """Release the lock if the lockout window has elapsed."""
        until = self._locked_until.get(username)
        if until is not None and time.time() >= until:
            del self._locked_until[username]
            self._attempts.pop(username, None)


# ── Module-level singletons ──────────────────────────────────

_password_policy: PasswordPolicy | None = None
_account_lockout: AccountLockout | None = None


def get_password_policy() -> PasswordPolicy:
    """Return the global ``PasswordPolicy`` singleton."""
    global _password_policy
    if _password_policy is None:
        _password_policy = PasswordPolicy()
    return _password_policy


def get_account_lockout() -> AccountLockout:
    """Return the global ``AccountLockout`` singleton."""
    global _account_lockout
    if _account_lockout is None:
        _account_lockout = AccountLockout()
    return _account_lockout
