"""Tests for password policy and account lockout.

Covers:
- PasswordPolicy: complexity rules, edge cases, custom config
- AccountLockout: failure tracking, locking, expiry, reset
- Module singletons: get_password_policy, get_account_lockout
"""

import time
from unittest.mock import patch

import pytest

from python_ai.auth.password_policy import (
    AccountLockout,
    LockoutStatus,
    PasswordPolicy,
    PasswordValidationResult,
    get_account_lockout,
    get_password_policy,
)


# ── PasswordPolicy ───────────────────────────────────────────


class TestPasswordPolicy:
    """Tests for PasswordPolicy.validate()."""

    def setup_method(self):
        """Create default policy for each test."""
        self.policy = PasswordPolicy()

    def test_valid_password(self):
        """A strong password passes all rules."""
        result = self.policy.validate("Str0ng!Pass#1")
        assert result.valid is True
        assert result.errors == []

    def test_too_short(self):
        """Password below min_length is rejected."""
        result = self.policy.validate("Ab1!")
        assert result.valid is False
        assert any("at least 8" in e for e in result.errors)

    def test_too_long(self):
        """Password exceeding max_length is rejected."""
        long_pw = "Aa1!" + "x" * 200
        result = self.policy.validate(long_pw)
        assert result.valid is False
        assert any("at most 128" in e for e in result.errors)

    def test_missing_uppercase(self):
        """Password without uppercase is rejected."""
        result = self.policy.validate("str0ng!pass#1")
        assert result.valid is False
        assert any("uppercase" in e for e in result.errors)

    def test_missing_lowercase(self):
        """Password without lowercase is rejected."""
        result = self.policy.validate("STR0NG!PASS#1")
        assert result.valid is False
        assert any("lowercase" in e for e in result.errors)

    def test_missing_digit(self):
        """Password without digit is rejected."""
        result = self.policy.validate("Strong!Pass#X")
        assert result.valid is False
        assert any("digit" in e for e in result.errors)

    def test_missing_special(self):
        """Password without special character is rejected."""
        result = self.policy.validate("StrongPass12")
        assert result.valid is False
        assert any("special" in e for e in result.errors)

    def test_multiple_violations(self):
        """All violations are reported at once."""
        result = self.policy.validate("abc")
        assert result.valid is False
        assert len(result.errors) >= 3  # short, no upper, no digit, no special

    def test_custom_min_length(self):
        """Custom min_length is honoured."""
        policy = PasswordPolicy(min_length=4, require_special=False)
        result = policy.validate("Abc1")
        assert result.valid is True

    def test_disabled_uppercase_rule(self):
        """Uppercase rule can be disabled."""
        policy = PasswordPolicy(require_uppercase=False)
        result = policy.validate("str0ng!pass#1")
        assert result.valid is True

    def test_disabled_special_rule(self):
        """Special char rule can be disabled."""
        policy = PasswordPolicy(require_special=False)
        result = policy.validate("StrongPass12")
        assert result.valid is True

    def test_empty_string(self):
        """Empty password fails."""
        result = self.policy.validate("")
        assert result.valid is False

    def test_result_dataclass_fields(self):
        """PasswordValidationResult has expected fields."""
        r = PasswordValidationResult(valid=True, errors=[])
        assert r.valid is True
        assert r.errors == []


# ── AccountLockout ────────────────────────────────────────────


class TestAccountLockout:
    """Tests for AccountLockout."""

    def setup_method(self):
        """Create lockout with small window for fast tests."""
        self.lockout = AccountLockout(
            max_attempts=3,
            lockout_seconds=0.5,
        )

    def test_no_lockout_initially(self):
        """Fresh user is not locked."""
        assert self.lockout.is_locked("alice") is False

    def test_status_fresh_user(self):
        """Status for unknown user shows 0 attempts."""
        status = self.lockout.status("alice")
        assert status.locked is False
        assert status.attempts == 0
        assert status.remaining == 3
        assert status.locked_until is None

    def test_record_failure_increments(self):
        """Each failure increments the counter."""
        s1 = self.lockout.record_failure("alice")
        assert s1.attempts == 1
        assert s1.remaining == 2
        assert s1.locked is False

        s2 = self.lockout.record_failure("alice")
        assert s2.attempts == 2
        assert s2.remaining == 1

    def test_lockout_after_max_attempts(self):
        """Account locks after max_attempts failures."""
        for _ in range(3):
            self.lockout.record_failure("alice")

        assert self.lockout.is_locked("alice") is True
        status = self.lockout.status("alice")
        assert status.locked is True
        assert status.remaining == 0
        assert status.locked_until is not None

    def test_lockout_expires(self):
        """Lockout auto-clears after lockout_seconds."""
        for _ in range(3):
            self.lockout.record_failure("alice")
        assert self.lockout.is_locked("alice") is True

        time.sleep(0.6)
        assert self.lockout.is_locked("alice") is False

    def test_record_success_clears(self):
        """Successful login clears failure count."""
        self.lockout.record_failure("alice")
        self.lockout.record_failure("alice")
        self.lockout.record_success("alice")

        status = self.lockout.status("alice")
        assert status.attempts == 0
        assert status.locked is False

    def test_reset_unlocks(self):
        """Admin reset immediately unlocks."""
        for _ in range(3):
            self.lockout.record_failure("alice")
        assert self.lockout.is_locked("alice") is True

        self.lockout.reset("alice")
        assert self.lockout.is_locked("alice") is False
        assert self.lockout.status("alice").attempts == 0

    def test_independent_users(self):
        """Lockout state is per-user."""
        for _ in range(3):
            self.lockout.record_failure("alice")
        assert self.lockout.is_locked("alice") is True
        assert self.lockout.is_locked("bob") is False

    def test_lockout_status_dataclass(self):
        """LockoutStatus has expected fields."""
        s = LockoutStatus(
            locked=True, attempts=5, remaining=0, locked_until=1.0
        )
        assert s.locked is True
        assert s.locked_until == 1.0


# ── Singletons ────────────────────────────────────────────────


class TestSingletons:
    """Tests for module-level singletons."""

    def test_get_password_policy_returns_policy(self):
        """get_password_policy returns a PasswordPolicy instance."""
        policy = get_password_policy()
        assert isinstance(policy, PasswordPolicy)

    def test_get_password_policy_is_singleton(self):
        """Same instance returned on repeated calls."""
        a = get_password_policy()
        b = get_password_policy()
        assert a is b

    def test_get_account_lockout_returns_lockout(self):
        """get_account_lockout returns an AccountLockout."""
        lockout = get_account_lockout()
        assert isinstance(lockout, AccountLockout)

    def test_get_account_lockout_is_singleton(self):
        """Same instance returned on repeated calls."""
        a = get_account_lockout()
        b = get_account_lockout()
        assert a is b
