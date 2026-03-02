"""
HMAC Request Signing for NEO Hybrid AI.

Provides HMAC-SHA256 signing and verification for
securing sensitive API endpoints (e.g. order placement,
account operations).
"""

import hashlib
import hmac
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default validity window for timestamps (seconds).
DEFAULT_MAX_AGE: float = 300.0


def sign_request(
    secret: str,
    method: str,
    path: str,
    body: str = "",
    timestamp: Optional[float] = None,
) -> Dict[str, str]:
    """Create HMAC-SHA256 signature headers.

    The signature covers:
    ``METHOD\\nPATH\\nTIMESTAMP\\nBODY``

    Args:
        secret: Shared secret key.
        method: HTTP method (uppercased internally).
        path: Request path (e.g. ``/v1/order``).
        body: Request body string (empty for GET).
        timestamp: Unix timestamp (defaults to now).

    Returns:
        Dict with ``X-Signature``, ``X-Timestamp``.
    """
    ts = str(int(timestamp or time.time()))
    payload = f"{method.upper()}\n{path}\n{ts}\n{body}"
    sig = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "X-Signature": sig,
        "X-Timestamp": ts,
    }


def verify_signature(
    secret: str,
    method: str,
    path: str,
    body: str,
    signature: str,
    timestamp: str,
    max_age: float = DEFAULT_MAX_AGE,
) -> bool:
    """Verify an HMAC-SHA256 signed request.

    Checks both the signature integrity and that the
    timestamp is within the allowed window.

    Args:
        secret: Shared secret key.
        method: HTTP method.
        path: Request path.
        body: Request body.
        signature: Value of ``X-Signature`` header.
        timestamp: Value of ``X-Timestamp`` header.
        max_age: Maximum age of the timestamp in seconds.

    Returns:
        ``True`` if the signature is valid and fresh.
    """
    try:
        ts_int = int(timestamp)
    except (ValueError, TypeError):
        logger.warning("Invalid timestamp format")
        return False

    age = abs(time.time() - ts_int)
    if age > max_age:
        logger.warning(
            "Request too old (%.0fs > %.0fs)",
            age,
            max_age,
        )
        return False

    payload = f"{method.upper()}\n{path}\n" f"{timestamp}\n{body}"
    expected = hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    valid = hmac.compare_digest(signature, expected)
    if not valid:
        logger.warning("HMAC signature mismatch")
    return valid


class HMACVerifier:
    """Stateful HMAC verifier for use as middleware.

    Wraps ``verify_signature`` with a configured secret
    and logs verification attempts.

    Args:
        secret: Shared HMAC secret.
        max_age: Maximum request age in seconds.
    """

    def __init__(
        self,
        secret: str,
        max_age: float = DEFAULT_MAX_AGE,
    ) -> None:
        """Initialise with the shared secret."""
        self._secret = secret
        self._max_age = max_age
        self._verified: int = 0
        self._rejected: int = 0

    def verify(
        self,
        method: str,
        path: str,
        body: str,
        signature: str,
        timestamp: str,
    ) -> bool:
        """Verify a request signature.

        Args:
            method: HTTP method.
            path: Request path.
            body: Request body.
            signature: ``X-Signature`` header value.
            timestamp: ``X-Timestamp`` header value.

        Returns:
            ``True`` if valid.
        """
        ok = verify_signature(
            self._secret,
            method,
            path,
            body,
            signature,
            timestamp,
            self._max_age,
        )
        if ok:
            self._verified += 1
        else:
            self._rejected += 1
        return ok

    def stats(self) -> Dict[str, int]:
        """Return verification statistics."""
        return {
            "verified": self._verified,
            "rejected": self._rejected,
        }
