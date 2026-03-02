"""
AES-256-GCM request/response payload encryption for NEO Hybrid AI.

Provides symmetric authenticated encryption using
AES-256 in GCM mode.  Suitable for protecting
sensitive payloads (e.g. API keys, order details)
in transit or at rest.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, Optional, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# Nonce size recommended by NIST for AES-GCM.
_NONCE_BYTES: int = 12

# Key size in bytes (256 bits).
_KEY_BYTES: int = 32


def generate_key() -> bytes:
    """Generate a cryptographically-secure 256-bit key.

    Returns:
        32-byte key suitable for AES-256-GCM.
    """
    return os.urandom(_KEY_BYTES)


def key_to_b64(key: bytes) -> str:
    """Encode a binary key as URL-safe base-64 string.

    Args:
        key: Raw key bytes.

    Returns:
        URL-safe base-64 string.
    """
    return base64.urlsafe_b64encode(key).decode("ascii")


def key_from_b64(encoded: str) -> bytes:
    """Decode a URL-safe base-64 key string.

    Args:
        encoded: Base-64 encoded key.

    Returns:
        Raw key bytes.

    Raises:
        ValueError: If the decoded key length is wrong.
    """
    raw = base64.urlsafe_b64decode(encoded)
    if len(raw) != _KEY_BYTES:
        raise ValueError(f"Key must be {_KEY_BYTES} bytes, got {len(raw)}")
    return raw


def encrypt_payload(
    data: Union[str, bytes, Dict[str, Any]],
    key: bytes,
    aad: Optional[bytes] = None,
) -> str:
    """Encrypt *data* with AES-256-GCM.

    When *data* is a dict it is JSON-serialised first.

    The return value is a URL-safe base-64 string
    containing ``nonce || ciphertext || tag`` (the tag
    is appended automatically by GCM).

    Args:
        data: Plaintext payload (str, bytes, or dict).
        key: 32-byte AES-256 key.
        aad: Optional additional authenticated data.

    Returns:
        Base-64 encoded ciphertext string.

    Raises:
        ValueError: If the key length is invalid.
    """
    if len(key) != _KEY_BYTES:
        raise ValueError(f"Key must be {_KEY_BYTES} bytes, got {len(key)}")

    if isinstance(data, dict):
        plaintext = json.dumps(data, separators=(",", ":")).encode("utf-8")
    elif isinstance(data, str):
        plaintext = data.encode("utf-8")
    else:
        plaintext = data

    nonce = os.urandom(_NONCE_BYTES)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, aad)
    # nonce || ciphertext+tag
    combined = nonce + ct
    logger.debug(
        "Encrypted payload (%d bytes plaintext -> %d bytes ct)",
        len(plaintext),
        len(combined),
    )
    return base64.urlsafe_b64encode(combined).decode("ascii")


def decrypt_payload(
    token: str,
    key: bytes,
    aad: Optional[bytes] = None,
    as_json: bool = False,
) -> Union[bytes, Dict[str, Any]]:
    """Decrypt an AES-256-GCM token produced by *encrypt_payload*.

    Args:
        token: Base-64 encoded ciphertext string.
        key: 32-byte AES-256 key.
        aad: Additional authenticated data (must match
            the value used during encryption).
        as_json: If ``True``, parse the decrypted bytes
            as JSON and return a dict.

    Returns:
        Decrypted bytes, or a dict when *as_json* is set.

    Raises:
        ValueError: On key length mismatch or
            authentication failure.
    """
    if len(key) != _KEY_BYTES:
        raise ValueError(f"Key must be {_KEY_BYTES} bytes, got {len(key)}")

    raw = base64.urlsafe_b64decode(token)
    if len(raw) < _NONCE_BYTES + 16:
        raise ValueError("Ciphertext too short")

    nonce = raw[:_NONCE_BYTES]
    ct = raw[_NONCE_BYTES:]
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ct, aad)
    except Exception as exc:
        raise ValueError("Decryption failed") from exc

    logger.debug("Decrypted payload (%d bytes)", len(plaintext))

    if as_json:
        parsed: Dict[str, Any] = json.loads(plaintext)
        return parsed
    return plaintext


class EncryptionMiddleware:
    """ASGI middleware for transparent payload encryption.

    Inspects the ``X-Encrypted`` request header; when
    present the request body is decrypted before passing
    to the inner app.  Response bodies are encrypted when
    the client sends ``Accept-Encrypted: true``.

    Args:
        app: Inner ASGI application.
        key: 32-byte AES-256 key.
    """

    def __init__(self, app: Any, key: bytes) -> None:
        """Initialise with inner *app* and *key*."""
        self.app = app
        self._key = key

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """Handle an ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        encrypted_req = headers.get(b"x-encrypted", b"") == b"true"
        want_enc_resp = headers.get(b"accept-encrypted", b"") == b"true"

        async def _receive() -> Dict[str, Any]:
            """Optionally decrypt incoming body."""
            msg = await receive()
            if encrypted_req and msg.get("body"):
                body = msg["body"]
                plaintext = decrypt_payload(body.decode("utf-8"), self._key)
                if isinstance(plaintext, bytes):
                    msg = dict(msg, body=plaintext)
            return msg  # type: ignore[no-any-return]

        async def _send(msg: Dict[str, Any]) -> None:
            """Optionally encrypt outgoing body."""
            if (
                want_enc_resp
                and msg.get("type") == "http.response.body"
                and msg.get("body")
            ):
                ct = encrypt_payload(msg["body"], self._key)
                msg = dict(msg, body=ct.encode("utf-8"))
            await send(msg)

        await self.app(scope, _receive, _send)
