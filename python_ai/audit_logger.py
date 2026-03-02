"""
Immutable Audit Logger for NEO Hybrid AI.

Provides tamper-evident audit logging with hash-chain
integrity verification.  Each entry's hash is derived
from its content AND the previous entry's hash,
creating a blockchain-like chain of custody.
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)

_GENESIS_HASH = (
    "0000000000000000000000000000000000000000" "000000000000000000000000"
)


@dataclass
class AuditEntry:
    """A single audit log entry.

    Attributes:
        sequence: Monotonic sequence number.
        timestamp: Unix epoch of the event.
        actor: Who performed the action.
        action: What was done.
        resource: What was acted upon.
        details: Arbitrary metadata.
        prev_hash: Hash of the preceding entry.
        entry_hash: Hash of this entry.
    """

    sequence: int
    timestamp: float
    actor: str
    action: str
    resource: str
    details: Dict[str, Any] = field(default_factory=dict)
    prev_hash: str = _GENESIS_HASH
    entry_hash: str = ""


def _compute_hash(entry: AuditEntry) -> str:
    """Compute SHA-256 hash for an audit entry.

    The hash covers all fields except ``entry_hash``
    itself.

    Args:
        entry: Entry to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = (
        f"{entry.sequence}|"
        f"{entry.timestamp}|"
        f"{entry.actor}|"
        f"{entry.action}|"
        f"{entry.resource}|"
        f"{json.dumps(entry.details, sort_keys=True)}|"
        f"{entry.prev_hash}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditLogger:
    """Tamper-evident audit logger with hash chain.

    Each log entry is chained to its predecessor via
    SHA-256, creating a verifiable audit trail.

    Args:
        log_file: Optional path for persistent storage
            (JSONL format).
        max_entries: In-memory entry cap.  ``0`` means
            unlimited.
    """

    def __init__(
        self,
        log_file: Optional[Union[str, Path]] = None,
        max_entries: int = 0,
    ) -> None:
        """Initialise the audit logger."""
        self._log_file = Path(log_file) if log_file else None
        self._max = max_entries
        self._entries: List[AuditEntry] = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._last_hash = _GENESIS_HASH

        if self._log_file and self._log_file.exists():
            self._load()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        actor: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Record an audit event.

        Args:
            actor: Identity performing the action.
            action: Action verb (``"create"``,
                ``"delete"``, etc.).
            resource: Target of the action.
            details: Extra metadata.

        Returns:
            The created :class:`AuditEntry`.
        """
        with self._lock:
            self._sequence += 1
            entry = AuditEntry(
                sequence=self._sequence,
                timestamp=time.time(),
                actor=actor,
                action=action,
                resource=resource,
                details=details or {},
                prev_hash=self._last_hash,
            )
            entry.entry_hash = _compute_hash(entry)
            self._last_hash = entry.entry_hash
            self._entries.append(entry)

            if self._max and len(self._entries) > self._max:
                self._entries = self._entries[-self._max :]

            self._persist(entry)

        logger.debug(
            "Audit #%d: %s %s %s",
            entry.sequence,
            actor,
            action,
            resource,
        )
        return entry

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """Verify the integrity of the hash chain.

        Returns:
            ``True`` if no tampering detected.
        """
        with self._lock:
            entries = list(self._entries)

        prev = _GENESIS_HASH
        for entry in entries:
            if entry.prev_hash != prev:
                logger.error(
                    "Chain broken at #%d: expected %s, " "got %s",
                    entry.sequence,
                    prev,
                    entry.prev_hash,
                )
                return False
            expected = _compute_hash(entry)
            if entry.entry_hash != expected:
                logger.error("Hash mismatch at #%d", entry.sequence)
                return False
            prev = entry.entry_hash
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entries(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        limit: int = 50,
    ) -> List[AuditEntry]:
        """Query audit entries with optional filters.

        Args:
            actor: Filter by actor.
            action: Filter by action.
            resource: Filter by resource.
            limit: Maximum results.

        Returns:
            Matching entries (newest first).
        """
        with self._lock:
            entries = list(reversed(self._entries))

        results: List[AuditEntry] = []
        for e in entries:
            if actor and e.actor != actor:
                continue
            if action and e.action != action:
                continue
            if resource and e.resource != resource:
                continue
            results.append(e)
            if len(results) >= limit:
                break
        return results

    @property
    def entry_count(self) -> int:
        """Total entries in the log."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence (JSONL)
    # ------------------------------------------------------------------

    def _persist(self, entry: AuditEntry) -> None:
        """Append an entry to the log file.

        Args:
            entry: Entry to persist.
        """
        if not self._log_file:
            return
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "seq": entry.sequence,
            "ts": entry.timestamp,
            "actor": entry.actor,
            "action": entry.action,
            "resource": entry.resource,
            "details": entry.details,
            "prev_hash": entry.prev_hash,
            "hash": entry.entry_hash,
        }
        with open(self._log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load(self) -> None:
        """Load entries from the log file."""
        if not self._log_file or not self._log_file.exists():
            return
        with open(self._log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                entry = AuditEntry(
                    sequence=rec["seq"],
                    timestamp=rec["ts"],
                    actor=rec["actor"],
                    action=rec["action"],
                    resource=rec["resource"],
                    details=rec.get("details", {}),
                    prev_hash=rec["prev_hash"],
                    entry_hash=rec["hash"],
                )
                self._entries.append(entry)
                self._sequence = max(self._sequence, entry.sequence)
                self._last_hash = entry.entry_hash

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return audit logger status.

        Returns:
            Dict with entry count, chain status, and
            log file info.
        """
        return {
            "entry_count": self.entry_count,
            "chain_valid": self.verify_chain(),
            "log_file": (str(self._log_file) if self._log_file else None),
            "last_hash": self._last_hash[:16] + "...",
        }
