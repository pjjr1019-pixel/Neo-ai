"""
Data and Model Lineage Tracking for NEO Hybrid AI.

Records the provenance of datasets, transformations, and models
so that any prediction can be traced back to the raw data and
processing steps that produced it.

Provides a lightweight, file-backed lineage store that does not
require external services (MLflow, DVC) but captures the same
core metadata: *source -> transform -> artefact* chains.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── dataclasses ────────────────────────────────────────────────


@dataclass
class LineageRecord:
    """Single node in the lineage graph.

    Attributes:
        record_id: Unique identifier for this record.
        record_type: One of ``data``, ``transform``, or ``model``.
        name: Human-readable name (e.g. ``ohlcv_btc_2024``).
        version: Version string.
        created_at: Unix timestamp of creation.
        data_hash: SHA-256 hash of the artefact content.
        parent_ids: IDs of upstream records that produced this one.
        metadata: Arbitrary key-value metadata.
    """

    record_id: str
    record_type: str
    name: str
    version: str = "1.0"
    created_at: float = 0.0
    data_hash: str = ""
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LineageRecord":
        """Reconstruct from a dict."""
        return LineageRecord(**d)


# ── lineage store ──────────────────────────────────────────────


class LineageStore:
    """Persistent, file-backed lineage store.

    Stores lineage records in a JSON file so that the full
    provenance graph can be queried without a running database.

    Args:
        store_path: Path to the JSON lineage file.
    """

    def __init__(self, store_path: str = "data/lineage.json") -> None:
        """Initialise the lineage store."""
        self._path = store_path
        self._records: Dict[str, LineageRecord] = {}
        self._load()

    # ── persistence ────────────────────────────────────

    def _load(self) -> None:
        """Load records from disk."""
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._records = {
                k: LineageRecord.from_dict(v) for k, v in raw.items()
            }
            logger.info(
                "Loaded %d lineage records from %s",
                len(self._records),
                self._path,
            )
        else:
            self._records = {}

    def _save(self) -> None:
        """Persist records to disk."""
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(
                {k: v.to_dict() for k, v in self._records.items()},
                fh,
                indent=2,
            )

    # ── public API ─────────────────────────────────────

    def record_data(
        self,
        name: str,
        data_hash: str,
        version: str = "1.0",
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record a data artefact in the lineage graph.

        Args:
            name: Human-readable name for the dataset.
            data_hash: SHA-256 hash of the dataset content.
            version: Version string.
            parent_ids: IDs of upstream lineage records.
            metadata: Extra metadata (source URL, row count, etc.).

        Returns:
            The created ``LineageRecord``.
        """
        record_id = self._make_id("data", name, data_hash)
        rec = LineageRecord(
            record_id=record_id,
            record_type="data",
            name=name,
            version=version,
            created_at=time.time(),
            data_hash=data_hash,
            parent_ids=parent_ids or [],
            metadata=metadata or {},
        )
        self._records[record_id] = rec
        self._save()
        logger.info("Recorded data lineage: %s (%s)", name, record_id)
        return rec

    def record_transform(
        self,
        name: str,
        input_ids: List[str],
        output_hash: str,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record a transformation step.

        Args:
            name: Transform name (e.g. ``normalize_features``).
            input_ids: Lineage IDs of the input datasets.
            output_hash: Hash of the transform output.
            version: Transform version.
            metadata: Extra metadata (parameters, duration, etc.).

        Returns:
            The created ``LineageRecord``.
        """
        record_id = self._make_id("transform", name, output_hash)
        rec = LineageRecord(
            record_id=record_id,
            record_type="transform",
            name=name,
            version=version,
            created_at=time.time(),
            data_hash=output_hash,
            parent_ids=input_ids,
            metadata=metadata or {},
        )
        self._records[record_id] = rec
        self._save()
        logger.info("Recorded transform lineage: %s (%s)", name, record_id)
        return rec

    def record_model(
        self,
        name: str,
        data_ids: List[str],
        model_hash: str,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record a trained model in the lineage graph.

        Args:
            name: Model name (e.g. ``neo_ensemble_v3``).
            data_ids: Lineage IDs of the training data.
            model_hash: Hash of the serialised model file.
            version: Model version string.
            metadata: Hyperparameters, metrics, etc.

        Returns:
            The created ``LineageRecord``.
        """
        record_id = self._make_id("model", name, model_hash)
        rec = LineageRecord(
            record_id=record_id,
            record_type="model",
            name=name,
            version=version,
            created_at=time.time(),
            data_hash=model_hash,
            parent_ids=data_ids,
            metadata=metadata or {},
        )
        self._records[record_id] = rec
        self._save()
        logger.info("Recorded model lineage: %s (%s)", name, record_id)
        return rec

    def get(self, record_id: str) -> Optional[LineageRecord]:
        """Look up a lineage record by ID."""
        return self._records.get(record_id)

    def get_ancestors(self, record_id: str) -> List[LineageRecord]:
        """Return the full ancestor chain for a record.

        Performs a breadth-first traversal of ``parent_ids``.
        """
        visited: Dict[str, LineageRecord] = {}
        queue = [record_id]
        while queue:
            rid = queue.pop(0)
            if rid in visited:
                continue
            rec = self._records.get(rid)
            if rec is None:
                continue
            visited[rid] = rec
            queue.extend(rec.parent_ids)
        # exclude self
        visited.pop(record_id, None)
        return list(visited.values())

    def list_records(
        self,
        record_type: Optional[str] = None,
    ) -> List[LineageRecord]:
        """Return all records, optionally filtered by type."""
        records = list(self._records.values())
        if record_type:
            records = [r for r in records if r.record_type == record_type]
        return records

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the lineage store."""
        by_type: Dict[str, int] = {}
        for rec in self._records.values():
            by_type[rec.record_type] = by_type.get(rec.record_type, 0) + 1
        return {
            "total_records": len(self._records),
            "by_type": by_type,
            "store_path": self._path,
        }

    # ── helpers ────────────────────────────────────────

    @staticmethod
    def _make_id(record_type: str, name: str, data_hash: str) -> str:
        """Generate a deterministic record ID."""
        raw = f"{record_type}:{name}:{data_hash}:{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── convenience helpers ────────────────────────────────────────


def hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── global singleton ──────────────────────────────────────────

_store: Optional[LineageStore] = None


def get_lineage_store(
    store_path: str = "data/lineage.json",
) -> LineageStore:
    """Get or create the global lineage store."""
    global _store
    if _store is None:
        _store = LineageStore(store_path)
    return _store
