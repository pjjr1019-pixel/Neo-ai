"""
Model Versioning System for NEO Hybrid AI.

Tracks model versions with metadata, supports rollback,
and maintains a registry of all trained models.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Snapshot of a trained model.

    Attributes:
        version: Semantic version string (e.g. ``v1.2.3``).
        created_at: Unix timestamp of creation.
        model_path: Path to the serialised model file.
        metrics: Training/validation metrics dict.
        hyperparameters: Model hyperparameters.
        data_hash: SHA-256 hash of the training data.
        tags: User-defined tags for filtering.
    """

    version: str
    created_at: float
    model_path: str
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    data_hash: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "data_hash": self.data_hash,
            "tags": self.tags,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelVersion":
        """Reconstruct from a dict."""
        return ModelVersion(
            version=d["version"],
            created_at=d["created_at"],
            model_path=d["model_path"],
            metrics=d.get("metrics", {}),
            hyperparameters=d.get("hyperparameters", {}),
            data_hash=d.get("data_hash", ""),
            tags=d.get("tags", []),
        )


class ModelRegistry:
    """Persistent model version registry.

    Stores metadata in a JSON manifest file and keeps
    model artefacts in a directory tree.

    Args:
        registry_dir: Root directory for model storage.
    """

    def __init__(self, registry_dir: str = "model_registry") -> None:
        """Initialise the registry directory."""
        self._dir = registry_dir
        os.makedirs(self._dir, exist_ok=True)
        self._manifest_path = os.path.join(self._dir, "manifest.json")
        self._versions: List[ModelVersion] = []
        self._load_manifest()

    # ── persistence ────────────────────────────────────

    def _load_manifest(self) -> None:
        """Load version list from manifest file."""
        if os.path.exists(self._manifest_path):
            with open(self._manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._versions = [ModelVersion.from_dict(v) for v in data]
            logger.info(
                "Loaded %d model versions from registry",
                len(self._versions),
            )
        else:
            self._versions = []

    def _save_manifest(self) -> None:
        """Persist version list to manifest file."""
        with open(self._manifest_path, "w", encoding="utf-8") as fh:
            json.dump(
                [v.to_dict() for v in self._versions],
                fh,
                indent=2,
            )

    # ── public API ─────────────────────────────────────

    def register(
        self,
        model_path: str,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        data_hash: str = "",
    ) -> ModelVersion:
        """Register a new model version.

        Copies the model file into the registry directory
        and assigns the next version number.

        Args:
            model_path: Path to the model file to register.
            metrics: Training / validation metrics.
            hyperparameters: Model hyperparameters.
            tags: Optional tags for categorisation.
            data_hash: Hash of the training data.

        Returns:
            The newly created ``ModelVersion``.
        """
        next_num = len(self._versions) + 1
        version_str = f"v{next_num}.0.0"

        dest_dir = os.path.join(self._dir, version_str)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(model_path))
        shutil.copy2(model_path, dest_path)

        mv = ModelVersion(
            version=version_str,
            created_at=time.time(),
            model_path=dest_path,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            data_hash=data_hash,
            tags=tags or [],
        )
        self._versions.append(mv)
        self._save_manifest()
        logger.info("Registered model %s", version_str)
        return mv

    def latest(self) -> Optional[ModelVersion]:
        """Return the most recently registered version."""
        if not self._versions:
            return None
        return self._versions[-1]

    def get(self, version: str) -> Optional[ModelVersion]:
        """Look up a version by its string identifier."""
        for mv in self._versions:
            if mv.version == version:
                return mv
        return None

    def list_versions(self) -> List[ModelVersion]:
        """Return all registered versions."""
        return list(self._versions)

    def rollback(self, version: str) -> Optional[str]:
        """Restore a previous model version.

        Returns the path to the restored model file,
        or ``None`` if the version is not found.
        """
        mv = self.get(version)
        if mv is None:
            logger.warning(
                "Version %s not found for rollback",
                version,
            )
            return None
        logger.info("Rolling back to model %s", version)
        return mv.model_path

    def summary(self) -> Dict[str, Any]:
        """Registry summary dict."""
        latest = self.latest()
        return {
            "total_versions": len(self._versions),
            "latest": (latest.to_dict() if latest else None),
            "all_versions": [v.version for v in self._versions],
        }


def hash_data(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw data."""
    return hashlib.sha256(data).hexdigest()


# ── global singleton ──────────────────────────────────

_registry: Optional[ModelRegistry] = None


def get_model_registry(
    registry_dir: str = "model_registry",
) -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(registry_dir)
    return _registry
