"""Model export/import utilities with pruning/distillation hooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, cast

import joblib
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExportMetrics:
    """Paritiy metrics for export/import validation."""

    mse: float
    max_abs_error: float


def prune_weights(
    weights: NDArray[np.float64],
    threshold: float = 1e-3,
) -> NDArray[np.float64]:
    """Magnitude prune small weights to zero."""
    pruned = weights.copy()
    pruned[np.abs(pruned) < threshold] = 0.0
    return np.asarray(pruned, dtype=np.float64)


def distill_predictions(
    teacher: NDArray[np.float64],
    temperature: float = 2.0,
) -> NDArray[np.float64]:
    """Generate softened distillation targets from teacher predictions."""
    scaled = teacher / max(1e-12, temperature)
    exp = np.exp(scaled - np.max(scaled))
    return np.asarray(exp / np.sum(exp), dtype=np.float64)


def export_model_artifact(
    model_payload: Dict[str, object],
    path: str | Path,
) -> Path:
    """Export model payload to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, target, compress=3)
    return target


def import_model_artifact(path: str | Path) -> Dict[str, object]:
    """Import model payload from disk."""
    payload = joblib.load(path)
    return cast(Dict[str, object], payload)


def export_import_parity(
    y_true: NDArray[np.float64],
    y_pred_before: NDArray[np.float64],
    y_pred_after: NDArray[np.float64],
) -> ExportMetrics:
    """Compute parity metrics between pre/post export predictions."""
    residual = y_pred_before - y_pred_after
    mse = float(np.mean((y_true - y_pred_after) ** 2))
    max_abs_error = float(np.max(np.abs(residual)))
    return ExportMetrics(mse=mse, max_abs_error=max_abs_error)
