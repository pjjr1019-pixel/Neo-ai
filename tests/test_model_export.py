"""Tests for model export/import parity and efficiency helpers."""

from pathlib import Path

import numpy as np

from python_ai.model_export import (
    distill_predictions,
    export_import_parity,
    export_model_artifact,
    import_model_artifact,
    prune_weights,
)


def test_prune_weights_zeroes_small_values() -> None:
    weights = np.array([1e-5, 0.5, -1e-4, 1.2], dtype=np.float64)
    pruned = prune_weights(weights, threshold=1e-3)
    assert pruned[0] == 0.0
    assert pruned[2] == 0.0
    assert pruned[1] != 0.0


def test_distill_predictions_normalized() -> None:
    teacher = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    distilled = distill_predictions(teacher, temperature=2.0)
    assert abs(float(np.sum(distilled)) - 1.0) < 1e-9


def test_export_import_and_parity(tmp_path: Path) -> None:
    artifact = {"weights": np.array([1.0, 2.0, 3.0])}
    path = export_model_artifact(artifact, tmp_path / "model.joblib")
    restored = import_model_artifact(path)
    assert "weights" in restored

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred_before = np.array([1.1, 1.9, 3.1])
    y_pred_after = np.array([1.1, 1.9, 3.1])
    metrics = export_import_parity(y_true, y_pred_before, y_pred_after)
    assert metrics.max_abs_error == 0.0
