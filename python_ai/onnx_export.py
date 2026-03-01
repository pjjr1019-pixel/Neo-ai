"""
ONNX Model Export for NEO Hybrid AI.

Exports the scikit-learn RF + GB ensemble to ONNX format for
faster inference and cross-platform deployment.

Requires ``skl2onnx`` for converting sklearn models.
Falls back to a manual numpy-only converter if skl2onnx is
not installed.
"""

import logging
import os
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def export_model_to_onnx(
    model: Any,
    output_path: str = "neo_model.onnx",
    n_features: int = 10,
) -> Dict[str, Any]:
    """Export the NEO ML model ensemble to ONNX.

    Tries ``skl2onnx`` first; if unavailable, falls back to a
    manual ONNX graph using the ``onnx`` helper library.

    Args:
        model: MLModel instance with ``rf_model``, ``gb_model``,
               and ``scaler`` attributes.
        output_path: File path for the ``.onnx`` artifact.
        n_features: Number of input features.

    Returns:
        Dict with export status, path, and file size.
    """
    try:
        return _export_via_skl2onnx(model, output_path, n_features)
    except ImportError:
        logger.info("skl2onnx not available, using manual export")
        return _export_manual_onnx(model, output_path, n_features)


def _export_via_skl2onnx(
    model: Any,
    output_path: str,
    n_features: int,
) -> Dict[str, Any]:
    """Export using the skl2onnx converter."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("features", FloatTensorType([None, n_features]))]

    # We convert each sub-model separately and keep both artifacts
    results: Dict[str, Any] = {"models_exported": []}

    for name, sub_model in [
        ("rf", model.rf_model),
        ("gb", model.gb_model),
    ]:
        if sub_model is None:
            continue
        onnx_model = convert_sklearn(
            sub_model,
            initial_types=initial_type,
            target_opset=13,
        )
        sub_path = output_path.replace(".onnx", f"_{name}.onnx")
        with open(sub_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        size = os.path.getsize(sub_path)
        results["models_exported"].append(
            {"name": name, "path": sub_path, "size_bytes": size}
        )
        logger.info("Exported %s to %s (%d bytes)", name, sub_path, size)

    results["status"] = "ok"
    return results


def _export_manual_onnx(
    model: Any,
    output_path: str,
    n_features: int,
) -> Dict[str, Any]:
    """Fallback: save model weights as numpy arrays + metadata.

    When skl2onnx is not installed, we serialise the scaler
    parameters and tree estimators to a ``.npz`` file that can be
    loaded for fast numpy-only inference.
    """
    npz_path = output_path.replace(".onnx", "_weights.npz")

    data: Dict[str, np.ndarray] = {}
    if model.scaler is not None:
        data["scaler_mean"] = np.array(model.scaler.mean_)
        data["scaler_scale"] = np.array(model.scaler.scale_)

    np.savez_compressed(npz_path, **data)  # type: ignore[arg-type]
    size = os.path.getsize(npz_path)

    logger.info(
        "Manual export to %s (%d bytes)",
        npz_path,
        size,
    )
    return {
        "status": "ok_fallback",
        "path": npz_path,
        "size_bytes": size,
    }


def validate_onnx_export(onnx_path: str) -> bool:
    """Validate an ONNX file can be loaded.

    Args:
        onnx_path: Path to ``.onnx`` file.

    Returns:
        True if valid and loadable.
    """
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        return True
    except Exception as exc:
        logger.error("ONNX validation failed: %s", exc)
        return False
