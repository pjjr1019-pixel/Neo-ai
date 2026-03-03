"""
SHAP-based Model Explainability for NEO Hybrid AI.

Provides per-prediction explanations using TreeSHAP for the
sklearn ensemble (RandomForest + GradientBoosting).

Falls back gracefully to sklearn's built-in ``feature_importances_``
when the ``shap`` library is not installed.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load flag
_SHAP_AVAILABLE: Optional[bool] = None


def _check_shap() -> bool:
    """Check whether the shap library is importable."""
    global _SHAP_AVAILABLE
    if _SHAP_AVAILABLE is None:
        try:
            import shap  # noqa: F401

            _SHAP_AVAILABLE = True
        except ImportError:
            _SHAP_AVAILABLE = False
            logger.info("shap not installed; using sklearn importances")
    return _SHAP_AVAILABLE


def explain_prediction(
    model: Any,
    features: List[float],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Explain a single prediction using SHAP or fallback.

    Args:
        model: MLModel instance with ``rf_model``, ``gb_model``,
               and ``scaler`` attributes.
        features: Raw feature vector (pre-scaling).
        feature_names: Optional list of feature names.

    Returns:
        Dict with ``method``, ``feature_contributions``,
        ``base_value``, and ``predicted_value``.
    """
    n = len(features)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n)]

    arr = np.array(features, dtype=np.float64).reshape(1, -1)

    # Scale if scaler is available
    if model.scaler is not None:
        try:
            arr = model.scaler.transform(arr)
        except Exception as exc:
            logger.debug(
                "Scaler transform failed; using raw features: %s",
                exc,
            )

    if _check_shap():
        return _explain_shap(model, arr, feature_names)
    return _explain_sklearn(model, feature_names)


def _explain_shap(
    model: Any,
    scaled_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Compute SHAP values with TreeExplainer."""
    import shap

    contributions: Dict[str, float] = {}
    base_values: List[float] = []

    for name, sub_model in [
        ("rf", model.rf_model),
        ("gb", model.gb_model),
    ]:
        if sub_model is None:
            continue
        explainer = shap.TreeExplainer(sub_model)
        shap_values = explainer.shap_values(scaled_features)

        # shap_values may be a list (multi-class) or ndarray
        if isinstance(shap_values, list):
            vals = np.array(shap_values[0]).flatten()
        else:
            vals = np.array(shap_values).flatten()

        for i, fname in enumerate(feature_names):
            if i < len(vals):
                key = f"{fname}"
                contributions[key] = contributions.get(key, 0.0) + float(
                    vals[i]
                )

        bv = explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            base_values.append(float(bv[0]))
        else:
            base_values.append(float(bv))

    # Average contributions across sub-models
    n_models = sum(
        1 for m in [model.rf_model, model.gb_model] if m is not None
    )
    if n_models > 1:
        contributions = {k: v / n_models for k, v in contributions.items()}

    return {
        "method": "shap_tree",
        "feature_contributions": contributions,
        "base_value": (float(np.mean(base_values)) if base_values else 0.0),
        "num_features": len(feature_names),
    }


def _explain_sklearn(
    model: Any,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Fallback: return sklearn gini importances."""
    rf_imp = (
        model.rf_model.feature_importances_
        if model.rf_model is not None
        else np.zeros(len(feature_names))
    )
    gb_imp = (
        model.gb_model.feature_importances_
        if model.gb_model is not None
        else np.zeros(len(feature_names))
    )

    avg = (rf_imp + gb_imp) / 2.0

    contributions = {}
    for i, fname in enumerate(feature_names):
        if i < len(avg):
            contributions[fname] = float(avg[i])

    return {
        "method": "sklearn_gini_importance",
        "feature_contributions": contributions,
        "base_value": 0.0,
        "num_features": len(feature_names),
    }


def get_global_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return global feature importance (not per-prediction).

    Args:
        model: MLModel instance.
        feature_names: Optional feature name list.

    Returns:
        Dict with ranked feature importances.
    """
    rf_imp = (
        model.rf_model.feature_importances_
        if model.rf_model is not None
        else np.array([])
    )
    gb_imp = (
        model.gb_model.feature_importances_
        if model.gb_model is not None
        else np.array([])
    )

    if len(rf_imp) == 0 and len(gb_imp) == 0:
        return {"features": [], "method": "none"}

    if len(rf_imp) > 0 and len(gb_imp) > 0:
        avg = (rf_imp + gb_imp) / 2.0
    elif len(rf_imp) > 0:
        avg = rf_imp
    else:
        avg = gb_imp

    n = len(avg)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n)]

    ranked = sorted(
        zip(feature_names[:n], avg.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "method": "ensemble_gini",
        "features": [
            {"name": name, "importance": imp} for name, imp in ranked
        ],
    }
