"""Tests for SHAP-based model explainability module."""

from types import SimpleNamespace

import numpy as np

from python_ai.explainability import (
    _check_shap,
    _explain_sklearn,
    explain_prediction,
    get_global_importance,
)


# ── helpers ────────────────────────────────────────────────


def _make_model(n_features: int = 5):
    """Build a minimal mock MLModel with importances."""
    rf = SimpleNamespace(
        feature_importances_=np.linspace(0.1, 0.5, n_features),
    )
    gb = SimpleNamespace(
        feature_importances_=np.linspace(0.5, 0.1, n_features),
    )
    scaler = SimpleNamespace(
        transform=lambda x: x,
    )
    return SimpleNamespace(
        rf_model=rf,
        gb_model=gb,
        scaler=scaler,
    )


def _make_model_no_scaler(n_features: int = 5):
    """Build a mock MLModel without a scaler."""
    rf = SimpleNamespace(
        feature_importances_=np.linspace(0.1, 0.5, n_features),
    )
    gb = SimpleNamespace(
        feature_importances_=np.linspace(0.5, 0.1, n_features),
    )
    return SimpleNamespace(
        rf_model=rf,
        gb_model=gb,
        scaler=None,
    )


def _make_model_rf_only(n_features: int = 5):
    """Build a mock MLModel with RF only."""
    rf = SimpleNamespace(
        feature_importances_=np.linspace(0.2, 0.8, n_features),
    )
    return SimpleNamespace(
        rf_model=rf,
        gb_model=None,
        scaler=None,
    )


# ── _check_shap ───────────────────────────────────────────


class TestCheckShap:
    """Tests for the shap availability check."""

    def test_returns_bool(self):
        """_check_shap returns a boolean."""
        result = _check_shap()
        assert isinstance(result, bool)


# ── _explain_sklearn ──────────────────────────────────────


class TestExplainSklearn:
    """Tests for the sklearn gini fallback."""

    def test_basic_output(self):
        """Returns dict with expected keys."""
        model = _make_model(3)
        names = ["a", "b", "c"]
        result = _explain_sklearn(model, names)

        assert result["method"] == "sklearn_gini_importance"
        assert "feature_contributions" in result
        assert result["num_features"] == 3

    def test_feature_names_match(self):
        """Contribution keys match the given feature names."""
        model = _make_model(3)
        names = ["x", "y", "z"]
        result = _explain_sklearn(model, names)
        assert set(result["feature_contributions"].keys()) == {
            "x",
            "y",
            "z",
        }

    def test_importance_values_are_floats(self):
        """All contribution values are Python floats."""
        model = _make_model(4)
        names = [f"f{i}" for i in range(4)]
        result = _explain_sklearn(model, names)
        for v in result["feature_contributions"].values():
            assert isinstance(v, float)


# ── explain_prediction ────────────────────────────────────


class TestExplainPrediction:
    """Tests for the per-prediction explainer."""

    def test_returns_dict(self):
        """explain_prediction returns a dict."""
        model = _make_model(3)
        result = explain_prediction(model, [1.0, 2.0, 3.0])
        assert isinstance(result, dict)
        assert "method" in result
        assert "feature_contributions" in result

    def test_auto_feature_names(self):
        """Default names are feature_0 .. feature_N."""
        model = _make_model(3)
        result = explain_prediction(model, [1.0, 2.0, 3.0])
        contribs = result["feature_contributions"]
        assert "feature_0" in contribs

    def test_custom_feature_names(self):
        """Custom names are propagated."""
        model = _make_model(2)
        result = explain_prediction(
            model,
            [1.0, 2.0],
            feature_names=["alpha", "beta"],
        )
        contribs = result["feature_contributions"]
        assert "alpha" in contribs or "beta" in contribs

    def test_no_scaler(self):
        """Works when model has no scaler."""
        model = _make_model_no_scaler(3)
        result = explain_prediction(model, [1.0, 2.0, 3.0])
        assert "method" in result

    def test_scaler_exception_handled(self):
        """Gracefully handles scaler.transform raising."""
        model = _make_model(3)
        model.scaler.transform = lambda x: (_ for _ in ()).throw(
            ValueError("bad")
        )
        result = explain_prediction(model, [1.0, 2.0, 3.0])
        assert "method" in result


# ── get_global_importance ─────────────────────────────────


class TestGetGlobalImportance:
    """Tests for the global importance ranking."""

    def test_basic_output(self):
        """Returns dict with method and features."""
        model = _make_model(5)
        result = get_global_importance(model)
        assert result["method"] == "ensemble_gini"
        assert "features" in result

    def test_sorted_descending(self):
        """Features are sorted by descending importance."""
        model = _make_model(5)
        result = get_global_importance(model)
        importances = [
            f["importance"] for f in result["features"]
        ]
        assert importances == sorted(
            importances, reverse=True
        )

    def test_custom_names(self):
        """Custom feature names appear in output."""
        model = _make_model(3)
        names = ["open", "high", "low"]
        result = get_global_importance(model, names)
        feature_names = [
            f["name"] for f in result["features"]
        ]
        for n in names:
            assert n in feature_names

    def test_rf_only(self):
        """Works with RF model only."""
        model = _make_model_rf_only(3)
        result = get_global_importance(model)
        assert len(result["features"]) == 3

    def test_no_models(self):
        """Returns empty when both models are None."""
        model = SimpleNamespace(
            rf_model=None,
            gb_model=None,
        )
        result = get_global_importance(model)
        assert result["method"] == "none"
        assert result["features"] == []
