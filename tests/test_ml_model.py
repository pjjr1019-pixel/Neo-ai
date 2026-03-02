"""Tests for ML Model."""

import os
import tempfile

from python_ai.ml_model import MLModel, get_model


class TestMLModel:
    """Test suite for ML Model."""

    def test_model_initialization(self) -> None:
        """Test model can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            model = MLModel(model_path)
            assert model is not None
            assert model.is_trained is True

    def test_model_predict(self) -> None:
        """Test model can make predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            model = MLModel(model_path)
            features = {
                "f1": 1.0,
                "f2": 2.0,
                "f3": 3.0,
                "f4": 4.0,
                "f5": 5.0,
            }
            pred, confidence, signal = model.predict(features)
            assert isinstance(pred, float)
            assert 0.5 <= confidence <= 0.95
            assert signal in ["BUY", "SELL"]

    def test_model_confidence_range(self) -> None:
        """Test confidence is in valid range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            model = MLModel(model_path)
            features = {
                "f0": 1.0,
                "f1": 2.0,
                "f2": -1.0,
            }
            for _ in range(10):
                _, confidence, _ = model.predict(features)
                assert 0.5 <= confidence <= 0.95

    def test_model_persistence(self) -> None:
        """Test model can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")

            # Train and save
            model1 = MLModel(model_path)
            features = {"f0": 1.0, "f1": 2.0}
            pred1, conf1, sig1 = model1.predict(features)

            # Load and predict again
            model2 = MLModel(model_path)
            pred2, conf2, sig2 = model2.predict(features)

            # Should get same predictions
            assert pred1 == pred2
            assert conf1 == conf2
            assert sig1 == sig2

    def test_get_model_singleton(self) -> None:
        """Test get_model returns same instance."""
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2


class TestMLModelFeatures:
    """Test feature handling."""

    def test_dict_to_array_conversion(self) -> None:
        """Test feature dict conversion to array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            model = MLModel(model_path)
            features = {
                "f0": 1.0,
                "f1": 2.0,
                "f2": 3.0,
            }
            arr = model._dict_to_array(features)
            assert len(arr) == 10
            assert arr[0] == 1.0
            assert arr[1] == 2.0
            assert arr[2] == 3.0

    def test_missing_features_padded(self) -> None:
        """Test missing features are padded with zeros."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            model = MLModel(model_path)
            features = {"f0": 1.0}
            arr = model._dict_to_array(features)
            assert arr[0] == 1.0
            assert arr[1] == 0.0
            assert arr[9] == 0.0
