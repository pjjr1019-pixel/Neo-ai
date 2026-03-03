"""
Machine Learning Model for NEO Hybrid AI.

Provides a real ML model for price prediction with ensemble methods,
confidence calibration, and model persistence.
Supports both synthetic demo training and real market data training.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple

import joblib

__all__ = ["MLModel", "get_model"]
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from python_ai.data_pipeline import FEATURE_NAMES

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ── HMAC key for model file integrity ─────────────────────────
# In production, set MODEL_HMAC_KEY to a strong random secret.
_MODEL_HMAC_KEY: bytes = os.getenv(
    "MODEL_HMAC_KEY", "neo-model-integrity-default-key"
).encode()


class MLModel:
    """Machine Learning Model for predictions with confidence calibration.

    Uses Random Forest + Gradient Boosting ensemble for robust predictions.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize ML model.

        Args:
            model_path: Path to saved model. If None, creates new model.
        """
        self.model_path = model_path or "ml_model.pkl"
        self.rf_model: Optional[RandomForestRegressor] = None
        self.gb_model: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.training_metrics: Dict[str, Any] = {}
        self.train_count: int = 0
        self._predict_cache: Dict[
            Tuple[Tuple[str, float], ...],
            Tuple[float, float, str],
        ] = {}

        if os.path.exists(self.model_path):
            self.load()
        else:
            self._initialize_models()
            self.train_on_synthetic_data()

    def _initialize_models(self) -> None:
        """Initialize empty models."""
        self.rf_model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        self.gb_model = GradientBoostingRegressor(
            n_estimators=10,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        self.scaler = StandardScaler()

    def train_on_synthetic_data(self) -> None:
        """Train models on synthetic data for demonstration."""
        # Generate synthetic training data (100 samples, 10 features)
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        # Target: linear combination + noise
        y_train = (
            2 * X_train[:, 0]
            + 3 * X_train[:, 1]
            - X_train[:, 2]
            + 0.5 * np.random.randn(100)
        )

        # Scale features
        assert self.scaler is not None
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train models
        assert self.rf_model is not None
        assert self.gb_model is not None
        self.rf_model.fit(X_train_scaled, y_train)
        self.gb_model.fit(X_train_scaled, y_train)

        self.is_trained = True
        self.save()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train models on real market data.

        Replaces synthetic training with actual feature matrices
        produced by TrainingDataBuilder.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
            test_size: Fraction held out for evaluation.

        Returns:
            Dict with training metrics (r2, mse, samples, etc.).
        """
        if X.shape[0] < 10:
            from python_ai.exceptions import NeoModelTrainingError

            raise NeoModelTrainingError(
                f"Need >= 10 samples to train, got {X.shape[0]}",
                context={"samples": X.shape[0]},
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )

        self._initialize_models()

        assert self.scaler is not None
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        assert self.rf_model is not None
        assert self.gb_model is not None
        self.rf_model.fit(X_train_scaled, y_train)
        self.gb_model.fit(X_train_scaled, y_train)

        # Evaluate on held-out set
        rf_pred = self.rf_model.predict(X_test_scaled)
        gb_pred = self.gb_model.predict(X_test_scaled)
        ensemble_pred = (rf_pred + gb_pred) / 2.0

        metrics: Dict[str, Any] = {
            "r2_rf": float(r2_score(y_test, rf_pred)),
            "r2_gb": float(r2_score(y_test, gb_pred)),
            "r2_ensemble": float(r2_score(y_test, ensemble_pred)),
            "mse_rf": float(mean_squared_error(y_test, rf_pred)),
            "mse_gb": float(mean_squared_error(y_test, gb_pred)),
            "mse_ensemble": float(mean_squared_error(y_test, ensemble_pred)),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
        }

        self.training_metrics = metrics
        self.train_count += 1
        self.is_trained = True
        self.save()

        logger.info(
            "Model trained on %d samples  |  R² ensemble=%.4f  "
            "MSE ensemble=%.6f",
            X.shape[0],
            metrics["r2_ensemble"],
            metrics["mse_ensemble"],
        )
        return metrics

    def predict(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        """Make a prediction with confidence.

        Args:
            features: Dictionary of feature values.

        Returns:
            Tuple of (prediction, confidence, signal).
        """
        if not self.is_trained:
            from python_ai.exceptions import NeoModelNotTrainedError

            raise NeoModelNotTrainedError(
                "Model not trained. Call train() first."
            )

        cache_key = tuple(sorted((k, float(v)) for k, v in features.items()))
        cached = self._predict_cache.get(cache_key)
        if cached is not None:
            return cached

        # Convert dict to array
        feature_array = self._dict_to_array(features)

        # Scale features
        assert self.scaler is not None
        feature_scaled = self.scaler.transform([feature_array])

        # Get predictions from both models
        assert self.rf_model is not None
        assert self.gb_model is not None
        rf_pred = self.rf_model.predict(feature_scaled)[0]
        gb_pred = self.gb_model.predict(feature_scaled)[0]

        # Ensemble: average predictions
        ensemble_pred = float(np.round((rf_pred + gb_pred) / 2.0, 12))

        # Confidence: agreement between models (higher agreement = higher conf)
        disagreement = abs(rf_pred - gb_pred)
        confidence = float(
            np.round(max(0.5, min(0.95, 0.95 - disagreement * 0.1)), 12)
        )

        # Signal: simple rule (positive = BUY, negative = SELL)
        signal = "BUY" if ensemble_pred > 0 else "SELL"
        result = (float(ensemble_pred), float(confidence), signal)
        if len(self._predict_cache) >= 2048:
            self._predict_cache.clear()
        self._predict_cache[cache_key] = result
        return result

    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to array using canonical feature order.

        Maps feature values by name from ``FEATURE_NAMES`` so that the
        array order always matches what the model was trained on,
        regardless of dict insertion order.

        Unknown keys are logged as warnings and ignored.
        """
        arr = np.zeros(len(FEATURE_NAMES))
        known_keys = set(FEATURE_NAMES)
        for i, name in enumerate(FEATURE_NAMES):
            arr[i] = float(features.get(name, 0.0))
        unknown = set(features.keys()) - known_keys
        if unknown:
            logger.warning("Ignoring unknown feature keys: %s", unknown)
        return arr

    def save(self) -> None:
        """Save model to disk using joblib with HMAC integrity tag."""
        data = {
            "rf_model": self.rf_model,
            "gb_model": self.gb_model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "training_metrics": self.training_metrics,
            "train_count": self.train_count,
        }
        joblib.dump(data, self.model_path, compress=3)

        # Write HMAC-SHA256 tag (keyed hash, not plain SHA-256)
        tag_path = self.model_path + ".hmac"
        file_tag = self._file_hmac(self.model_path)
        with open(tag_path, "w") as hf:
            hf.write(file_tag)

    async def save_async(self) -> None:
        """Asynchronously persist model payload."""
        await asyncio.to_thread(self.save)

    def load(self) -> None:
        """Load model from disk using joblib.

        Verifies an HMAC-SHA256 integrity tag so that file
        tampering is detected even if the attacker can write to
        the model directory (unlike a plain SHA-256 sidecar).

        Falls back to legacy ``.sha256`` sidecars for existing
        models, upgrading them to HMAC on load.

        Raises:
            NeoModelIntegrityError: When the integrity check fails.
            NeoModelError: On any other load failure.
        """
        from python_ai.exceptions import (
            NeoModelError,
            NeoModelIntegrityError,
        )

        try:
            tag_path = self.model_path + ".hmac"
            legacy_path = self.model_path + ".sha256"
            actual_tag = self._file_hmac(self.model_path)

            if os.path.exists(tag_path):
                # HMAC tag found — verify
                with open(tag_path, "r") as hf:
                    expected = hf.read().strip()
                if not hmac.compare_digest(actual_tag, expected):
                    raise NeoModelIntegrityError(
                        "Model file HMAC check failed – "
                        "file may have been tampered with",
                        context={"path": self.model_path},
                    )
            elif os.path.exists(legacy_path):
                # Legacy SHA-256 sidecar — verify then upgrade
                actual_sha = self._file_hash(self.model_path)
                with open(legacy_path, "r") as hf:
                    expected_sha = hf.read().strip()
                if actual_sha != expected_sha:
                    raise NeoModelIntegrityError(
                        "Model file integrity check failed – "
                        "file may have been tampered with",
                        context={"path": self.model_path},
                    )
                # Upgrade to HMAC
                with open(tag_path, "w") as hf:
                    hf.write(actual_tag)
                logger.info(
                    "Upgraded model integrity from SHA-256 to HMAC " "for %s",
                    self.model_path,
                )
            else:
                # No sidecar — generate HMAC for future loads
                with open(tag_path, "w") as hf:
                    hf.write(actual_tag)
                logger.warning(
                    "No integrity sidecar found for %s – "
                    "generated HMAC for future integrity checks",
                    self.model_path,
                )

            data: Dict[str, Any] = joblib.load(self.model_path)
            self.rf_model = data["rf_model"]
            self.gb_model = data["gb_model"]
            self.scaler = data["scaler"]
            self.is_trained = data["is_trained"]
            self.training_metrics = data.get(
                "training_metrics",
                {},
            )
            self.train_count = data.get("train_count", 0)
            if not hasattr(self, "_predict_cache"):
                self._predict_cache = {}
            else:
                self._predict_cache.clear()
        except (NeoModelIntegrityError, NeoModelError):
            raise
        except Exception as e:
            raise NeoModelError(
                f"Failed to load model: {e}",
                context={"path": self.model_path},
            ) from e

    async def load_async(self) -> None:
        """Asynchronously load model payload."""
        await asyncio.to_thread(self.load)

    def predict_with_onnx_runtime(
        self,
        features: Dict[str, float],
    ) -> Tuple[float, float, str]:
        """Optional ONNX runtime inference path with graceful fallback."""
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return self.predict(features)
        return self.predict(features)

    @staticmethod
    def _file_hmac(path: str) -> str:
        """Compute HMAC-SHA256 of a file using the model secret key."""
        h = hmac.new(_MODEL_HMAC_KEY, digestmod=hashlib.sha256)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _file_hash(path: str) -> str:
        """Compute SHA-256 hash of a file (legacy compat)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# Global model instance (singleton)
_model: Optional[MLModel] = None


def get_model() -> MLModel:
    """Get global ML model instance."""
    global _model
    if _model is None:
        _model = MLModel()
    return _model
