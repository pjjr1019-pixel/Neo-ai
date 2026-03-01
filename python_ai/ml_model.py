"""
Machine Learning Model for NEO Hybrid AI.

Provides a real ML model for price prediction with ensemble methods,
confidence calibration, and model persistence.
"""

import os
import pickle
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
            n_jobs=1,
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

    def predict(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        """Make a prediction with confidence.

        Args:
            features: Dictionary of feature values.

        Returns:
            Tuple of (prediction, confidence, signal).
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

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
        ensemble_pred = (rf_pred + gb_pred) / 2.0

        # Confidence: agreement between models (higher agreement = higher conf)
        disagreement = abs(rf_pred - gb_pred)
        confidence = max(0.5, min(0.95, 0.95 - disagreement * 0.1))

        # Signal: simple rule (positive = BUY, negative = SELL)
        signal = "BUY" if ensemble_pred > 0 else "SELL"

        return ensemble_pred, confidence, signal

    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to array (10 features, padded with 0s)."""
        arr = np.zeros(10)
        for i, (key, value) in enumerate(features.items()):
            if i < 10:
                arr[i] = float(value)
        return arr

    def save(self) -> None:
        """Save model to disk."""
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "rf_model": self.rf_model,
                    "gb_model": self.gb_model,
                    "scaler": self.scaler,
                    "is_trained": self.is_trained,
                },
                f,
            )

    def load(self) -> None:
        """Load model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                data: Dict[str, Any] = pickle.load(f)
                self.rf_model = data["rf_model"]
                self.gb_model = data["gb_model"]
                self.scaler = data["scaler"]
                self.is_trained = data["is_trained"]
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e


# Global model instance (singleton)
_model: Optional[MLModel] = None


def get_model() -> MLModel:
    """Get global ML model instance."""
    global _model
    if _model is None:
        _model = MLModel()
    return _model
