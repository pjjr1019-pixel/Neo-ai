"""
Anomaly Detection Autoencoder for NEO Hybrid AI.

Uses a PyTorch autoencoder to learn normal market
behaviour from feature vectors.  Data points with
high reconstruction error are flagged as anomalies.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _Autoencoder(nn.Module):
    """Symmetric dense autoencoder.

    Architecture: input → encoder (hidden layers
    with ReLU) → bottleneck → decoder (mirrored) →
    output (same dim as input).
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 4) -> None:
        """Build encoder and decoder layers.

        Args:
            input_dim: Dimensionality of input features.
            bottleneck_dim: Size of the latent space.
        """
        super().__init__()
        mid = max((input_dim + bottleneck_dim) // 2, 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode *x*.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Reconstructed tensor of same shape.
        """
        latent = self.encoder(x)
        return self.decoder(latent)  # type: ignore[no-any-return]


class AnomalyDetector:
    """Autoencoder-based anomaly detector.

    Workflow:
        1. ``fit(data)`` — train the autoencoder on
           normal data.
        2. ``detect(data)`` — compute reconstruction
           errors and compare against the learned
           threshold.

    Args:
        input_dim: Number of input features.
        bottleneck_dim: Latent-space size.
        threshold_sigma: Number of standard deviations
            above the mean training error to set as
            the anomaly threshold.
        learning_rate: Optimiser learning rate.
        epochs: Training epochs.
        batch_size: Mini-batch size.
    """

    def __init__(
        self,
        input_dim: int = 10,
        bottleneck_dim: int = 4,
        threshold_sigma: float = 2.0,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        """Initialise the detector."""
        self._input_dim = input_dim
        self._model = _Autoencoder(input_dim, bottleneck_dim)
        self._sigma = threshold_sigma
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._threshold: Optional[float] = None
        self._train_mean: Optional[float] = None
        self._train_std: Optional[float] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,  # type: ignore[type-arg]
    ) -> Dict[str, Any]:
        """Train the autoencoder on *data*.

        After training, the anomaly threshold is set to
        ``mean(errors) + sigma * std(errors)``.

        Args:
            data: 2-D array of shape ``(n_samples, input_dim)``.

        Returns:
            Dict with ``final_loss``, ``threshold``,
            ``epochs``.
        """
        arr = np.asarray(data, dtype=np.float32)
        tensor = torch.from_numpy(arr)
        dataset = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
        )

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()
        self._model.train()

        final_loss = 0.0
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            final_loss = epoch_loss / len(arr)

        # Compute threshold from training errors.
        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor)
            errors = (recon - tensor).pow(2).mean(dim=1).numpy()
        self._train_mean = float(np.mean(errors))
        self._train_std = float(np.std(errors)) + 1e-8
        self._threshold = self._train_mean + self._sigma * self._train_std
        logger.info(
            "Autoencoder trained: loss=%.6f, " "threshold=%.6f",
            final_loss,
            self._threshold,
        )
        return {
            "final_loss": final_loss,
            "threshold": self._threshold,
            "epochs": self._epochs,
        }

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        data: np.ndarray,  # type: ignore[type-arg]
    ) -> Dict[str, Any]:
        """Detect anomalies in *data*.

        Args:
            data: 2-D array of shape ``(n_samples, input_dim)``.

        Returns:
            Dict with ``scores`` (per-sample MSE),
            ``is_anomaly`` (boolean array),
            ``threshold``, and ``anomaly_count``.

        Raises:
            RuntimeError: If the model has not been fit.
        """
        if self._threshold is None:
            raise RuntimeError("Call fit() first")

        arr = np.asarray(data, dtype=np.float32)
        tensor = torch.from_numpy(arr)
        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor)
            scores = (recon - tensor).pow(2).mean(dim=1).numpy()
        is_anomaly = scores > self._threshold
        return {
            "scores": scores,
            "is_anomaly": is_anomaly,
            "threshold": self._threshold,
            "anomaly_count": int(np.sum(is_anomaly)),
        }

    def reconstruction_errors(
        self,
        data: np.ndarray,  # type: ignore[type-arg]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Return per-sample reconstruction errors.

        Args:
            data: 2-D array of shape ``(n_samples, input_dim)``.

        Returns:
            1-D array of MSE values.
        """
        arr = np.asarray(data, dtype=np.float32)
        tensor = torch.from_numpy(arr)
        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor)
            return (  # type: ignore[no-any-return]
                (recon - tensor).pow(2).mean(dim=1).numpy()
            )

    def summary(self) -> Dict[str, Any]:
        """Return model summary.

        Returns:
            Dict with architecture info, threshold,
            and training stats.
        """
        return {
            "input_dim": self._input_dim,
            "threshold": self._threshold,
            "train_mean_error": self._train_mean,
            "train_std_error": self._train_std,
            "fitted": self._threshold is not None,
        }
