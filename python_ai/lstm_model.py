"""
LSTM Time-Series Predictor for NEO Hybrid AI.

Provides a configurable LSTM network for sequential
prediction (price direction, return forecasting).
Can serve as an ensemble member alongside the existing
RandomForest and GradientBoosting models.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    """Multi-layer LSTM followed by a fully-connected head.

    Architecture:
        input (seq_len, input_dim) → LSTM layers →
        last hidden state → FC → output_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Build the LSTM network.

        Args:
            input_dim: Feature dimensionality per step.
            hidden_dim: LSTM hidden-state size.
            num_layers: Stacked LSTM layers.
            output_dim: Output size (1 for regression,
                >1 for classification).
            dropout: Dropout between LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Output tensor of shape ``(batch, output_dim)``.
        """
        out, _ = self.lstm(x)
        # Use last time-step hidden state.
        last = out[:, -1, :]
        return self.fc(last)  # type: ignore[no-any-return]


def _prepare_sequences(
    data: np.ndarray,  # type: ignore[type-arg]
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Create sliding-window sequences.

    For a 2-D array of shape ``(T, F)``, produces
    input windows of shape ``(T - seq_len, seq_len, F)``
    and target values (last column of the next row).

    Args:
        data: 2-D feature array.
        seq_len: Window length.

    Returns:
        Tuple ``(X, y)`` of numpy arrays.
    """
    xs: List[np.ndarray] = []  # type: ignore[type-arg]
    ys: List[float] = []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(float(data[i + seq_len, -1]))
    return np.array(xs), np.array(ys)


class LSTMPredictor:
    """High-level LSTM predictor for time-series data.

    Wraps :class:`_LSTMNet` with a training loop,
    sequence preparation, and prediction API.

    Args:
        input_dim: Features per time step.
        seq_len: Sliding window length.
        hidden_dim: LSTM hidden size.
        num_layers: LSTM depth.
        output_dim: Prediction dimensionality.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.
        epochs: Training epochs.
        batch_size: Mini-batch size.
    """

    def __init__(
        self,
        input_dim: int = 10,
        seq_len: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        """Initialise the predictor."""
        self._seq_len = seq_len
        self._input_dim = input_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._model = _LSTMNet(
            input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            dropout,
        )
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data: np.ndarray,  # type: ignore[type-arg]
    ) -> Dict[str, Any]:
        """Train the LSTM on a feature matrix.

        The last column of *data* is used as the target.
        Sequences of length ``seq_len`` are created via
        a sliding window.

        Args:
            data: 2-D array ``(T, input_dim)`` where
                ``T > seq_len``.

        Returns:
            Dict with ``final_loss`` and ``epochs``.
        """
        arr = np.asarray(data, dtype=np.float32)
        x_np, y_np = _prepare_sequences(arr, self._seq_len)
        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np.astype(np.float32)).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(x_t, y_t)
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
            for x_batch, y_batch in loader:
                optimiser.zero_grad()
                pred = self._model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(x_batch)
            final_loss = epoch_loss / len(x_np)

        self._trained = True
        logger.info(
            "LSTM trained: %d epochs, loss=%.6f",
            self._epochs,
            final_loss,
        )
        return {
            "final_loss": final_loss,
            "epochs": self._epochs,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        sequence: np.ndarray,  # type: ignore[type-arg]
    ) -> float:
        """Predict the next value from a feature sequence.

        Args:
            sequence: 2-D array of shape
                ``(seq_len, input_dim)``.

        Returns:
            Scalar prediction.

        Raises:
            RuntimeError: If the model is not trained.
        """
        if not self._trained:
            raise RuntimeError("Call train() first")

        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :]
        tensor = torch.from_numpy(arr)
        self._model.eval()
        with torch.no_grad():
            out = self._model(tensor)
        return float(out.item())

    def predict_batch(
        self,
        sequences: np.ndarray,  # type: ignore[type-arg]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Predict a batch of sequences.

        Args:
            sequences: 3-D array ``(batch, seq_len,
                input_dim)``.

        Returns:
            1-D array of predictions.
        """
        if not self._trained:
            raise RuntimeError("Call train() first")

        arr = np.asarray(sequences, dtype=np.float32)
        tensor = torch.from_numpy(arr)
        self._model.eval()
        with torch.no_grad():
            out = self._model(tensor)
        return out.squeeze(1).numpy()  # type: ignore[no-any-return]

    def summary(self) -> Dict[str, Any]:
        """Return model summary.

        Returns:
            Dict with architecture and training status.
        """
        total = sum(p.numel() for p in self._model.parameters())
        return {
            "input_dim": self._input_dim,
            "seq_len": self._seq_len,
            "parameters": total,
            "trained": self._trained,
        }
