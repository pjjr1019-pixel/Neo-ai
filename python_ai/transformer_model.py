"""
Transformer Attention Model for NEO Hybrid AI.

Lightweight Transformer encoder for sequential pattern
recognition in time-series data.  Uses positional
encoding and multi-head self-attention.  Can serve as
an alternative or ensemble member alongside LSTM and
tree-based models.
"""

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        """Build positional encoding table.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x*.

        Args:
            x: ``(batch, seq_len, d_model)``

        Returns:
            Position-encoded tensor of same shape.
        """
        out: torch.Tensor = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return out


class _TransformerNet(nn.Module):
    """Transformer encoder with a regression/classification head.

    Architecture:
        input → linear projection → positional encoding →
        N transformer-encoder layers → mean pooling → FC → output
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Build the transformer.

        Args:
            input_dim: Feature dimensionality per step.
            d_model: Internal embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Transformer encoder layers.
            output_dim: Prediction size.
            dropout: Dropout rate.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(batch, seq_len, input_dim)``

        Returns:
            ``(batch, output_dim)``
        """
        h = self.proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        # Mean-pool across the sequence dimension.
        h = h.mean(dim=1)
        return self.fc(h)  # type: ignore[no-any-return]


def _prepare_sequences(
    data: np.ndarray,  # type: ignore[type-arg]
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Create sliding-window sequences for training.

    Last column of the *next* row is the target.

    Args:
        data: 2-D feature array ``(T, F)``.
        seq_len: Window length.

    Returns:
        Tuple ``(X, y)``.
    """
    xs: List[np.ndarray] = []  # type: ignore[type-arg]
    ys: List[float] = []
    for i in range(len(data) - seq_len):
        xs.append(data[i : i + seq_len])
        ys.append(float(data[i + seq_len, -1]))
    return np.array(xs), np.array(ys)


class TransformerPredictor:
    """High-level Transformer predictor.

    API mirrors :class:`LSTMPredictor` for drop-in
    replacement.

    Args:
        input_dim: Features per time step.
        seq_len: Sliding window length.
        d_model: Internal embedding dimension.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        output_dim: Prediction dimension.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.
        epochs: Training epochs.
        batch_size: Mini-batch size.
    """

    def __init__(
        self,
        input_dim: int = 10,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
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
        self._model = _TransformerNet(
            input_dim,
            d_model,
            n_heads,
            n_layers,
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
        """Train on a 2-D feature matrix.

        Last column is used as target.

        Args:
            data: ``(T, input_dim)`` where ``T > seq_len``.

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
            "Transformer trained: %d epochs, loss=%.6f",
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
        """Predict the next value.

        Args:
            sequence: ``(seq_len, input_dim)``

        Returns:
            Scalar prediction.

        Raises:
            RuntimeError: If not trained.
        """
        if not self._trained:
            raise RuntimeError("Call train() first")

        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :]
        tensor = torch.from_numpy(arr)
        self._model.eval()
        with torch.no_grad():
            return float(self._model(tensor).item())

    def predict_batch(
        self,
        sequences: np.ndarray,  # type: ignore[type-arg]
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Predict a batch of sequences.

        Args:
            sequences: ``(batch, seq_len, input_dim)``

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
        return out.squeeze(1).detach().numpy()  # type: ignore[no-any-return]

    def summary(self) -> Dict[str, Any]:
        """Return model summary.

        Returns:
            Dict with architecture and training info.
        """
        total = sum(p.numel() for p in self._model.parameters())
        return {
            "input_dim": self._input_dim,
            "seq_len": self._seq_len,
            "parameters": total,
            "trained": self._trained,
        }
