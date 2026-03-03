"""Tests for Phase 4 ML model optimization hooks."""

import asyncio

import numpy as np

from python_ai.ml_model import MLModel


def test_predict_cache_returns_stable_results(tmp_path) -> None:
    model = MLModel(model_path=str(tmp_path / "model.pkl"))
    features = {"sma_14": 1.0, "rsi_14": 50.0}
    first = model.predict(features)
    second = model.predict(features)
    assert first == second


def test_async_save_and_load(tmp_path) -> None:
    path = tmp_path / "model.pkl"
    model = MLModel(model_path=str(path))
    asyncio.run(model.save_async())

    other = MLModel(model_path=str(path))
    asyncio.run(other.load_async())
    assert other.is_trained is True


def test_predict_with_onnx_runtime_fallback(tmp_path) -> None:
    model = MLModel(model_path=str(tmp_path / "model.pkl"))
    pred = model.predict_with_onnx_runtime({"sma_14": 1.0, "rsi_14": 50.0})
    assert len(pred) == 3
    assert isinstance(pred[0], (float, np.floating))
