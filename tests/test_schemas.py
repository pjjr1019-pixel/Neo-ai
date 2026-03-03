"""Tests for shared API schemas."""

import pytest
from pydantic import ValidationError

from python_ai.schemas import LearnRequest, PredictRequest


def test_predict_request_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(features={"rsi": float("nan")})


def test_learn_request_accepts_valid_payload() -> None:
    payload = LearnRequest(features=[1.0, 2.0], target=0.5)
    assert payload.target == 0.5
    assert payload.features == [1.0, 2.0]
