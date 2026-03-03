"""Shared API request/response schemas for FastAPI endpoints."""

from __future__ import annotations

import math
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class ComputeFeaturesRequest(BaseModel):
    """Request payload for feature computation."""

    symbol: str
    ohlcv_data: Dict[str, List[float]]


class PredictRequest(BaseModel):
    """Request payload for prediction."""

    features: Dict[str, float] = Field(..., min_length=1)

    @field_validator("features")
    @classmethod
    def validate_feature_values(
        cls,
        value: Dict[str, float],
    ) -> Dict[str, float]:
        """Reject NaN/Inf values from predict payload."""
        for name, feature in value.items():
            if math.isnan(feature) or math.isinf(feature):
                raise ValueError(f"Feature '{name}' contains NaN/Inf")
        return value


class PredictResponse(BaseModel):
    """Response payload for prediction."""

    prediction: float
    confidence: float
    signal: str


class LearnRequest(BaseModel):
    """Request payload for online learning."""

    features: List[float] = Field(..., min_length=1, max_length=1000)
    target: float = Field(..., ge=-1e6, le=1e6)

    @field_validator("features")
    @classmethod
    def validate_feature_vector(cls, value: List[float]) -> List[float]:
        """Reject NaN/Inf values from learning feature vectors."""
        for idx, item in enumerate(value):
            if math.isnan(item) or math.isinf(item):
                raise ValueError(f"features[{idx}] contains NaN/Inf")
        return value


class LearnResponse(BaseModel):
    """Response payload for learning endpoint."""

    status: str
    buffer_size: int | None = None
    retrain_at: int | None = None
