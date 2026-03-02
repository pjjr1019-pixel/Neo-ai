"""Tests for OHLCV and feature data validation.

Covers:
- validate_ohlcv: missing columns, length mismatch, empty data,
  NaN/Inf, high/low violations, negative volume, error truncation
- validate_features: missing features, non-numeric, NaN/Inf
"""

import math

import pytest

from python_ai.data_pipeline import (
    FEATURE_NAMES,
    validate_features,
    validate_ohlcv,
)


# ── validate_ohlcv ───────────────────────────────────────────


class TestValidateOhlcv:
    """Tests for validate_ohlcv."""

    @staticmethod
    def _good_ohlcv(n=3):
        """Return valid OHLCV data with *n* bars."""
        return {
            "open": [100.0] * n,
            "high": [110.0] * n,
            "low": [90.0] * n,
            "close": [105.0] * n,
            "volume": [1000.0] * n,
        }

    def test_valid_data_no_errors(self):
        """Clean data passes validation."""
        errors = validate_ohlcv(self._good_ohlcv())
        assert errors == []

    def test_missing_column(self):
        """Missing required column is reported."""
        data = self._good_ohlcv()
        del data["high"]
        errors = validate_ohlcv(data)
        assert any("Missing" in e for e in errors)

    def test_length_mismatch(self):
        """Columns of different lengths are reported."""
        data = self._good_ohlcv()
        data["close"] = [105.0, 106.0]
        errors = validate_ohlcv(data)
        assert any("length" in e.lower() for e in errors)

    def test_empty_data(self):
        """Empty OHLCV arrays are rejected."""
        data = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
        }
        errors = validate_ohlcv(data)
        assert any("empty" in e.lower() for e in errors)

    def test_nan_in_ohlc(self):
        """NaN values are detected."""
        data = self._good_ohlcv(1)
        data["close"] = [float("nan")]
        errors = validate_ohlcv(data)
        assert any("NaN" in e for e in errors)

    def test_inf_in_ohlc(self):
        """Inf values are detected."""
        data = self._good_ohlcv(1)
        data["open"] = [float("inf")]
        errors = validate_ohlcv(data)
        assert any("Inf" in e or "NaN" in e for e in errors)

    def test_high_less_than_low(self):
        """high < low is reported."""
        data = self._good_ohlcv(1)
        data["high"] = [80.0]
        data["low"] = [90.0]
        errors = validate_ohlcv(data)
        assert any("high" in e and "low" in e for e in errors)

    def test_high_less_than_close(self):
        """high < close is reported."""
        data = self._good_ohlcv(1)
        data["high"] = [100.0]
        data["close"] = [110.0]
        errors = validate_ohlcv(data)
        assert any("high" in e and "close" in e for e in errors)

    def test_negative_volume(self):
        """Negative volume is reported."""
        data = self._good_ohlcv(1)
        data["volume"] = [-5.0]
        errors = validate_ohlcv(data)
        assert any("volume" in e.lower() for e in errors)

    def test_error_truncation(self):
        """Errors are capped at ~20."""
        # Create 25 bars all with high < low
        data = {
            "open": [100.0] * 25,
            "high": [80.0] * 25,
            "low": [90.0] * 25,
            "close": [85.0] * 25,
        }
        errors = validate_ohlcv(data)
        assert len(errors) <= 22  # 20 + truncation msg + slack
        assert any("truncated" in e for e in errors)

    def test_volume_optional(self):
        """Volume column is optional."""
        data = {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
        }
        errors = validate_ohlcv(data)
        assert errors == []


# ── validate_features ─────────────────────────────────────────


class TestValidateFeatures:
    """Tests for validate_features."""

    @staticmethod
    def _good_features():
        """Return a valid feature dict."""
        return {name: 0.5 for name in FEATURE_NAMES}

    def test_valid_features_no_errors(self):
        """Complete, finite features pass."""
        errors = validate_features(self._good_features())
        assert errors == []

    def test_missing_feature(self):
        """Missing feature name is reported."""
        feats = self._good_features()
        del feats["rsi_14"]
        errors = validate_features(feats)
        assert any("rsi_14" in e for e in errors)

    def test_nan_feature(self):
        """NaN feature value is detected."""
        feats = self._good_features()
        feats["atr_14"] = float("nan")
        errors = validate_features(feats)
        assert any("NaN" in e or "Inf" in e for e in errors)

    def test_inf_feature(self):
        """Inf feature value is detected."""
        feats = self._good_features()
        feats["macd_value"] = float("inf")
        errors = validate_features(feats)
        assert any("Inf" in e or "NaN" in e for e in errors)

    def test_non_numeric_feature(self):
        """Non-numeric feature value is detected."""
        feats = self._good_features()
        feats["rsi_14"] = "not_a_number"
        errors = validate_features(feats)
        assert any("not numeric" in e for e in errors)

    def test_extra_features_ok(self):
        """Extra features beyond canonical set don't cause error."""
        feats = self._good_features()
        feats["extra_col"] = 1.0
        errors = validate_features(feats)
        assert errors == []

    def test_integer_values_accepted(self):
        """Integer values are accepted as numeric."""
        feats = {name: 0 for name in FEATURE_NAMES}
        errors = validate_features(feats)
        assert errors == []
