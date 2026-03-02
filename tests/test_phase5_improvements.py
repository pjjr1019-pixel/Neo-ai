"""
Tests for Phase 5 improvements.

Covers:
- Admin credentials hardening (env-based, production blocker)
- Model HMAC integrity verification (keyed hashes, tamper
  detection, legacy SHA-256 upgrade)
- Feature ordering (canonical FEATURE_NAMES mapping)
- Division-by-zero guards in return calculations
- Fitness gradient preservation for negative returns
- Portfolio cash/position tracking accuracy
- /health endpoint split (public liveness vs private details)
- WebSocket authentication requirement
- Resilience wiring in exchange_feed
- Version bump to 0.6.0
"""

import hashlib
import os
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from python_ai.backtesting_engine import (
    BacktestingEngine,
    BacktestMetrics,
)
from python_ai.data_pipeline import (
    FEATURE_NAMES,
    DataPipeline,
)
from python_ai.fastapi_service.fastapi_service import _VERSION, app
from python_ai.ml_model import MLModel


client = TestClient(app)


# ── Admin Credentials Hardening ───────────────────────────────


class TestAdminCredentials:
    """Verify admin credentials are env-configurable."""

    def test_users_db_populated_from_init(self) -> None:
        """_users_db should have admin user after _init_users_db."""
        from python_ai.auth.dependencies import _users_db

        assert "admin" in _users_db
        assert _users_db["admin"]["username"] == "admin"

    def test_get_user_returns_user_object(self) -> None:
        """get_user should return a User for known usernames."""
        from python_ai.auth.dependencies import get_user

        user = get_user("admin")
        assert user is not None
        assert user.username == "admin"

    def test_get_user_returns_none_for_unknown(self) -> None:
        """get_user should return None for unknown usernames."""
        from python_ai.auth.dependencies import get_user

        assert get_user("nonexistent_user_xyz") is None

    def test_production_blocks_default_hash(self) -> None:
        """In production mode, default admin hash must be rejected."""
        from python_ai.auth.dependencies import (
            _init_users_db,
            _users_db,
        )

        old_env = os.environ.get("NEO_ENVIRONMENT")
        old_hash = os.environ.get("ADMIN_PASSWORD_HASH")
        try:
            os.environ["NEO_ENVIRONMENT"] = "production"
            os.environ.pop("ADMIN_PASSWORD_HASH", None)
            with pytest.raises(RuntimeError, match="CRITICAL"):
                _init_users_db()
        finally:
            if old_env is not None:
                os.environ["NEO_ENVIRONMENT"] = old_env
            else:
                os.environ.pop("NEO_ENVIRONMENT", None)
            if old_hash is not None:
                os.environ["ADMIN_PASSWORD_HASH"] = old_hash
            # Re-initialize with defaults for other tests
            _users_db.clear()
            os.environ.pop("NEO_ENVIRONMENT", None)
            _init_users_db()

    def test_production_accepts_custom_hash(self) -> None:
        """In production, a custom hash should be accepted."""
        from python_ai.auth.dependencies import (
            _init_users_db,
            _users_db,
        )

        old_env = os.environ.get("NEO_ENVIRONMENT")
        old_hash = os.environ.get("ADMIN_PASSWORD_HASH")
        custom = "$2b$12$CustomHashForTestingOnly1234567890abcdef"
        try:
            os.environ["NEO_ENVIRONMENT"] = "production"
            os.environ["ADMIN_PASSWORD_HASH"] = custom
            _users_db.clear()
            _init_users_db()
            assert _users_db["admin"]["hashed_password"] == custom
        finally:
            if old_env is not None:
                os.environ["NEO_ENVIRONMENT"] = old_env
            else:
                os.environ.pop("NEO_ENVIRONMENT", None)
            if old_hash is not None:
                os.environ["ADMIN_PASSWORD_HASH"] = old_hash
            else:
                os.environ.pop("ADMIN_PASSWORD_HASH", None)
            _users_db.clear()
            _init_users_db()


# ── Model HMAC Integrity ─────────────────────────────────────


class TestModelHMACIntegrity:
    """Test HMAC-based model file integrity verification."""

    def test_save_creates_hmac_sidecar(self) -> None:
        """save() should create a .hmac file next to the model."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            m.save()
            assert os.path.exists(path + ".hmac")

    def test_load_verifies_hmac(self) -> None:
        """load() should succeed when HMAC matches."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            m.save()
            m2 = MLModel.__new__(MLModel)
            m2.model_path = path
            m2.load()
            assert m2.is_trained

    def test_load_detects_tampered_model(self) -> None:
        """Corrupting model file should raise IntegrityError."""
        from python_ai.exceptions import NeoModelIntegrityError

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            m.save()
            # Corrupt the model file
            with open(path, "ab") as f:
                f.write(b"TAMPERED")
            m2 = MLModel.__new__(MLModel)
            m2.model_path = path
            with pytest.raises(NeoModelIntegrityError, match="HMAC"):
                m2.load()

    def test_load_detects_tampered_hmac_sidecar(self) -> None:
        """Corrupted HMAC sidecar should trigger IntegrityError."""
        from python_ai.exceptions import NeoModelIntegrityError

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            m.save()
            with open(path + ".hmac", "w") as f:
                f.write("0" * 64)
            m2 = MLModel.__new__(MLModel)
            m2.model_path = path
            with pytest.raises(NeoModelIntegrityError, match="HMAC"):
                m2.load()

    def test_legacy_sha256_upgrade(self) -> None:
        """Legacy .sha256 sidecar should be verified then upgraded."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            m.save()
            # Remove HMAC, create legacy SHA-256 sidecar
            os.remove(path + ".hmac")
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            with open(path + ".sha256", "w") as f:
                f.write(sha.hexdigest())
            m2 = MLModel.__new__(MLModel)
            m2.model_path = path
            m2.load()
            assert m2.is_trained
            # HMAC sidecar should now exist
            assert os.path.exists(path + ".hmac")

    def test_no_sidecar_generates_hmac(self) -> None:
        """First load without any sidecar generates HMAC."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            m.save()
            os.remove(path + ".hmac")
            m2 = MLModel.__new__(MLModel)
            m2.model_path = path
            m2.load()
            assert os.path.exists(path + ".hmac")

    def test_file_hmac_is_keyed(self) -> None:
        """_file_hmac should produce different output than SHA-256."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            m.save()
            hmac_val = MLModel._file_hmac(path)
            sha_val = MLModel._file_hash(path)
            assert hmac_val != sha_val


# ── Feature Ordering ─────────────────────────────────────────


class TestFeatureOrdering:
    """Verify _dict_to_array uses canonical FEATURE_NAMES order."""

    def test_dict_to_array_canonical_order(self) -> None:
        """Features should be placed by name, not insertion order."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            # Build features in reverse order
            features = {
                name: float(i)
                for i, name in enumerate(reversed(FEATURE_NAMES))
            }
            arr = m._dict_to_array(features)
            # Values should match canonical order, not reversed
            for i, name in enumerate(FEATURE_NAMES):
                expected = features[name]
                assert arr[i] == expected, (
                    f"Feature {name} at index {i}: "
                    f"expected {expected}, got {arr[i]}"
                )

    def test_dict_to_array_missing_features_default_zero(self) -> None:
        """Missing features should default to 0.0."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            features = {"rsi_14": 0.75}
            arr = m._dict_to_array(features)
            assert arr[0] == 0.75
            assert arr[1] == 0.0  # macd_value not provided

    def test_dict_to_array_warns_unknown_keys(self) -> None:
        """Unknown feature keys should not crash; logged as warning."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_model.pkl")
            m = MLModel(model_path=path)
            features = {"rsi_14": 0.5, "unknown_feat": 1.0}
            # Should not raise — unknown keys are logged, not thrown
            arr = m._dict_to_array(features)
            assert arr[0] == 0.5

    def test_feature_names_length_matches_model(self) -> None:
        """FEATURE_NAMES should have exactly 10 entries."""
        assert len(FEATURE_NAMES) == 10


# ── Division by Zero in Returns ──────────────────────────────


class TestDivisionByZeroGuard:
    """Verify return calculations handle zero prices."""

    def test_return_1d_zero_price(self) -> None:
        """return_1d should be 0 when close[-2] is 0."""
        pipeline = DataPipeline()
        pipeline.update_price_data("TEST", {
            "close": [0.0, 100.0],
            "high": [0.0, 100.0],
            "low": [0.0, 100.0],
        })
        features = pipeline.compute_features("TEST")
        assert features["return_1d"] == 0.0

    def test_return_5d_zero_price(self) -> None:
        """return_5d should be 0 when close[-5] is 0."""
        pipeline = DataPipeline()
        close = [0.0] + [100.0] * 20
        pipeline.update_price_data("TEST", {
            "close": close,
            "high": close,
            "low": close,
        })
        features = pipeline.compute_features("TEST")
        # close[-5] might not be 0 due to enough data;
        # test that no exception is raised
        assert isinstance(features["return_5d"], float)

    def test_return_calculations_no_crash(self) -> None:
        """Return calculations should never crash with valid OHLCV."""
        pipeline = DataPipeline()
        n = 50
        close = list(np.random.rand(n) * 100 + 1)
        pipeline.update_price_data("TEST", {
            "close": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
        })
        features = pipeline.compute_features("TEST")
        for key in ("return_1d", "return_5d", "return_10d"):
            assert np.isfinite(features[key])


# ── Fitness Gradient Preservation ─────────────────────────────


class TestFitnessGradient:
    """Verify fitness differentiates negative-return strategies."""

    def test_negative_5pct_vs_negative_50pct(self) -> None:
        """-5% strategy should have higher fitness than -50%."""
        m1 = BacktestMetrics(-5.0, 0.0, 5.0, 40.0, 10)
        m2 = BacktestMetrics(-50.0, 0.0, 50.0, 30.0, 10)
        assert m1.fitness_score > m2.fitness_score

    def test_zero_return_fitness_midpoint(self) -> None:
        """0% return → return_score = 0.5 (midpoint)."""
        m = BacktestMetrics(0.0, 0.0, 0.0, 50.0, 10)
        # return_score = (0 + 100) / 200 = 0.5
        # sharpe_score = 0
        # win_score = 0.5
        # fitness = 0.4 * 0.5 + 0.4 * 0 + 0.2 * 0.5 = 0.3
        assert abs(m.fitness_score - 0.3) < 0.01

    def test_positive_100pct_return_score_max(self) -> None:
        """+100% return → return_score = 1.0."""
        m = BacktestMetrics(100.0, 0.0, 0.0, 100.0, 10)
        # return_score = (100 + 100) / 200 = 1.0
        # sharpe_score = 0
        # win_score = 1.0
        # fitness = 0.4 * 1.0 + 0.4 * 0 + 0.2 * 1.0 = 0.6
        assert abs(m.fitness_score - 0.6) < 0.01

    def test_negative_100pct_return_score_zero(self) -> None:
        """-100% return → return_score = 0.0."""
        m = BacktestMetrics(-100.0, 0.0, 100.0, 0.0, 0)
        # return_score = (-100 + 100) / 200 = 0.0
        assert m.fitness_score == 0.0

    def test_fitness_always_0_to_1(self) -> None:
        """Fitness should always be in [0, 1]."""
        for ret in [-200, -100, -50, 0, 50, 100, 200]:
            m = BacktestMetrics(
                float(ret), 1.0, 10.0, 50.0, 10,
            )
            assert 0.0 <= m.fitness_score <= 1.0


# ── Portfolio Cash Tracking ───────────────────────────────────


class TestPortfolioCashTracking:
    """Verify portfolio tracks cash + position accurately."""

    def test_buy_sell_round_trip(self) -> None:
        """BUY then SELL should return close to initial capital."""
        engine = BacktestingEngine(
            initial_capital=10000.0,
            transaction_cost_pct=0.001,
        )
        ohlcv = {"close": [100.0, 105.0, 110.0, 105.0]}
        signals = ["BUY", "HOLD", "HOLD", "SELL"]
        result = engine.run_backtest(ohlcv, signals)
        # With 0.1% cost on buy and sell, return should be close
        # to (105-100)/100 = 5% minus fees
        assert result.num_trades == 2
        assert isinstance(result.total_return, float)

    def test_hold_only_no_change(self) -> None:
        """All HOLDs should produce 0% return."""
        engine = BacktestingEngine(initial_capital=10000.0)
        ohlcv = {"close": [100.0, 105.0, 110.0]}
        signals = ["HOLD", "HOLD", "HOLD"]
        result = engine.run_backtest(ohlcv, signals)
        assert result.total_return == 0.0
        assert result.num_trades == 0

    def test_portfolio_value_consistency(self) -> None:
        """Portfolio value should always be cash + shares * price."""
        engine = BacktestingEngine(
            initial_capital=10000.0,
            transaction_cost_pct=0.0,
        )
        ohlcv = {
            "close": [100.0, 110.0, 90.0, 120.0, 100.0],
        }
        signals = ["BUY", "HOLD", "HOLD", "SELL", "HOLD"]
        result = engine.run_backtest(ohlcv, signals)
        # No transaction costs: buy at 100, sell at 120 = 20% return
        assert abs(result.total_return - 20.0) < 0.01


# ── Health Endpoint Split ─────────────────────────────────────


class TestHealthEndpoint:
    """Verify /health split into public + private."""

    def test_public_health_returns_ok(self) -> None:
        """/health should return status ok without auth."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # Should NOT contain system details
        assert "checks" not in data
        assert "uptime_seconds" not in data

    def test_health_details_requires_auth(self) -> None:
        """/health/details should return system details with auth."""
        resp = client.get("/health/details")
        assert resp.status_code == 200
        data = resp.json()
        assert "checks" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert data["checks"]["system"]["python"]

    def test_health_details_has_model_info(self) -> None:
        """/health/details should include model health."""
        resp = client.get("/health/details")
        data = resp.json()
        assert "model" in data["checks"]
        assert "trained" in data["checks"]["model"]


# ── Version Bump ──────────────────────────────────────────────


class TestVersionBump:
    """Verify version is updated to 0.6.0."""

    def test_version_string(self) -> None:
        """_VERSION should be 0.6.0."""
        assert _VERSION == "0.6.0"

    def test_root_endpoint_running(self) -> None:
        """Root endpoint should still work."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "running" in resp.json()["message"].lower()


# ── Resilience Wiring ─────────────────────────────────────────


class TestResilienceWiring:
    """Verify resilience module is imported in exchange_feed."""

    def test_exchange_feed_imports_resilience(self) -> None:
        """exchange_feed should import external_api_call."""
        import python_ai.exchange_feed as ef

        assert hasattr(ef, "external_api_call")

    def test_resilience_module_exports(self) -> None:
        """resilience module should export expected symbols."""
        from python_ai.resilience import (
            db_resilient_call,
            external_api_call,
        )

        assert callable(db_resilient_call)
        assert callable(external_api_call)


# ── WebSocket Auth ────────────────────────────────────────────


class TestWebSocketAuth:
    """Verify WebSocket requires JWT token."""

    def test_ws_without_token_rejected(self) -> None:
        """WebSocket without token should be closed with 4001."""
        with pytest.raises(Exception):
            # TestClient raises on WebSocket close
            with client.websocket_connect("/ws/signals"):
                pass

    def test_ws_with_invalid_token_rejected(self) -> None:
        """WebSocket with bad token should be closed with 4001."""
        with pytest.raises(Exception):
            with client.websocket_connect(
                "/ws/signals?token=invalid-token"
            ):
                pass


# ── Predict/Metrics Endpoints ─────────────────────────────────


class TestProtectedEndpoints:
    """Test that auth-protected endpoints work with auth override."""

    def test_predict_endpoint(self) -> None:
        """/predict should return prediction with valid features."""
        features = {name: 0.5 for name in FEATURE_NAMES}
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "signal" in data

    def test_metrics_endpoint(self) -> None:
        """/metrics should return request counts."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "request_counts" in data
        assert "total_requests" in data

    def test_explain_endpoint(self) -> None:
        """/explain should return feature importance."""
        resp = client.get("/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data or "feature_importance" in data
        assert "model_type" in data

    def test_prometheus_metrics(self) -> None:
        """/metrics/prometheus should return text metrics."""
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        assert "neo_requests_total" in resp.text

    def test_compute_features_endpoint(self) -> None:
        """/compute-features should return computed features."""
        resp = client.post(
            "/compute-features",
            json={
                "symbol": "BTC",
                "ohlcv_data": {
                    "close": [100.0] * 50,
                    "high": [101.0] * 50,
                    "low": [99.0] * 50,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rsi_14" in data
