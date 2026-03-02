"""
Tests for Session 4 modules.

Covers 18 new modules:
- model_versioning.py
- retry.py
- graceful_degradation.py
- timeout.py
- var_risk.py
- correlation_matrix.py
- portfolio_rebalancer.py
- monte_carlo.py
- walk_forward.py
- ws_signal_stream.py
- config_hot_reload.py
- ab_testing.py
- multi_exchange_aggregator.py
- api_versioning.py
- hmac_signing.py
- vectorized_indicators.py
- circular_buffer.py
- alembic migrations (schema check)
"""

import asyncio
import json
import time

import numpy as np
import pytest

# ────────────────────────────────────────────────────────
# 1. Model Versioning
# ────────────────────────────────────────────────────────


class TestModelVersioning:
    """Tests for model_versioning.py."""

    def test_register_and_list(self, tmp_path):
        """Register a model and list versions."""
        from python_ai.model_versioning import (
            ModelRegistry,
        )

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"fake-model-bytes")

        reg = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        mv = reg.register(str(model_file))
        assert mv.version == "v1.0.0"
        assert len(reg.list_versions()) == 1

    def test_latest(self, tmp_path):
        """Latest returns the most recent version."""
        from python_ai.model_versioning import (
            ModelRegistry,
        )

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"data")
        reg = ModelRegistry(str(tmp_path / "registry"))
        reg.register(str(model_file))
        reg.register(str(model_file))
        assert reg.latest().version == "v2.0.0"

    def test_get_by_version(self, tmp_path):
        """Look up a specific version."""
        from python_ai.model_versioning import (
            ModelRegistry,
        )

        model_file = tmp_path / "m.pkl"
        model_file.write_bytes(b"data")
        reg = ModelRegistry(str(tmp_path / "registry"))
        reg.register(str(model_file))
        assert reg.get("v1.0.0") is not None
        assert reg.get("v99.0.0") is None

    def test_rollback(self, tmp_path):
        """Rollback returns the model path."""
        from python_ai.model_versioning import (
            ModelRegistry,
        )

        model_file = tmp_path / "m.pkl"
        model_file.write_bytes(b"data")
        reg = ModelRegistry(str(tmp_path / "registry"))
        reg.register(str(model_file))
        path = reg.rollback("v1.0.0")
        assert path is not None
        assert reg.rollback("v99.0.0") is None

    def test_summary(self, tmp_path):
        """Summary includes version list."""
        from python_ai.model_versioning import (
            ModelRegistry,
        )

        model_file = tmp_path / "m.pkl"
        model_file.write_bytes(b"data")
        reg = ModelRegistry(str(tmp_path / "registry"))
        reg.register(str(model_file))
        s = reg.summary()
        assert s["total_versions"] == 1

    def test_hash_data(self):
        """hash_data returns a hex string."""
        from python_ai.model_versioning import hash_data

        h = hash_data(b"hello")
        assert len(h) == 64

    def test_model_version_serialise(self):
        """To/from dict round-trip."""
        from python_ai.model_versioning import (
            ModelVersion,
        )

        mv = ModelVersion(
            version="v1.0.0",
            created_at=1.0,
            model_path="/tmp/m.pkl",
            metrics={"acc": 0.9},
        )
        d = mv.to_dict()
        mv2 = ModelVersion.from_dict(d)
        assert mv2.version == mv.version
        assert mv2.metrics == mv.metrics


# ────────────────────────────────────────────────────────
# 2. Retry
# ────────────────────────────────────────────────────────


class TestRetry:
    """Tests for retry.py."""

    def test_succeeds_first_try(self):
        """No retries needed on success."""
        from python_ai.retry import retry

        @retry(max_attempts=3, base_delay=0.01)
        def ok():
            """Return success."""
            return 42

        assert ok() == 42

    def test_retries_on_failure(self):
        """Retries transient errors."""
        from python_ai.retry import retry

        call_count = {"n": 0}

        @retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        def flaky():
            """Fail first, succeed second."""
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("transient")
            return "ok"

        assert flaky() == "ok"
        assert call_count["n"] == 2

    def test_exhausts_retries(self):
        """Raises after max attempts."""
        from python_ai.retry import retry

        @retry(
            max_attempts=2,
            base_delay=0.01,
            retryable_exceptions=(RuntimeError,),
        )
        def always_fail():
            """Always fail."""
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError):
            always_fail()

    def test_retry_call_api(self):
        """Non-decorator retry_call works."""
        from python_ai.retry import retry_call

        result = retry_call(
            lambda: 10,
            max_attempts=2,
            base_delay=0.01,
        )
        assert result == 10


# ────────────────────────────────────────────────────────
# 3. Graceful Degradation
# ────────────────────────────────────────────────────────


class TestGracefulDegradation:
    """Tests for graceful_degradation.py."""

    def test_primary_succeeds(self):
        """Returns primary result when healthy."""
        from python_ai.graceful_degradation import (
            GracefulDegradation,
        )

        gd = GracefulDegradation(name="test")
        r = gd.call(lambda: 42)
        assert r.value == 42
        assert not r.degraded

    def test_fallback_used(self):
        """Falls back on primary failure."""
        from python_ai.graceful_degradation import (
            GracefulDegradation,
        )

        gd = GracefulDegradation(name="test")
        gd.add_fallback("cache", lambda: "cached-val")

        def bad():
            """Always fail."""
            raise ConnectionError("down")

        r = gd.call(bad)
        assert r.value == "cached-val"
        assert r.degraded
        assert r.source == "cache"

    def test_default_when_all_fail(self):
        """Returns default when all fallbacks fail."""
        from python_ai.graceful_degradation import (
            GracefulDegradation,
        )

        gd = GracefulDegradation(name="test", default=-1)

        def bad():
            """Always fail."""
            raise RuntimeError("nope")

        r = gd.call(bad)
        assert r.value == -1
        assert r.degraded
        assert r.source == "default"

    def test_cache_backed_fallback(self, tmp_path):
        """CacheBackedFallback caches and returns."""
        from python_ai.graceful_degradation import (
            CacheBackedFallback,
        )

        cb = CacheBackedFallback(ttl=60.0)
        r1 = cb.call_with_cache("k1", lambda: 100)
        assert r1.value == 100
        assert not r1.degraded

        def fail():
            """Simulate failure."""
            raise ConnectionError("down")

        r2 = cb.call_with_cache("k1", fail)
        assert r2.value == 100
        assert r2.degraded

    def test_stats(self):
        """Stats track primary and fallback usage."""
        from python_ai.graceful_degradation import (
            GracefulDegradation,
        )

        gd = GracefulDegradation()
        gd.call(lambda: 1)
        s = gd.stats()
        assert s["primary_ok"] == 1

    def test_fallback_result_to_dict(self):
        """FallbackResult serialises."""
        from python_ai.graceful_degradation import (
            FallbackResult,
        )

        fr = FallbackResult(42, True, "cache")
        d = fr.to_dict()
        assert d["value"] == 42
        assert d["degraded"] is True


# ────────────────────────────────────────────────────────
# 4. Timeout
# ────────────────────────────────────────────────────────


class TestTimeout:
    """Tests for timeout.py."""

    def test_fast_function_succeeds(self):
        """Fast function returns within timeout."""
        from python_ai.timeout import with_timeout

        @with_timeout(5.0)
        def fast():
            """Return quickly."""
            return "done"

        assert fast() == "done"

    def test_timeout_call_api(self):
        """Non-decorator timeout_call works."""
        from python_ai.timeout import timeout_call

        result = timeout_call(lambda: 99, timeout=5.0)
        assert result == 99

    def test_propagates_exceptions(self):
        """Exceptions from the function propagate."""
        from python_ai.timeout import with_timeout

        @with_timeout(5.0)
        def bad():
            """Always raises."""
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            bad()


# ────────────────────────────────────────────────────────
# 5. VaR Risk
# ────────────────────────────────────────────────────────


class TestVarRisk:
    """Tests for var_risk.py."""

    def test_historical_var(self):
        """Historical VaR produces a positive value."""
        from python_ai.var_risk import historical_var

        returns = [
            0.01,
            -0.02,
            0.03,
            -0.01,
            0.005,
            -0.03,
            0.02,
            -0.015,
            0.01,
            -0.005,
        ] * 5
        v = historical_var(returns, 0.95)
        assert v >= 0.0

    def test_parametric_var(self):
        """Parametric VaR with portfolio value."""
        from python_ai.var_risk import parametric_var

        returns = [0.01, -0.02, 0.015, -0.005] * 10
        v = parametric_var(returns, 0.95, portfolio_value=10000)
        assert v >= 0.0

    def test_monte_carlo_var(self):
        """Monte Carlo VaR is deterministic with seed."""
        from python_ai.var_risk import monte_carlo_var

        returns = [0.01, -0.02, 0.015, -0.005] * 10
        v1 = monte_carlo_var(returns, 0.95, simulations=1000, seed=42)
        v2 = monte_carlo_var(returns, 0.95, simulations=1000, seed=42)
        assert v1 == v2

    def test_conditional_var(self):
        """CVaR is >= VaR."""
        from python_ai.var_risk import (
            conditional_var,
            historical_var,
        )

        returns = [
            0.01,
            -0.02,
            0.03,
            -0.04,
            0.005,
            -0.03,
            0.02,
            -0.015,
            0.01,
            -0.05,
        ] * 5
        var = historical_var(returns, 0.95)
        cvar = conditional_var(returns, 0.95)
        assert cvar >= var or abs(cvar - var) < 0.01

    def test_portfolio_var(self):
        """Portfolio VaR combines assets."""
        from python_ai.var_risk import portfolio_var

        rets = {
            "BTC": [0.01, -0.02, 0.03, -0.01] * 10,
            "ETH": [0.02, -0.03, 0.01, -0.02] * 10,
        }
        weights = {"BTC": 0.6, "ETH": 0.4}
        v = portfolio_var(rets, weights, 0.95)
        assert v >= 0.0

    def test_empty_returns(self):
        """Short series returns 0."""
        from python_ai.var_risk import historical_var

        assert historical_var([], 0.95) == 0.0
        assert historical_var([0.01], 0.95) == 0.0


# ────────────────────────────────────────────────────────
# 6. Correlation Matrix
# ────────────────────────────────────────────────────────


class TestCorrelationMatrix:
    """Tests for correlation_matrix.py."""

    def test_self_correlation_is_one(self):
        """Asset correlates perfectly with itself."""
        from python_ai.correlation_matrix import (
            compute_correlation,
        )

        rets = {
            "BTC": [0.01, -0.02, 0.03] * 10,
            "ETH": [0.02, -0.01, 0.01] * 10,
        }
        corr = compute_correlation(rets)
        assert abs(corr["BTC"]["BTC"] - 1.0) < 1e-6

    def test_covariance_matrix(self):
        """Covariance matrix is symmetric."""
        from python_ai.correlation_matrix import (
            compute_covariance,
        )

        rets = {
            "A": [0.01, -0.02, 0.03] * 10,
            "B": [0.02, -0.01, 0.01] * 10,
        }
        cov = compute_covariance(rets)
        assert abs(cov["A"]["B"] - cov["B"]["A"]) < 1e-10

    def test_rolling_correlation(self):
        """Rolling correlation returns correct len."""
        from python_ai.correlation_matrix import (
            rolling_correlation,
        )

        a = [float(i) for i in range(50)]
        b = [float(i * 2) for i in range(50)]
        rc = rolling_correlation(a, b, window=10)
        assert len(rc) == 50
        assert rc[0] is None
        assert rc[-1] is not None

    def test_ewma_correlation(self):
        """EWMA correlation returns values."""
        from python_ai.correlation_matrix import (
            ewma_correlation,
        )

        a = [0.01, -0.02, 0.03, -0.01] * 10
        b = [0.02, -0.01, 0.01, -0.03] * 10
        ec = ewma_correlation(a, b, span=10)
        assert len(ec) == 40
        assert ec[0] is None

    def test_diversification_ratio(self):
        """Diversification ratio >= 1."""
        from python_ai.correlation_matrix import (
            diversification_ratio,
        )

        rets = {
            "A": [0.01, -0.02, 0.03, -0.01] * 10,
            "B": [-0.01, 0.02, -0.03, 0.01] * 10,
        }
        weights = {"A": 0.5, "B": 0.5}
        dr = diversification_ratio(rets, weights)
        assert dr >= 1.0


# ────────────────────────────────────────────────────────
# 7. Portfolio Rebalancer
# ────────────────────────────────────────────────────────


class TestPortfolioRebalancer:
    """Tests for portfolio_rebalancer.py."""

    def test_no_rebalance_needed(self):
        """No orders when within threshold."""
        from python_ai.portfolio_rebalancer import (
            PortfolioRebalancer,
        )

        rb = PortfolioRebalancer(
            target_weights={"BTC": 0.5, "ETH": 0.5},
            threshold=0.1,
        )
        assert not rb.needs_rebalance({"BTC": 0.52, "ETH": 0.48})

    def test_rebalance_triggered(self):
        """Orders generated on threshold breach."""
        from python_ai.portfolio_rebalancer import (
            PortfolioRebalancer,
        )

        rb = PortfolioRebalancer(
            target_weights={"BTC": 0.5, "ETH": 0.5},
            threshold=0.05,
        )
        assert rb.needs_rebalance({"BTC": 0.7, "ETH": 0.3})
        orders = rb.generate_orders(
            {"BTC": 0.7, "ETH": 0.3},
            portfolio_value=10000,
        )
        assert len(orders) >= 1
        sides = {o.side for o in orders}
        assert "sell" in sides or "buy" in sides

    def test_order_serialise(self):
        """RebalanceOrder serialises to dict."""
        from python_ai.portfolio_rebalancer import (
            RebalanceOrder,
        )

        o = RebalanceOrder("BTC", "buy", 100.0)
        d = o.to_dict()
        assert d["symbol"] == "BTC"
        assert d["side"] == "buy"

    def test_summary(self):
        """Summary includes config."""
        from python_ai.portfolio_rebalancer import (
            PortfolioRebalancer,
        )

        rb = PortfolioRebalancer(target_weights={"BTC": 1.0})
        s = rb.summary()
        assert "threshold" in s

    def test_update_targets(self):
        """Targets can be updated."""
        from python_ai.portfolio_rebalancer import (
            PortfolioRebalancer,
        )

        rb = PortfolioRebalancer(target_weights={"BTC": 1.0})
        rb.update_targets({"ETH": 1.0})
        s = rb.summary()
        assert "ETH" in s["target_weights"]


# ────────────────────────────────────────────────────────
# 8. Monte Carlo
# ────────────────────────────────────────────────────────


class TestMonteCarlo:
    """Tests for monte_carlo.py."""

    def test_simulate_shape(self):
        """Paths array has correct shape."""
        from python_ai.monte_carlo import (
            MonteCarloSimulator,
        )

        mc = MonteCarloSimulator(seed=42)
        returns = [0.01, -0.02, 0.015, -0.005] * 10
        paths = mc.simulate(returns, n_paths=100, horizon=10)
        assert paths.shape == (100, 11)

    def test_confidence_interval(self):
        """Confidence interval has three bands."""
        from python_ai.monte_carlo import (
            MonteCarloSimulator,
        )

        mc = MonteCarloSimulator(seed=42)
        mc.simulate(
            [0.01, -0.02] * 20,
            n_paths=100,
            horizon=5,
        )
        ci = mc.confidence_interval(0.95)
        assert len(ci["lower"]) == 6
        assert len(ci["median"]) == 6

    def test_terminal_stats(self):
        """Terminal stats include mean and std."""
        from python_ai.monte_carlo import (
            MonteCarloSimulator,
        )

        mc = MonteCarloSimulator(seed=42)
        mc.simulate(
            [0.01, -0.005] * 20,
            n_paths=500,
            horizon=10,
        )
        ts = mc.terminal_stats()
        assert "mean" in ts
        assert "std" in ts

    def test_max_drawdown_distribution(self):
        """Drawdown stats are non-negative."""
        from python_ai.monte_carlo import (
            MonteCarloSimulator,
        )

        mc = MonteCarloSimulator(seed=42)
        mc.simulate(
            [0.01, -0.02] * 20,
            n_paths=200,
            horizon=10,
        )
        dd = mc.max_drawdown_distribution()
        assert dd["mean"] >= 0.0

    def test_probability_of_loss(self):
        """Prob of loss is between 0 and 1."""
        from python_ai.monte_carlo import (
            MonteCarloSimulator,
        )

        mc = MonteCarloSimulator(seed=42)
        mc.simulate(
            [0.01, -0.02] * 20,
            n_paths=200,
            horizon=10,
        )
        p = mc.probability_of_loss()
        assert 0.0 <= p <= 1.0

    def test_summary(self):
        """Summary includes all sections."""
        from python_ai.monte_carlo import (
            MonteCarloSimulator,
        )

        mc = MonteCarloSimulator(seed=42)
        mc.simulate([0.01, -0.02] * 20, n_paths=50, horizon=5)
        s = mc.summary()
        assert "terminal" in s
        assert "prob_loss" in s


# ────────────────────────────────────────────────────────
# 9. Walk-Forward
# ────────────────────────────────────────────────────────


class TestWalkForward:
    """Tests for walk_forward.py."""

    def test_basic_run(self):
        """Walk-forward produces fold results."""
        from python_ai.walk_forward import (
            WalkForwardOptimizer,
        )

        data = list(range(100))
        wf = WalkForwardOptimizer(n_folds=3, train_ratio=0.7)

        def opt_fn(train_data):
            """Return dummy params."""
            return {"threshold": 0.5}

        def eval_fn(data_slice, params):
            """Return dummy metric."""
            return sum(data_slice) / max(len(data_slice), 1)

        results = wf.run(data, opt_fn, eval_fn)
        assert len(results) >= 1

    def test_summary(self):
        """Summary includes overfitting ratio."""
        from python_ai.walk_forward import (
            WalkForwardOptimizer,
        )

        data = list(range(100))
        wf = WalkForwardOptimizer(n_folds=3)
        wf.run(
            data,
            lambda d: {"x": 1},
            lambda d, p: float(sum(d)) / max(len(d), 1),
        )
        s = wf.summary()
        assert "overfitting_ratio" in s

    def test_anchored_mode(self):
        """Anchored mode always starts at 0."""
        from python_ai.walk_forward import (
            WalkForwardOptimizer,
        )

        data = list(range(100))
        wf = WalkForwardOptimizer(n_folds=2, anchored=True)
        results = wf.run(
            data,
            lambda d: {},
            lambda d, p: 1.0,
        )
        for r in results:
            assert r.train_range[0] == 0

    def test_fold_serialise(self):
        """WalkForwardResult serialises to dict."""
        from python_ai.walk_forward import (
            WalkForwardResult,
        )

        r = WalkForwardResult(
            fold=0,
            train_range=(0, 70),
            test_range=(70, 100),
            train_metric=0.8,
            test_metric=0.75,
        )
        d = r.to_dict()
        assert d["fold"] == 0


# ────────────────────────────────────────────────────────
# 10. WebSocket Signal Streaming
# ────────────────────────────────────────────────────────


class TestSignalBroadcaster:
    """Tests for ws_signal_stream.py."""

    @pytest.mark.asyncio
    async def test_subscribe_and_broadcast(self):
        """Subscriber receives broadcast signal."""
        from python_ai.ws_signal_stream import (
            SignalBroadcaster,
        )

        bc = SignalBroadcaster()
        q = await bc.subscribe()
        await bc.broadcast({"signal": "buy"})
        msg = await asyncio.wait_for(q.get(), timeout=1)
        data = json.loads(msg)
        assert data["signal"] == "buy"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Unsubscribed client no longer receives."""
        from python_ai.ws_signal_stream import (
            SignalBroadcaster,
        )

        bc = SignalBroadcaster()
        q = await bc.subscribe()
        await bc.unsubscribe(q)
        assert bc.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_recent_signals(self):
        """Recent signals are stored."""
        from python_ai.ws_signal_stream import (
            SignalBroadcaster,
        )

        bc = SignalBroadcaster()
        await bc.broadcast({"type": "test"})
        assert len(bc.recent_signals) == 1


# ────────────────────────────────────────────────────────
# 11. Config Hot Reload
# ────────────────────────────────────────────────────────


class TestConfigHotReload:
    """Tests for config_hot_reload.py."""

    def test_initial_load(self, tmp_path):
        """Config loads on init."""
        from python_ai.config_hot_reload import (
            ConfigHotReloader,
        )

        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"key": "val"}))
        hr = ConfigHotReloader(str(cfg_file), poll_interval=0.1)
        hr.reload_now()
        assert hr.config["key"] == "val"

    def test_detects_change(self, tmp_path):
        """Detects config file change."""
        from python_ai.config_hot_reload import (
            ConfigHotReloader,
        )

        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"v": 1}))
        hr = ConfigHotReloader(str(cfg_file), poll_interval=0.1)
        hr.reload_now()
        assert hr.config["v"] == 1

        cfg_file.write_text(json.dumps({"v": 2}))
        changed = hr.reload_now()
        assert changed
        assert hr.config["v"] == 2

    def test_callback_invoked(self, tmp_path):
        """Callbacks fire on config change."""
        from python_ai.config_hot_reload import (
            ConfigHotReloader,
        )

        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"a": 1}))
        hr = ConfigHotReloader(str(cfg_file), poll_interval=0.1)
        hr.reload_now()  # initial load
        received = []
        hr.on_change(lambda c: received.append(c))
        cfg_file.write_text(json.dumps({"a": 2}))
        hr.reload_now()
        assert len(received) == 1
        assert received[0]["a"] == 2

    def test_missing_file(self, tmp_path):
        """Missing config file handled gracefully."""
        from python_ai.config_hot_reload import (
            ConfigHotReloader,
        )

        hr = ConfigHotReloader(str(tmp_path / "missing.json"))
        assert not hr.reload_now()


# ────────────────────────────────────────────────────────
# 12. A/B Testing
# ────────────────────────────────────────────────────────


class TestABTesting:
    """Tests for ab_testing.py."""

    def test_assign_returns_variant_name(self):
        """Assign returns a valid variant name."""
        from python_ai.ab_testing import ABTest

        ab = ABTest(split=0.5)
        name = ab.assign()
        assert name in ("control", "treatment")

    def test_record_and_summary(self):
        """Records are reflected in summary."""
        from python_ai.ab_testing import ABTest

        ab = ABTest()
        for _ in range(10):
            ab.record("control", 1.0)
            ab.record("treatment", 1.5)
        s = ab.summary()
        assert s["variant_a"]["n"] == 10
        assert s["variant_b"]["n"] == 10

    def test_welch_t_test_insufficient_data(self):
        """T-test returns None with < 2 samples."""
        from python_ai.ab_testing import ABTest

        ab = ABTest()
        ab.record("control", 1.0)
        t = ab.welch_t_test()
        assert t["t_stat"] is None

    def test_welch_t_test_detects_difference(self):
        """T-test detects a significant difference."""
        from python_ai.ab_testing import ABTest

        ab = ABTest()
        for i in range(100):
            ab.record("control", 1.0 + 0.01 * i)
            ab.record("treatment", 5.0 + 0.01 * i)
        t = ab.welch_t_test()
        assert t["significant"] is True
        assert t["winner"] == "treatment"

    def test_variant_stats(self):
        """StrategyVariant computes mean/std."""
        from python_ai.ab_testing import StrategyVariant

        sv = StrategyVariant("test")
        sv.record(1.0)
        sv.record(3.0)
        assert sv.mean == 2.0
        assert sv.std > 0


# ────────────────────────────────────────────────────────
# 13. Multi-Exchange Aggregator
# ────────────────────────────────────────────────────────


class TestMultiExchangeAggregator:
    """Tests for multi_exchange_aggregator.py."""

    def _make_quote(self, exchange, bid, ask, vol):
        """Create a test quote."""
        from python_ai.multi_exchange_aggregator import (
            ExchangeQuote,
        )

        return ExchangeQuote(
            exchange=exchange,
            symbol="BTC/USDT",
            bid=bid,
            ask=ask,
            last=(bid + ask) / 2,
            volume_24h=vol,
        )

    def test_vwap(self):
        """VWAP weights by volume."""
        from python_ai.multi_exchange_aggregator import (
            MultiExchangeAggregator,
        )

        agg = MultiExchangeAggregator(stale_seconds=60)
        agg.update(self._make_quote("binance", 100, 101, 1000))
        agg.update(self._make_quote("kraken", 99, 100, 500))
        v = agg.vwap("BTC/USDT")
        assert 99.0 < v < 101.0

    def test_median_price(self):
        """Median price is the middle value."""
        from python_ai.multi_exchange_aggregator import (
            MultiExchangeAggregator,
        )

        agg = MultiExchangeAggregator(stale_seconds=60)
        agg.update(self._make_quote("a", 100, 102, 100))
        agg.update(self._make_quote("b", 200, 202, 100))
        agg.update(self._make_quote("c", 150, 152, 100))
        med = agg.median_price("BTC/USDT")
        assert 150.0 <= med <= 152.0

    def test_best_bid_ask(self):
        """Best bid is highest, best ask is lowest."""
        from python_ai.multi_exchange_aggregator import (
            MultiExchangeAggregator,
        )

        agg = MultiExchangeAggregator(stale_seconds=60)
        agg.update(self._make_quote("a", 100, 105, 100))
        agg.update(self._make_quote("b", 101, 104, 100))
        assert agg.best_bid("BTC/USDT") == 101
        assert agg.best_ask("BTC/USDT") == 104

    def test_spread_summary(self):
        """Spread summary contains exchange count."""
        from python_ai.multi_exchange_aggregator import (
            MultiExchangeAggregator,
        )

        agg = MultiExchangeAggregator(stale_seconds=60)
        agg.update(self._make_quote("a", 100, 101, 100))
        ss = agg.spread_summary("BTC/USDT")
        assert ss["exchanges"] == 1

    def test_quote_mid_and_spread(self):
        """ExchangeQuote computes mid and spread."""
        from python_ai.multi_exchange_aggregator import (
            ExchangeQuote,
        )

        q = ExchangeQuote("test", "BTC/USDT", bid=100, ask=102, last=101)
        assert q.mid == 101.0
        assert q.spread > 0


# ────────────────────────────────────────────────────────
# 14. API Versioning
# ────────────────────────────────────────────────────────


class TestAPIVersioning:
    """Tests for api_versioning.py."""

    def test_extract_version_from_path(self):
        """Version is parsed from URL prefix."""
        from python_ai.api_versioning import (
            APIVersionMiddleware,
        )

        mw = APIVersionMiddleware(app=None, min_version=1, max_version=3)
        ver, path = mw._extract_version("/v2/predict")
        assert ver == 2
        assert path == "/predict"

    def test_default_version(self):
        """No prefix defaults to version 1."""
        from python_ai.api_versioning import (
            APIVersionMiddleware,
        )

        mw = APIVersionMiddleware(app=None)
        ver, path = mw._extract_version("/predict")
        assert ver == 1
        assert path == "/predict"

    def test_get_api_version(self):
        """Helper reads version from scope."""
        from python_ai.api_versioning import (
            get_api_version,
        )

        assert get_api_version({"api_version": 2}) == 2
        assert get_api_version({}) == 1


# ────────────────────────────────────────────────────────
# 15. HMAC Signing
# ────────────────────────────────────────────────────────


class TestHMACSigning:
    """Tests for hmac_signing.py."""

    def test_sign_and_verify(self):
        """Sign then verify round-trip."""
        from python_ai.hmac_signing import (
            sign_request,
            verify_signature,
        )

        secret = "my-secret-key"
        headers = sign_request(secret, "POST", "/v1/order", '{"qty": 1}')
        ok = verify_signature(
            secret,
            "POST",
            "/v1/order",
            '{"qty": 1}',
            headers["X-Signature"],
            headers["X-Timestamp"],
        )
        assert ok

    def test_tampered_body_fails(self):
        """Modified body fails verification."""
        from python_ai.hmac_signing import (
            sign_request,
            verify_signature,
        )

        secret = "secret"
        headers = sign_request(secret, "POST", "/order", "original")
        ok = verify_signature(
            secret,
            "POST",
            "/order",
            "tampered",
            headers["X-Signature"],
            headers["X-Timestamp"],
        )
        assert not ok

    def test_expired_timestamp(self):
        """Old timestamp fails verification."""
        from python_ai.hmac_signing import (
            sign_request,
            verify_signature,
        )

        secret = "secret"
        old_ts = time.time() - 600
        headers = sign_request(secret, "GET", "/data", "", old_ts)
        ok = verify_signature(
            secret,
            "GET",
            "/data",
            "",
            headers["X-Signature"],
            headers["X-Timestamp"],
            max_age=300,
        )
        assert not ok

    def test_hmac_verifier_stats(self):
        """HMACVerifier tracks stats."""
        from python_ai.hmac_signing import (
            HMACVerifier,
            sign_request,
        )

        v = HMACVerifier("key")
        h = sign_request("key", "GET", "/x")
        v.verify("GET", "/x", "", h["X-Signature"], h["X-Timestamp"])
        s = v.stats()
        assert s["verified"] == 1


# ────────────────────────────────────────────────────────
# 16. Vectorized Indicators
# ────────────────────────────────────────────────────────


class TestVectorizedIndicators:
    """Tests for vectorized_indicators.py."""

    def _prices(self, n=100):
        """Generate synthetic prices."""
        rng = np.random.default_rng(42)
        return (100 + np.cumsum(rng.normal(0, 1, n))).tolist()

    def test_sma(self):
        """SMA has NaN for warm-up, values after."""
        from python_ai.vectorized_indicators import sma

        result = sma(self._prices(), window=14)
        assert np.isnan(result[0])
        assert not np.isnan(result[14])

    def test_ema(self):
        """EMA has no NaN values."""
        from python_ai.vectorized_indicators import ema

        result = ema(self._prices(), span=12)
        assert not np.any(np.isnan(result))

    def test_rsi_range(self):
        """RSI values are between 0 and 100."""
        from python_ai.vectorized_indicators import rsi

        result = rsi(self._prices(), window=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_macd_keys(self):
        """MACD returns macd, signal, histogram."""
        from python_ai.vectorized_indicators import macd

        m = macd(self._prices())
        assert "macd" in m
        assert "signal" in m
        assert "histogram" in m

    def test_bollinger_bands(self):
        """Upper > middle > lower."""
        from python_ai.vectorized_indicators import (
            bollinger_bands,
        )

        bb = bollinger_bands(self._prices(), window=20)
        idx = 25
        assert bb["upper"][idx] > bb["middle"][idx]
        assert bb["middle"][idx] > bb["lower"][idx]

    def test_compute_all(self):
        """compute_all returns all indicators."""
        from python_ai.vectorized_indicators import (
            compute_all,
        )

        result = compute_all(self._prices())
        assert "sma_14" in result
        assert "rsi_14" in result
        assert "macd" in result


# ────────────────────────────────────────────────────────
# 17. Circular Buffer
# ────────────────────────────────────────────────────────


class TestCircularBuffer:
    """Tests for circular_buffer.py."""

    def test_append_and_read(self):
        """Values accessible after append."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=5)
        buf.append(1.0)
        buf.append(2.0)
        assert len(buf) == 2
        assert buf.last == 2.0

    def test_overflow_wraps(self):
        """Overflow overwrites oldest."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=3)
        for i in range(5):
            buf.append(float(i))
        assert len(buf) == 3
        arr = buf.to_list()
        assert arr == [2.0, 3.0, 4.0]

    def test_extend(self):
        """Extend appends multiple values."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=10)
        buf.extend([1.0, 2.0, 3.0])
        assert len(buf) == 3

    def test_latest(self):
        """Latest returns n most recent."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=10)
        buf.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        latest = buf.latest(3).tolist()
        assert latest == [3.0, 4.0, 5.0]

    def test_stats(self):
        """Mean and std are correct."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=10)
        buf.extend([2.0, 4.0, 6.0])
        assert buf.mean() == 4.0
        assert buf.min_val() == 2.0
        assert buf.max_val() == 6.0

    def test_is_full(self):
        """is_full reports correctly."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=2)
        assert not buf.is_full
        buf.extend([1.0, 2.0])
        assert buf.is_full

    def test_invalid_capacity(self):
        """Capacity < 1 raises ValueError."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        with pytest.raises(ValueError):
            CircularBuffer(capacity=0)

    def test_summary(self):
        """Summary includes all stats."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=5)
        buf.extend([1.0, 2.0])
        s = buf.summary()
        assert s["count"] == 2
        assert s["capacity"] == 5

    def test_empty_buffer(self):
        """Empty buffer returns safe defaults."""
        from python_ai.circular_buffer import (
            CircularBuffer,
        )

        buf = CircularBuffer(capacity=5)
        assert buf.last is None
        assert buf.first is None
        assert buf.mean() == 0.0


# ────────────────────────────────────────────────────────
# 18. Alembic Migration Schema
# ────────────────────────────────────────────────────────


class TestAlembicMigration:
    """Verify the initial migration schema is valid."""

    def test_migration_file_exists(self):
        """Migration file exists and is valid Python."""
        import importlib.util
        import pathlib

        migration = (
            pathlib.Path(__file__).resolve().parents[1]
            / "alembic"
            / "versions"
            / "001_initial_schema.py"
        )
        assert migration.exists()
        spec = importlib.util.spec_from_file_location(
            "initial_schema", str(migration)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "upgrade")
        assert hasattr(mod, "downgrade")
