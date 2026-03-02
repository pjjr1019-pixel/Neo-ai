"""
Tests for all new modules added in the production-hardening batch.

Covers:
- circuit_breaker.py
- rate_limiter.py
- position_tracker.py
- paper_trading.py
- risk_management.py  (Kelly Criterion, ATR stop-loss)
- drift_monitor.py
- onnx_export.py
- incremental_indicators.py
- data_quality.py
- signal_alerts.py
- strategy_config.py
- perf_timing.py
- trade_journal.py
- FastAPI /health and /metrics/prometheus endpoints
"""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ────────────────────────────────────────────────────────────
# 1. Circuit Breaker
# ────────────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for circuit_breaker.py."""

    def test_starts_closed(self):
        """Test starts closed."""
        from python_ai.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_on_success(self):
        """Test stays closed on success."""
        from python_ai.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        """Test opens after threshold failures."""
        from python_ai.circuit_breaker import (
            CircuitBreaker,
            CircuitOpenError,
            CircuitState,
        )

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        def fail():
            """Raise an exception for testing."""
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(fail)

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            cb.call(lambda: 1)

    def test_half_open_after_timeout(self):
        """Test half open after timeout."""
        from python_ai.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        def fail():
            """Raise an exception for testing."""
            raise ValueError("x")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(fail)

        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_resets_after_half_open_success(self):
        """Test resets after half open success."""
        from python_ai.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)

        def fail():
            """Raise an exception for testing."""
            raise RuntimeError("x")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        time.sleep(0.1)
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_decorator_usage(self):
        """Test decorator usage."""
        from python_ai.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)

        @cb
        def add(a, b):
            """Simple addition function for testing."""
            return a + b

        assert add(1, 2) == 3

    def test_to_dict(self):
        """Test to dict."""
        from python_ai.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test_cb")
        d = cb.to_dict()
        assert d["name"] == "test_cb"
        assert d["state"] == "closed"

    def test_manual_reset(self):
        """Test manual reset."""
        from python_ai.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))

        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


# ────────────────────────────────────────────────────────────
# 2. Rate Limiter
# ────────────────────────────────────────────────────────────


class TestRateLimiter:
    """Tests for rate_limiter.py."""

    def test_token_bucket_allows_within_capacity(self):
        """Test token bucket allows within capacity."""
        from python_ai.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        for _ in range(10):
            assert bucket.consume() is True

    def test_token_bucket_denies_over_capacity(self):
        """Test token bucket denies over capacity."""
        from python_ai.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=2, refill_rate=0.0)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_rate_limiter_per_key(self):
        """Test rate limiter per key."""
        from python_ai.rate_limiter import RateLimiter

        rl = RateLimiter(capacity=1, refill_rate=0.0)
        assert rl.allow("ip1") is True
        assert rl.allow("ip1") is False
        assert rl.allow("ip2") is True  # different key

    def test_middleware_returns_429(self):
        """Test middleware returns 429."""
        from fastapi import FastAPI

        from python_ai.rate_limiter import RateLimitMiddleware

        test_app = FastAPI()
        test_app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=2,
        )

        @test_app.get("/test")
        def dummy():
            """Dummy function for testing."""
            return {"ok": True}

        client = TestClient(test_app)
        client.get("/test")
        client.get("/test")
        resp = client.get("/test")
        assert resp.status_code == 429


# ────────────────────────────────────────────────────────────
# 3. Position Tracker
# ────────────────────────────────────────────────────────────


class TestPositionTracker:
    """Tests for position_tracker.py."""

    def test_open_and_close_long(self):
        """Test open and close long."""
        from python_ai.position_tracker import Position

        pos = Position("BTC/USDT", "long", 100.0, 1.0)
        assert pos.is_open is True
        assert pos.unrealised_pnl(110.0) == 10.0
        pnl = pos.close(110.0)
        assert pnl == 10.0
        assert pos.is_open is False

    def test_short_position(self):
        """Test short position."""
        from python_ai.position_tracker import Position

        pos = Position("ETH/USDT", "short", 200.0, 2.0)
        assert pos.unrealised_pnl(190.0) == 20.0  # profit on short
        assert pos.unrealised_pnl(210.0) == -20.0  # loss on short

    def test_stop_loss_trigger(self):
        """Test stop loss trigger."""
        from python_ai.position_tracker import Position

        pos = Position("X", "long", 100.0, 1.0, stop_loss=95.0)
        assert pos.should_stop_out(96.0) is False
        assert pos.should_stop_out(95.0) is True
        assert pos.should_stop_out(90.0) is True

    def test_take_profit_trigger(self):
        """Test take profit trigger."""
        from python_ai.position_tracker import Position

        pos = Position("X", "long", 100.0, 1.0, take_profit=110.0)
        assert pos.should_take_profit(109.0) is False
        assert pos.should_take_profit(110.0) is True

    def test_portfolio_equity(self):
        """Test portfolio equity."""
        from python_ai.position_tracker import PortfolioTracker

        pt = PortfolioTracker(10000.0)
        pt.open_position("BTC", "long", 100.0, 10.0)
        assert pt.balance == 9000.0
        eq = pt.equity({"BTC": 110.0})
        # balance + notional + unrealised = 9000 + 1000 + 100
        assert eq == 10100.0

    def test_portfolio_win_rate(self):
        """Test portfolio win rate."""
        from python_ai.position_tracker import PortfolioTracker

        pt = PortfolioTracker(10000.0)
        p1 = pt.open_position("A", "long", 100.0, 1.0)
        pt.close_position(p1, 110.0)  # win
        p2 = pt.open_position("B", "long", 100.0, 1.0)
        pt.close_position(p2, 90.0)  # loss
        assert pt.win_rate() == 50.0

    def test_insufficient_balance(self):
        """Test insufficient balance."""
        from python_ai.position_tracker import PortfolioTracker

        pt = PortfolioTracker(100.0)
        with pytest.raises(ValueError, match="Insufficient"):
            pt.open_position("X", "long", 200.0, 1.0)


# ────────────────────────────────────────────────────────────
# 4. Paper Trading Engine
# ────────────────────────────────────────────────────────────


class TestPaperTrading:
    """Tests for paper_trading.py."""

    def test_buy_and_sell(self):
        """Test buy and sell."""
        from python_ai.paper_trading import PaperTradingEngine

        eng = PaperTradingEngine(initial_capital=10000.0)
        pos = eng.execute_buy("BTC", 100.0, quantity=1.0)
        assert pos is not None
        pnl = eng.execute_sell("BTC", 110.0)
        assert pnl is not None
        assert pnl > 0

    def test_auto_sizing(self):
        """Test auto sizing."""
        from python_ai.paper_trading import PaperTradingEngine

        eng = PaperTradingEngine(initial_capital=10000.0)
        pos = eng.execute_buy("ETH", 100.0)
        assert pos is not None
        assert pos.quantity > 0

    def test_signal_buy_hold_sell(self):
        """Test signal buy hold sell."""
        from python_ai.paper_trading import PaperTradingEngine

        eng = PaperTradingEngine(initial_capital=10000.0)
        r1 = eng.act_on_signal("X", "BUY", 100.0)
        assert r1 is not None and r1["action"] == "BUY"
        r2 = eng.act_on_signal("X", "HOLD", 105.0)
        assert r2 is None
        r3 = eng.act_on_signal("X", "SELL", 110.0)
        assert r3 is not None and r3["action"] == "SELL"

    def test_check_exits(self):
        """Test check exits."""
        from python_ai.paper_trading import PaperTradingEngine

        eng = PaperTradingEngine(initial_capital=10000.0)
        eng.execute_buy("X", 100.0, quantity=1.0, stop_loss=90.0)
        exits = eng.check_exits({"X": 85.0})
        assert len(exits) == 1
        assert exits[0]["reason"] == "stop_loss"

    def test_order_log(self):
        """Test order log."""
        from python_ai.paper_trading import PaperTradingEngine

        eng = PaperTradingEngine(initial_capital=10000.0)
        eng.execute_buy("A", 50.0, quantity=1.0)
        assert len(eng.order_log) == 1

    def test_summary(self):
        """Test summary."""
        from python_ai.paper_trading import PaperTradingEngine

        eng = PaperTradingEngine(initial_capital=5000.0)
        s = eng.summary()
        assert s["initial_capital"] == 5000.0
        assert "total_orders" in s


# ────────────────────────────────────────────────────────────
# 5. Risk Management (Kelly + ATR)
# ────────────────────────────────────────────────────────────


class TestRiskManagement:
    """Tests for risk_management.py."""

    def test_kelly_fraction_50_50(self):
        """Test kelly fraction 50 50."""
        from python_ai.risk_management import kelly_fraction

        f = kelly_fraction(0.5, 2.0, 1.0)
        assert 0 < f <= 0.25

    def test_kelly_fraction_zero_win_rate(self):
        """Test kelly fraction zero win rate."""
        from python_ai.risk_management import kelly_fraction

        f = kelly_fraction(0.0, 1.0, 1.0)
        assert f == 0.0

    def test_kelly_position_size(self):
        """Test kelly position size."""
        from python_ai.risk_management import kelly_position_size

        qty = kelly_position_size(10000, 0.6, 2.0, 1.0, 100.0)
        assert qty > 0

    def test_atr_stop_loss_long(self):
        """Test atr stop loss long."""
        from python_ai.risk_management import atr_stop_loss

        sl = atr_stop_loss(100.0, 5.0, multiplier=2.0, side="long")
        assert sl == 90.0

    def test_atr_stop_loss_short(self):
        """Test atr stop loss short."""
        from python_ai.risk_management import atr_stop_loss

        sl = atr_stop_loss(100.0, 5.0, multiplier=2.0, side="short")
        assert sl == 110.0

    def test_atr_take_profit(self):
        """Test atr take profit."""
        from python_ai.risk_management import atr_take_profit

        tp = atr_take_profit(100.0, 5.0, multiplier=3.0, side="long")
        assert tp == 115.0

    def test_trailing_stop(self):
        """Test trailing stop."""
        from python_ai.risk_management import atr_trailing_stop

        ts = atr_trailing_stop(120.0, 5.0, multiplier=2.0, side="long")
        assert ts == 110.0

    def test_max_drawdown(self):
        """Test max drawdown."""
        from python_ai.risk_management import max_drawdown_from_equity

        dd = max_drawdown_from_equity([100, 110, 90, 100])
        assert dd > 0

    def test_should_halt_trading(self):
        """Test should halt trading."""
        from python_ai.risk_management import should_halt_trading

        assert should_halt_trading([100, 50], max_allowed_drawdown=0.20)
        assert not should_halt_trading([100, 95], max_allowed_drawdown=0.20)


# ────────────────────────────────────────────────────────────
# 6. Drift Monitor
# ────────────────────────────────────────────────────────────


class TestDriftMonitor:
    """Tests for drift_monitor.py."""

    def test_no_drift_without_baseline(self):
        """Test no drift without baseline."""
        from python_ai.drift_monitor import DriftDetector

        dd = DriftDetector()
        for i in range(20):
            dd.record_prediction(float(i))
        assert dd.is_drifted() is False

    def test_detects_drift(self):
        """Test detects drift."""
        from python_ai.drift_monitor import DriftDetector

        dd = DriftDetector(window_size=50, drift_threshold=1.5)
        dd.set_baseline([0.0] * 50)

        for _ in range(50):
            dd.record_prediction(10.0)

        assert dd.is_drifted() is True

    def test_accuracy_score(self):
        """Test accuracy score."""
        from python_ai.drift_monitor import DriftDetector

        dd = DriftDetector(window_size=50)
        for i in range(20):
            dd.record_prediction(float(i), actual=float(i))
        r2 = dd.accuracy_score()
        assert r2 is not None
        assert r2 > 0.99

    def test_auto_retrain_fires(self):
        """Test auto retrain fires."""
        from python_ai.drift_monitor import AutoRetrainTrigger, DriftDetector

        dd = DriftDetector(window_size=20, drift_threshold=1.0)
        dd.set_baseline([0.0] * 20)

        mock_retrain = MagicMock(return_value={"r2": 0.9})
        trigger = AutoRetrainTrigger(
            dd,
            retrain_callback=mock_retrain,
            cooldown_seconds=0,
        )

        for _ in range(20):
            dd.record_prediction(100.0)

        result = trigger.check()
        assert result is not None
        mock_retrain.assert_called_once()

    def test_cooldown_prevents_retrain(self):
        """Test cooldown prevents retrain."""
        from python_ai.drift_monitor import AutoRetrainTrigger, DriftDetector

        dd = DriftDetector(window_size=20, drift_threshold=1.0)
        dd.set_baseline([0.0] * 20)

        mock_retrain = MagicMock(return_value={"r2": 0.9})
        trigger = AutoRetrainTrigger(
            dd,
            retrain_callback=mock_retrain,
            cooldown_seconds=3600,
        )

        for _ in range(20):
            dd.record_prediction(100.0)

        trigger.check()
        result2 = trigger.check()
        assert result2 is None  # blocked by cooldown

    def test_to_dict(self):
        """Test to dict."""
        from python_ai.drift_monitor import DriftDetector

        dd = DriftDetector()
        d = dd.to_dict()
        assert "drift_score" in d
        assert "is_drifted" in d


# ────────────────────────────────────────────────────────────
# 7. Incremental Indicators
# ────────────────────────────────────────────────────────────


class TestIncrementalIndicators:
    """Tests for incremental_indicators.py."""

    def test_update_returns_features(self):
        """Test update returns features."""
        from python_ai.incremental_indicators import IncrementalIndicators

        inc = IncrementalIndicators()
        prices = [100 + i * 0.5 for i in range(30)]
        features = inc.update(prices)
        assert "f0" in features
        assert "f9" in features

    def test_append_bar(self):
        """Test append bar."""
        from python_ai.incremental_indicators import IncrementalIndicators

        inc = IncrementalIndicators()
        prices = [100.0 + i for i in range(20)]
        inc.update(prices)
        feat = inc.append_bar(121.0)
        assert feat["f0"] > 0  # RSI should be non-zero

    def test_latest_features_defaults(self):
        """Test lafeatures defaults."""
        from python_ai.incremental_indicators import IncrementalIndicators

        inc = IncrementalIndicators()
        feat = inc.latest_features()
        assert all(v == 0.0 for v in feat.values())

    def test_cache_reuse(self):
        """Test cache reuse."""
        from python_ai.incremental_indicators import IncrementalIndicators

        inc = IncrementalIndicators()
        prices = [100 + i for i in range(25)]
        f1 = inc.update(prices)
        f2 = inc.update(prices)  # same data, should use cache
        assert f1 == f2


# ────────────────────────────────────────────────────────────
# 8. Data Quality Monitor
# ────────────────────────────────────────────────────────────


class TestDataQuality:
    """Tests for data_quality.py."""

    def _make_ohlcv(self, n=50):
        """make ohlcv."""
        base = [100 + i * 0.1 for i in range(n)]
        return {
            "open": base,
            "high": [p + 1 for p in base],
            "low": [p - 1 for p in base],
            "close": base,
            "volume": [1000.0] * n,
        }

    def test_valid_data_passes(self):
        """Test valid data passes."""
        from python_ai.data_quality import DataQualityMonitor

        mon = DataQualityMonitor()
        report = mon.validate(self._make_ohlcv())
        assert report.is_valid is True

    def test_missing_columns(self):
        """Test missing columns."""
        from python_ai.data_quality import DataQualityMonitor

        mon = DataQualityMonitor()
        report = mon.validate({"close": [1, 2, 3]})
        assert report.is_valid is False
        assert "Missing columns" in report.errors[0]

    def test_nan_detected(self):
        """Test nan detected."""
        from python_ai.data_quality import DataQualityMonitor

        data = self._make_ohlcv()
        data["close"][5] = float("nan")
        mon = DataQualityMonitor()
        report = mon.validate(data)
        assert report.is_valid is False

    def test_negative_volume(self):
        """Test negative volume."""
        from python_ai.data_quality import DataQualityMonitor

        data = self._make_ohlcv()
        data["volume"][0] = -100.0
        mon = DataQualityMonitor()
        report = mon.validate(data)
        assert report.is_valid is False

    def test_insufficient_bars(self):
        """Test insufficient bars."""
        from python_ai.data_quality import DataQualityMonitor

        data = {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
        mon = DataQualityMonitor()
        report = mon.validate(data)
        assert report.is_valid is False

    def test_to_dict(self):
        """Test to dict."""
        from python_ai.data_quality import DataQualityMonitor

        mon = DataQualityMonitor()
        report = mon.validate(self._make_ohlcv())
        d = report.to_dict()
        assert "is_valid" in d
        assert "checks" in d


# ────────────────────────────────────────────────────────────
# 9. Signal Alerts
# ────────────────────────────────────────────────────────────


class TestSignalAlerts:
    """Tests for signal_alerts.py."""

    def test_fire_alert(self):
        """Test fire alert."""
        from python_ai.signal_alerts import AlertDispatcher, AlertLevel

        disp = AlertDispatcher()
        alert = disp.fire(AlertLevel.INFO, "signal", "BUY BTC")
        assert alert.level == AlertLevel.INFO
        assert alert.category == "signal"

    def test_history(self):
        """Test history."""
        from python_ai.signal_alerts import AlertDispatcher, AlertLevel

        disp = AlertDispatcher()
        disp.fire(AlertLevel.INFO, "a", "msg1")
        disp.fire(AlertLevel.WARNING, "b", "msg2")
        assert len(disp.history) == 2

    def test_recent(self):
        """Test recent."""
        from python_ai.signal_alerts import AlertDispatcher, AlertLevel

        disp = AlertDispatcher()
        for i in range(5):
            disp.fire(AlertLevel.INFO, "test", f"msg{i}")
        recent = disp.recent(3)
        assert len(recent) == 3

    def test_handler_called(self):
        """Test handler called."""
        from python_ai.signal_alerts import AlertDispatcher, AlertLevel

        handler = MagicMock()
        disp = AlertDispatcher()
        disp.add_handler(handler)
        disp.fire(AlertLevel.INFO, "sig", "test")
        handler.assert_called_once()

    def test_file_handler_factory(self):
        """Test file handler factory."""
        from python_ai.signal_alerts import (
            AlertDispatcher,
            AlertLevel,
            file_handler_factory,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name

        try:
            disp = AlertDispatcher()
            disp.add_handler(file_handler_factory(path))
            disp.fire(AlertLevel.INFO, "test", "hello")
            with open(path) as fh:
                lines = fh.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["message"] == "hello"
        finally:
            os.unlink(path)

    def test_alert_to_json(self):
        """Test alert to json."""
        from python_ai.signal_alerts import Alert, AlertLevel

        a = Alert(AlertLevel.INFO, "cat", "msg", {"k": 1})
        j = a.to_json()
        parsed = json.loads(j)
        assert parsed["category"] == "cat"


# ────────────────────────────────────────────────────────────
# 10. Strategy Config
# ────────────────────────────────────────────────────────────


class TestStrategyConfig:
    """Tests for strategy_config.py."""

    def test_default_strategy(self):
        """Test default strategy."""
        from python_ai.strategy_config import load_strategy

        cfg = load_strategy()
        assert cfg.name == "default"
        assert len(cfg.symbols) > 0

    def test_properties(self):
        """Test properties."""
        from python_ai.strategy_config import load_strategy

        cfg = load_strategy()
        assert cfg.max_position_pct == 0.25
        assert cfg.stop_loss_atr_mult == 2.0
        assert cfg.is_paper is True

    def test_load_json(self):
        """Test load json."""
        from python_ai.strategy_config import load_strategy

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"name": "test_strat", "symbols": ["SOL/USDT"]}, f)
            path = f.name

        try:
            cfg = load_strategy(path)
            assert cfg.name == "test_strat"
            assert "SOL/USDT" in cfg.symbols
        finally:
            os.unlink(path)

    def test_save_and_load(self):
        """Test save and load."""
        from python_ai.strategy_config import (
            StrategyConfig,
            load_strategy,
            save_strategy,
        )

        cfg = StrategyConfig({"name": "roundtrip", "version": "2.0"})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_strategy(cfg, path)
            loaded = load_strategy(path)
            assert loaded.name == "roundtrip"
        finally:
            os.unlink(path)

    def test_to_dict(self):
        """Test to dict."""
        from python_ai.strategy_config import load_strategy

        cfg = load_strategy()
        d = cfg.to_dict()
        assert "name" in d


# ────────────────────────────────────────────────────────────
# 11. Perf Timing
# ────────────────────────────────────────────────────────────


class TestPerfTiming:
    """Tests for perf_timing.py."""

    def test_timed_decorator(self):
        """Test timed decorator."""
        from python_ai.perf_timing import get_timing_registry, timed

        reg = get_timing_registry()
        reg.reset()

        @timed
        def add(a, b):
            """Simple addition function for testing."""
            return a + b

        result = add(3, 4)
        assert result == 7

        stats = reg.all_stats()
        assert any("add" in key for key in stats)

    def test_timing_stats(self):
        """Test timing stats."""
        from python_ai.perf_timing import TimingRegistry

        reg = TimingRegistry()
        for _ in range(10):
            reg.record("fn", 0.1)
        s = reg.stats("fn")
        assert s["count"] == 10
        assert abs(s["mean"] - 0.1) < 0.01

    def test_empty_stats(self):
        """Test empty stats."""
        from python_ai.perf_timing import TimingRegistry

        reg = TimingRegistry()
        s = reg.stats("nonexistent")
        assert s["count"] == 0


# ────────────────────────────────────────────────────────────
# 12. Trade Journal
# ────────────────────────────────────────────────────────────


class TestTradeJournal:
    """Tests for trade_journal.py."""

    def test_log_and_query(self):
        """Test log and query."""
        from python_ai.trade_journal import TradeJournal

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            j = TradeJournal(path)
            j.log_trade("BTC", "BUY", 100.0, 1.0)
            j.log_trade("BTC", "SELL", 110.0, 1.0, pnl=10.0)
            assert len(j.trades) == 2
            assert j.total_pnl() == 10.0
        finally:
            os.unlink(path)

    def test_win_rate(self):
        """Test win rate."""
        from python_ai.trade_journal import TradeJournal

        j = TradeJournal(auto_flush=False, file_path="test_dummy.jsonl")
        j.log_trade("X", "SELL", 100, 1, pnl=10)
        j.log_trade("X", "SELL", 100, 1, pnl=-5)
        assert j.win_rate() == 50.0

    def test_summary(self):
        """Test summary."""
        from python_ai.trade_journal import TradeJournal

        j = TradeJournal(auto_flush=False, file_path="test_dummy.jsonl")
        j.log_trade("BTC", "BUY", 100, 1)
        s = j.summary()
        assert s["total_trades"] == 1

    def test_persistence_round_trip(self):
        """Test persistence round trip."""
        from python_ai.trade_journal import TradeJournal

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            j1 = TradeJournal(path)
            j1.log_trade("ETH", "BUY", 200, 0.5)
            j1.log_trade("ETH", "SELL", 220, 0.5, pnl=10)

            j2 = TradeJournal(path)
            loaded = j2.load()
            assert loaded == 2
            assert j2.total_pnl() == 10.0
        finally:
            os.unlink(path)

    def test_trades_for_symbol(self):
        """Test trades for symbol."""
        from python_ai.trade_journal import TradeJournal

        j = TradeJournal(auto_flush=False, file_path="test_dummy.jsonl")
        j.log_trade("BTC", "BUY", 100, 1)
        j.log_trade("ETH", "BUY", 200, 1)
        assert len(j.trades_for_symbol("BTC")) == 1


# ────────────────────────────────────────────────────────────
# 13. FastAPI endpoints (/health, /metrics/prometheus)
# ────────────────────────────────────────────────────────────


class TestFastAPINewEndpoints:
    """Tests for new /health and /metrics/prometheus endpoints."""

    def test_health_endpoint(self):
        """Test health endpoint."""
        from python_ai.fastapi_service.fastapi_service import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "uptime_seconds" in data
        assert "checks" in data

    def test_prometheus_endpoint(self):
        """Test prometheus endpoint."""
        from python_ai.fastapi_service.fastapi_service import app

        client = TestClient(app)
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
        body = resp.text
        assert "neo_requests_total" in body
        assert "neo_uptime_seconds" in body
        assert "neo_model_trained" in body


# ────────────────────────────────────────────────────────────
# 14. ONNX Export
# ────────────────────────────────────────────────────────────


class TestOnnxExport:
    """Tests for onnx_export.py."""

    def test_manual_export_fallback(self):
        """Test manual export fallback."""
        from python_ai.ml_model import MLModel
        from python_ai.onnx_export import export_model_to_onnx

        model = MLModel.__new__(MLModel)
        model.rf_model = None
        model.gb_model = None
        model.scaler = None
        model.is_trained = False
        model.training_metrics = {}
        model.train_count = 0
        model.model_path = "test.pkl"
        model._initialize_models()
        model.train_on_synthetic_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test.onnx")
            result = export_model_to_onnx(model, out)
            assert result["status"] in ("ok", "ok_fallback")
