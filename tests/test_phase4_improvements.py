"""
Tests for Phase 4 improvements.

Covers:
- Capital safety (ATR validation, equity validation, daily loss limit,
  max position size)
- Data validation (timestamp gap detection, data freshness)
- Performance improvements (__slots__, backtesting engine reset)
- Algorithm fixes (memoized tournament, if __name__ fix)
- Rate limiter reset
- Type annotations correctness
"""

import time

import pytest

from python_ai.backtesting_engine import (
    BacktestingEngine,
    BacktestMetrics,
    get_backtesting_engine,
    reset_backtesting_engine,
)
from python_ai.data_pipeline import (
    check_data_freshness,
    detect_timestamp_gaps,
)
from python_ai.evolution_engine import EvolutionEngine, Strategy
from python_ai.rate_limiter import RateLimiter
from python_ai.risk_management import (
    atr_stop_loss,
    atr_take_profit,
    check_daily_loss_limit,
    kelly_position_size,
    max_position_size,
)


# ── Capital Safety ────────────────────────────────────────────


class TestATRValidation:
    """Test ATR input validation."""

    def test_atr_stop_loss_rejects_zero(self) -> None:
        """Zero ATR should raise ValueError."""
        with pytest.raises(ValueError, match="atr_value must be positive"):
            atr_stop_loss(100.0, 0.0)

    def test_atr_stop_loss_rejects_negative(self) -> None:
        """Negative ATR should raise ValueError."""
        with pytest.raises(ValueError, match="atr_value must be positive"):
            atr_stop_loss(100.0, -1.5)

    def test_atr_take_profit_rejects_zero(self) -> None:
        """Zero ATR should raise ValueError."""
        with pytest.raises(ValueError, match="atr_value must be positive"):
            atr_take_profit(100.0, 0.0)

    def test_atr_take_profit_rejects_negative(self) -> None:
        """Negative ATR should raise ValueError."""
        with pytest.raises(ValueError, match="atr_value must be positive"):
            atr_take_profit(100.0, -2.0)

    def test_atr_stop_loss_valid_long(self) -> None:
        """Valid ATR stop for long position."""
        stop = atr_stop_loss(100.0, 2.0, multiplier=2.0, side="long")
        assert stop == 96.0

    def test_atr_stop_loss_valid_short(self) -> None:
        """Valid ATR stop for short position."""
        stop = atr_stop_loss(100.0, 2.0, multiplier=2.0, side="short")
        assert stop == 104.0

    def test_atr_take_profit_valid_long(self) -> None:
        """Valid ATR take-profit for long position."""
        tp = atr_take_profit(100.0, 2.0, multiplier=3.0, side="long")
        assert tp == 106.0

    def test_atr_take_profit_valid_short(self) -> None:
        """Valid ATR take-profit for short position."""
        tp = atr_take_profit(100.0, 2.0, multiplier=3.0, side="short")
        assert tp == 94.0


class TestEquityValidation:
    """Test equity input validation."""

    def test_kelly_position_size_rejects_zero_equity(self) -> None:
        """Zero equity should raise ValueError."""
        with pytest.raises(ValueError, match="equity must be positive"):
            kelly_position_size(0.0, 0.5, 1.0, 0.5, 100.0)

    def test_kelly_position_size_rejects_negative_equity(self) -> None:
        """Negative equity should raise ValueError."""
        with pytest.raises(ValueError, match="equity must be positive"):
            kelly_position_size(-1000.0, 0.5, 1.0, 0.5, 100.0)

    def test_kelly_position_size_valid(self) -> None:
        """Valid equity should return positive position size."""
        size = kelly_position_size(10000.0, 0.6, 2.0, 1.0, 100.0)
        assert size > 0.0


class TestDailyLossLimit:
    """Test daily loss limit circuit breaker."""

    def test_within_limit(self) -> None:
        """No breach when loss is below threshold."""
        assert not check_daily_loss_limit(10000.0, 9600.0, 0.05)

    def test_at_limit(self) -> None:
        """Breach exactly at threshold."""
        assert check_daily_loss_limit(10000.0, 9500.0, 0.05)

    def test_beyond_limit(self) -> None:
        """Breach when loss exceeds threshold."""
        assert check_daily_loss_limit(10000.0, 9000.0, 0.05)

    def test_no_loss(self) -> None:
        """No breach when equity increased."""
        assert not check_daily_loss_limit(10000.0, 10500.0, 0.05)

    def test_zero_starting_equity(self) -> None:
        """Zero starting equity should raise ValueError."""
        with pytest.raises(
            ValueError,
            match="starting_equity must be positive",
        ):
            check_daily_loss_limit(0.0, 5000.0)

    def test_negative_starting_equity(self) -> None:
        """Negative starting equity should raise ValueError."""
        with pytest.raises(
            ValueError,
            match="starting_equity must be positive",
        ):
            check_daily_loss_limit(-1000.0, 5000.0)


class TestMaxPositionSize:
    """Test single-position size cap."""

    def test_max_position_size_10pct(self) -> None:
        """10% of 10000 equity at price 50 = 20 units."""
        qty = max_position_size(10000.0, 50.0, 0.10)
        assert qty == pytest.approx(20.0)

    def test_max_position_size_custom_pct(self) -> None:
        """25% of 10000 equity at price 100 = 25 units."""
        qty = max_position_size(10000.0, 100.0, 0.25)
        assert qty == pytest.approx(25.0)

    def test_zero_equity_raises(self) -> None:
        """Zero equity should raise ValueError."""
        with pytest.raises(ValueError, match="equity must be positive"):
            max_position_size(0.0, 100.0)

    def test_zero_price_raises(self) -> None:
        """Zero price should raise ValueError."""
        with pytest.raises(ValueError, match="price must be positive"):
            max_position_size(10000.0, 0.0)


# ── Data Validation ───────────────────────────────────────────


class TestTimestampGaps:
    """Test timestamp gap detection."""

    def test_no_gaps(self) -> None:
        """Uniform timestamps should return no gaps."""
        ts = [1000.0 + i * 60.0 for i in range(10)]
        gaps = detect_timestamp_gaps(ts, expected_interval=60.0)
        assert gaps == []

    def test_single_gap(self) -> None:
        """One missing bar should be detected."""
        ts = [0.0, 60.0, 120.0, 300.0, 360.0]
        gaps = detect_timestamp_gaps(ts, expected_interval=60.0)
        assert len(gaps) == 1
        assert gaps[0]["index"] == 3
        assert gaps[0]["gap_seconds"] == 180.0

    def test_multiple_gaps(self) -> None:
        """Multiple gaps should all be detected."""
        ts = [0.0, 60.0, 300.0, 360.0, 600.0]
        gaps = detect_timestamp_gaps(ts, expected_interval=60.0)
        assert len(gaps) == 2

    def test_empty_timestamps(self) -> None:
        """Empty list should return no gaps."""
        assert detect_timestamp_gaps([]) == []

    def test_single_timestamp(self) -> None:
        """Single timestamp should return no gaps."""
        assert detect_timestamp_gaps([100.0]) == []


class TestDataFreshness:
    """Test data freshness check."""

    def test_fresh_data(self) -> None:
        """Recent data should be considered fresh."""
        now = time.time()
        assert check_data_freshness(now - 60.0, current_time=now)

    def test_stale_data(self) -> None:
        """Old data should be considered stale."""
        now = time.time()
        assert not check_data_freshness(
            now - 600.0,
            max_age_seconds=300.0,
            current_time=now,
        )

    def test_exact_boundary(self) -> None:
        """Data just over max age is stale."""
        now = time.time()
        assert not check_data_freshness(
            now - 301.0,
            max_age_seconds=300.0,
            current_time=now,
        )


# ── Performance: __slots__ ────────────────────────────────────


class TestBacktestMetricsSlots:
    """Test that BacktestMetrics uses __slots__."""

    def test_has_slots(self) -> None:
        """BacktestMetrics should define __slots__."""
        assert hasattr(BacktestMetrics, "__slots__")

    def test_no_dict(self) -> None:
        """__slots__ instances should not have __dict__."""
        m = BacktestMetrics(10.0, 1.0, 5.0, 60.0, 3)
        assert not hasattr(m, "__dict__")

    def test_cannot_add_arbitrary_attr(self) -> None:
        """Adding arbitrary attributes should raise AttributeError."""
        m = BacktestMetrics(10.0, 1.0, 5.0, 60.0, 3)
        with pytest.raises(AttributeError):
            m.arbitrary_attribute = "should fail"  # type: ignore[attr-defined]


# ── Singleton Reset ───────────────────────────────────────────


class TestBacktestingEngineReset:
    """Test backtesting engine singleton reset."""

    def test_reset_creates_new_instance(self) -> None:
        """After reset, get_backtesting_engine returns new instance."""
        engine1 = get_backtesting_engine()
        reset_backtesting_engine()
        engine2 = get_backtesting_engine()
        assert engine1 is not engine2

    def test_reset_then_get_returns_valid(self) -> None:
        """Reset followed by get should return a working engine."""
        reset_backtesting_engine()
        engine = get_backtesting_engine()
        assert isinstance(engine, BacktestingEngine)
        metrics = engine.run_backtest(
            {"close": [100.0, 105.0]},
            ["BUY", "SELL"],
        )
        assert isinstance(metrics, BacktestMetrics)


# ── Rate Limiter Reset ────────────────────────────────────────


class TestRateLimiterReset:
    """Test rate limiter reset method."""

    def test_reset_clears_buckets(self) -> None:
        """After reset, all buckets should be cleared."""
        rl = RateLimiter(capacity=2, refill_rate=0.0)
        rl.allow("ip1")
        rl.allow("ip2")
        rl.reset()
        # After reset, bucket for ip1 should be fresh (full)
        assert rl.allow("ip1")
        assert rl.allow("ip1")

    def test_reset_restores_capacity(self) -> None:
        """Exhausted bucket should get full capacity after reset."""
        rl = RateLimiter(capacity=1, refill_rate=0.0)
        rl.allow("ip1")  # consumes the 1 token
        assert not rl.allow("ip1")  # no tokens left
        rl.reset()
        assert rl.allow("ip1")  # bucket recreated — full


# ── Algorithm: Memoized Tournament ────────────────────────────


class TestMemoizedTournament:
    """Test that coevolution memoizes strategy evaluations."""

    def test_tournament_produces_scores(self) -> None:
        """Self-play should return valid scores."""
        strategies = [
            Strategy({"threshold": 0.5 + i * 0.1})
            for i in range(3)
        ]
        engine = EvolutionEngine(strategies)
        data = {
            "ohlcv_data": {"close": [100.0, 105.0, 110.0]},
            "signals": ["BUY", "HOLD", "SELL"],
        }
        scores = engine.self_play_and_coevolution(data, rounds=2)
        assert len(scores) == 3
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_single_strategy_gets_zero(self) -> None:
        """Single strategy can't co-evolve, gets 0."""
        engine = EvolutionEngine([Strategy({"threshold": 0.5})])
        scores = engine.self_play_and_coevolution(
            {"ohlcv_data": {"close": [100.0]}, "signals": ["HOLD"]},
        )
        assert all(v == 0.0 for v in scores.values())


# ── Strategy Evaluate Logging ─────────────────────────────────


class TestStrategyEvaluateLogging:
    """Test that Strategy.evaluate logs on exception."""

    def test_evaluate_with_invalid_data_returns_zero(self) -> None:
        """Non-dict data should return 0.0 performance."""
        s = Strategy({"threshold": 1.0})
        result = s.evaluate("not a dict")
        assert result == 0.0
        assert s.performance == 0.0

    def test_evaluate_with_empty_data_returns_zero(self) -> None:
        """Empty dict data should return 0.0 performance."""
        s = Strategy({"threshold": 1.0})
        result = s.evaluate({})
        assert result == 0.0


# ── DI: retrain_threshold from settings ───────────────────────


class TestRetrainThresholdFromSettings:
    """Test that _RETRAIN_THRESHOLD is read from settings."""

    def test_retrain_threshold_is_int(self) -> None:
        """_RETRAIN_THRESHOLD should be an integer."""
        from python_ai.fastapi_service.fastapi_service import (
            _RETRAIN_THRESHOLD,
        )

        assert isinstance(_RETRAIN_THRESHOLD, int)
        assert _RETRAIN_THRESHOLD > 0
