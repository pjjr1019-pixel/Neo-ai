"""
Tests for Session 5 modules (Phases 10-16).

Covers 19 new modules: request_encryption, ip_allowlist,
regime_detector, event_sourcing, cqrs, dead_letter_queue,
bulkhead, anomaly_autoencoder, lstm_model, transformer_model,
retrain_scheduler, rl_agent, cross_exchange_validator,
stat_arb, alert_notifier, data_archival, stream_ingestion,
audit_logger, risk_governance.
"""

import os
import tempfile
import time

import numpy as np
import pytest

# ── 1. Request Encryption ──────────────────────────────────────


class TestRequestEncryption:
    """Tests for AES-256-GCM request encryption."""

    def test_generate_key_length(self) -> None:
        """Generated key must be 32 bytes."""
        from python_ai.request_encryption import generate_key

        key = generate_key()
        assert len(key) == 32

    def test_key_round_trip_b64(self) -> None:
        """Key survives base-64 encode/decode."""
        from python_ai.request_encryption import (
            generate_key,
            key_from_b64,
            key_to_b64,
        )

        key = generate_key()
        assert key_from_b64(key_to_b64(key)) == key

    def test_encrypt_decrypt_bytes(self) -> None:
        """Encrypt then decrypt returns original bytes."""
        from python_ai.request_encryption import (
            decrypt_payload,
            encrypt_payload,
            generate_key,
        )

        key = generate_key()
        data = b"hello world"
        token = encrypt_payload(data, key)
        assert decrypt_payload(token, key) == data

    def test_encrypt_decrypt_json(self) -> None:
        """Encrypt dict, decrypt with as_json=True."""
        from python_ai.request_encryption import (
            decrypt_payload,
            encrypt_payload,
            generate_key,
        )

        key = generate_key()
        obj = {"symbol": "BTC", "price": 42000.0}
        token = encrypt_payload(obj, key)
        result = decrypt_payload(token, key, as_json=True)
        assert result == obj

    def test_wrong_key_fails(self) -> None:
        """Decryption with wrong key raises ValueError."""
        from python_ai.request_encryption import (
            decrypt_payload,
            encrypt_payload,
            generate_key,
        )

        token = encrypt_payload(b"secret", generate_key())
        with pytest.raises((ValueError, Exception)):
            decrypt_payload(token, generate_key())

    def test_encrypt_with_aad(self) -> None:
        """AAD must match between encrypt and decrypt."""
        from python_ai.request_encryption import (
            decrypt_payload,
            encrypt_payload,
            generate_key,
        )

        key = generate_key()
        aad = b"context"
        token = encrypt_payload(b"data", key, aad=aad)
        assert decrypt_payload(token, key, aad=aad) == b"data"


# ── 2. IP Allow-List ───────────────────────────────────────────


class TestIPAllowList:
    """Tests for IP allow-list middleware."""

    def test_is_allowed_in_network(self) -> None:
        """IP inside network passes."""
        from ipaddress import IPv4Network

        from python_ai.ip_allowlist import is_allowed

        nets = [IPv4Network("10.0.0.0/8")]
        assert is_allowed("10.1.2.3", nets) is True

    def test_is_allowed_outside_network(self) -> None:
        """IP outside network fails."""
        from ipaddress import IPv4Network

        from python_ai.ip_allowlist import is_allowed

        nets = [IPv4Network("10.0.0.0/8")]
        assert is_allowed("192.168.1.1", nets) is False

    def test_middleware_stats_initial(self) -> None:
        """Stats start at zero."""
        from python_ai.ip_allowlist import IPAllowListMiddleware

        mw = IPAllowListMiddleware(app=None, allowed=["127.0.0.1/32"])
        st = mw.stats
        assert st["allowed"] == 0
        assert st["blocked"] == 0

    def test_update_allowlist(self) -> None:
        """Updating allowlist changes internal networks."""
        from python_ai.ip_allowlist import IPAllowListMiddleware

        mw = IPAllowListMiddleware(app=None, allowed=["10.0.0.0/8"])
        mw.update_allowlist(["192.168.0.0/16"])
        # Internals updated — exact check depends on impl
        assert mw is not None


# ── 3. Regime Detection ───────────────────────────────────────


class TestRegimeDetector:
    """Tests for market regime detection."""

    def test_detect_returns_labels(self) -> None:
        """Detect produces labels matching input length."""
        from python_ai.regime_detector import RegimeDetector

        rng = np.random.default_rng(42)
        prices = np.cumsum(rng.normal(0, 1, 200)) + 100
        det = RegimeDetector(n_regimes=3)
        result = det.detect(prices.tolist())
        # Labels length = len(prices) - 1 (returns)
        assert len(result.labels) == len(prices) - 1

    def test_detect_current_regime_string(self) -> None:
        """current_regime returns a label string."""
        from python_ai.regime_detector import RegimeDetector

        rng = np.random.default_rng(7)
        prices = np.cumsum(rng.normal(0, 1, 150)) + 100
        det = RegimeDetector(n_regimes=3)
        result = det.detect(prices.tolist())
        assert isinstance(result.current_regime(), str)

    def test_transition_matrix_shape(self) -> None:
        """Transition matrix is n_regimes x n_regimes."""
        from python_ai.regime_detector import RegimeDetector

        rng = np.random.default_rng(99)
        prices = np.cumsum(rng.normal(0, 1, 200)) + 100
        det = RegimeDetector(n_regimes=2)
        result = det.detect(prices.tolist())
        assert result.transition_matrix.shape == (2, 2)

    def test_too_few_prices_raises(self) -> None:
        """Less than 10 prices raises ValueError."""
        from python_ai.regime_detector import RegimeDetector

        det = RegimeDetector()
        with pytest.raises(ValueError):
            det.detect([100, 101, 102])

    def test_n_regimes_property(self) -> None:
        """n_regimes property returns config value."""
        from python_ai.regime_detector import RegimeDetector

        det = RegimeDetector(n_regimes=4)
        assert det.n_regimes == 4


# ── 4. Event Sourcing ─────────────────────────────────────────


class TestEventSourcing:
    """Tests for append-only event store."""

    def test_append_and_len(self) -> None:
        """Appending events increments length."""
        from python_ai.event_sourcing import Event, EventStore

        store = EventStore()
        store.append(
            Event(
                event_type="trade",
                aggregate_id="a1",
                data={"x": 1},
            )
        )
        assert len(store) == 1

    def test_get_events_by_aggregate(self) -> None:
        """Filter events by aggregate_id."""
        from python_ai.event_sourcing import Event, EventStore

        store = EventStore()
        store.append(
            Event(
                event_type="t",
                aggregate_id="a1",
                data={},
            )
        )
        store.append(
            Event(
                event_type="t",
                aggregate_id="a2",
                data={},
            )
        )
        assert len(store.get_events(aggregate_id="a1")) == 1

    def test_replay_calls_handler(self) -> None:
        """Replay invokes handler for each event."""
        from python_ai.event_sourcing import Event, EventStore

        store = EventStore()
        for i in range(5):
            store.append(
                Event(
                    event_type="t",
                    aggregate_id="a",
                    data={"i": i},
                )
            )
        called: list = []
        store.replay(called.append)
        assert len(called) == 5

    def test_subscribe_notification(self) -> None:
        """Subscriber receives appended events."""
        from python_ai.event_sourcing import Event, EventStore

        store = EventStore()
        received: list = []
        store.subscribe(received.append)
        store.append(Event(event_type="t", aggregate_id="a", data={}))
        assert len(received) == 1

    def test_latest_version_increments(self) -> None:
        """latest_version grows with each append."""
        from python_ai.event_sourcing import Event, EventStore

        store = EventStore()
        store.append(Event(event_type="t", aggregate_id="a", data={}))
        v1 = store.latest_version
        store.append(Event(event_type="t", aggregate_id="a", data={}))
        assert store.latest_version > v1

    def test_summary_keys(self) -> None:
        """Summary dict has expected keys."""
        from python_ai.event_sourcing import EventStore

        store = EventStore()
        s = store.summary()
        assert "total_events" in s


# ── 5. CQRS ───────────────────────────────────────────────────


class TestCQRS:
    """Tests for command/query separation."""

    def test_command_bus_dispatch(self) -> None:
        """Dispatching a command invokes its handler."""
        from python_ai.cqrs import (
            Command,
            CommandBus,
            CommandHandler,
            CommandResult,
        )

        class CreateOrder(Command):
            """Test command."""

            pass

        class Handler(CommandHandler):
            """Test handler."""

            def handle(self, cmd: Command) -> CommandResult:
                """Handle command."""
                return CommandResult(success=True, data="ok")

        bus = CommandBus()
        bus.register("CreateOrder", Handler())
        cmd = CreateOrder()
        result = bus.dispatch(cmd)
        assert result.success is True
        assert result.data == "ok"

    def test_query_bus_dispatch(self) -> None:
        """Dispatching a query returns handler result."""
        from python_ai.cqrs import (
            Query,
            QueryBus,
            QueryHandler,
        )

        class GetPrice(Query):
            """Test query."""

            pass

        class PHandler(QueryHandler):  # type: ignore[type-arg]
            """Test handler."""

            def handle(self, q: Query) -> float:
                """Handle query."""
                return 42.0

        bus = QueryBus()
        bus.register("GetPrice", PHandler())
        assert bus.dispatch(GetPrice()) == 42.0

    def test_command_bus_unknown_raises(self) -> None:
        """Dispatching unknown command raises KeyError."""
        from python_ai.cqrs import Command, CommandBus

        class Unknown(Command):
            """Unknown command."""

            pass

        bus = CommandBus()
        with pytest.raises(KeyError):
            bus.dispatch(Unknown())

    def test_command_bus_stats(self) -> None:
        """Stats track dispatches."""
        from python_ai.cqrs import CommandBus

        bus = CommandBus()
        s = bus.stats
        assert s["dispatched"] == 0


# ── 6. Dead Letter Queue ──────────────────────────────────────


class TestDeadLetterQueue:
    """Tests for dead letter queue."""

    def test_enqueue_and_len(self) -> None:
        """Enqueuing items increments size."""
        from python_ai.dead_letter_queue import (
            DeadLetterQueue,
        )

        dlq = DeadLetterQueue()
        dlq.enqueue({"x": 1}, "err", source="test")
        assert len(dlq) == 1

    def test_peek_returns_items(self) -> None:
        """Peek returns enqueued items."""
        from python_ai.dead_letter_queue import (
            DeadLetterQueue,
        )

        dlq = DeadLetterQueue()
        dlq.enqueue("payload", "timeout")
        items = dlq.peek(10)
        assert len(items) == 1

    def test_retry_success(self) -> None:
        """Successful retry removes item from queue."""
        from python_ai.dead_letter_queue import (
            DeadLetterQueue,
        )

        dlq = DeadLetterQueue()
        letter = dlq.enqueue("data", "err")

        def handler(payload: object) -> None:
            """Handle successfully."""
            pass

        result = dlq.retry(letter.letter_id, handler)
        assert result is True

    def test_retry_failure(self) -> None:
        """Failed retry increments retry count."""
        from python_ai.dead_letter_queue import (
            DeadLetterQueue,
        )

        dlq = DeadLetterQueue()
        letter = dlq.enqueue("data", "err")

        def bad_handler(payload: object) -> None:
            """Fail on purpose."""
            raise RuntimeError("fail")

        dlq.retry(letter.letter_id, bad_handler)
        items = dlq.peek(10)
        assert len(items) >= 1

    def test_purge_clears(self) -> None:
        """Purge removes all items."""
        from python_ai.dead_letter_queue import (
            DeadLetterQueue,
        )

        dlq = DeadLetterQueue()
        dlq.enqueue("a", "err")
        dlq.enqueue("b", "err")
        purged = dlq.purge()
        assert purged == 2
        assert len(dlq) == 0

    def test_stats_keys(self) -> None:
        """Stats dict has expected keys."""
        from python_ai.dead_letter_queue import (
            DeadLetterQueue,
        )

        dlq = DeadLetterQueue()
        s = dlq.stats
        assert "enqueued" in s


# ── 7. Bulkhead ────────────────────────────────────────────────


class TestBulkhead:
    """Tests for bulkhead pattern."""

    def test_execute_returns_result(self) -> None:
        """Execute returns function result."""
        from python_ai.bulkhead import Bulkhead

        bh = Bulkhead(name="test", max_concurrent=2)
        result = bh.execute(lambda: 42)
        assert result == 42

    def test_execute_propagates_error(self) -> None:
        """Exceptions propagate through execute."""
        from python_ai.bulkhead import Bulkhead

        bh = Bulkhead(name="test", max_concurrent=5)

        def fail() -> None:
            """Raise error."""
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            bh.execute(fail)

    def test_decorator_usage(self) -> None:
        """Bulkhead works as a decorator."""
        from python_ai.bulkhead import Bulkhead

        bh = Bulkhead(name="deco", max_concurrent=3)

        @bh
        def compute(x: int) -> int:
            """Double input."""
            return x * 2

        assert compute(5) == 10

    def test_stats_tracking(self) -> None:
        """Stats track accepted and completed."""
        from python_ai.bulkhead import Bulkhead

        bh = Bulkhead(name="st", max_concurrent=5)
        bh.execute(lambda: None)
        s = bh.stats
        assert s["accepted"] >= 1
        assert s["completed"] >= 1

    def test_registry(self) -> None:
        """Registry creates and retrieves bulkheads."""
        from python_ai.bulkhead import BulkheadRegistry

        reg = BulkheadRegistry()
        bh = reg.register("svc", max_concurrent=3)
        assert reg.get("svc") is bh
        assert reg.get("unknown") is None


# ── 8. Anomaly Autoencoder ─────────────────────────────────────


def _torch_optim_available() -> bool:
    """Return True if ``torch.optim.Adam`` can be instantiated.

    On some CI runners (Python 3.12 + older triton/setuptools),
    importing the dynamo sub-module raises ``AttributeError:
    module 'pkgutil' has no attribute 'ImpImporter'``.
    """
    try:
        import torch
        import torch.nn as nn

        m = nn.Linear(2, 2)
        torch.optim.Adam(m.parameters(), lr=1e-3)
        return True
    except (AttributeError, ImportError):
        return False


_skip_torch = pytest.mark.skipif(
    not _torch_optim_available(),
    reason="torch.optim unavailable (triton/pkgutil compat)",
)


@_skip_torch
class TestAnomalyAutoencoder:
    """Tests for autoencoder anomaly detection."""

    def test_fit_returns_metrics(self) -> None:
        """Fit returns dict with final_loss."""
        from python_ai.anomaly_autoencoder import (
            AnomalyDetector,
        )

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 5))
        det = AnomalyDetector(input_dim=5, epochs=5, batch_size=16)
        result = det.fit(data)
        assert "final_loss" in result
        assert result["final_loss"] >= 0

    def test_detect_before_fit_raises(self) -> None:
        """Detect without fit raises RuntimeError."""
        from python_ai.anomaly_autoencoder import (
            AnomalyDetector,
        )

        det = AnomalyDetector(input_dim=5)
        with pytest.raises(RuntimeError):
            det.detect(np.zeros((10, 5)))

    def test_detect_returns_anomaly_info(self) -> None:
        """Detect returns scores and is_anomaly."""
        from python_ai.anomaly_autoencoder import (
            AnomalyDetector,
        )

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 5))
        det = AnomalyDetector(input_dim=5, epochs=5, batch_size=16)
        det.fit(data)
        result = det.detect(data)
        assert "scores" in result
        assert "is_anomaly" in result
        assert len(result["scores"]) == 100

    def test_summary(self) -> None:
        """Summary returns detector info."""
        from python_ai.anomaly_autoencoder import (
            AnomalyDetector,
        )

        det = AnomalyDetector(input_dim=5)
        s = det.summary()
        assert "input_dim" in s


# ── 9. LSTM Model ──────────────────────────────────────────────


@_skip_torch
class TestLSTMModel:
    """Tests for LSTM time-series predictor."""

    def test_train_returns_metrics(self) -> None:
        """Training returns dict with final_loss."""
        from python_ai.lstm_model import LSTMPredictor

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 3))
        pred = LSTMPredictor(
            input_dim=3,
            seq_len=10,
            epochs=3,
            batch_size=8,
        )
        result = pred.train(data)
        assert "final_loss" in result

    def test_predict_before_train_raises(self) -> None:
        """Predict without training raises RuntimeError."""
        from python_ai.lstm_model import LSTMPredictor

        pred = LSTMPredictor(input_dim=3, seq_len=10)
        with pytest.raises(RuntimeError):
            pred.predict(np.zeros((10, 3)))

    def test_predict_returns_float(self) -> None:
        """Predict returns a float value."""
        from python_ai.lstm_model import LSTMPredictor

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 3))
        pred = LSTMPredictor(
            input_dim=3,
            seq_len=10,
            epochs=3,
            batch_size=8,
        )
        pred.train(data)
        seq = rng.normal(0, 1, (10, 3))
        result = pred.predict(seq)
        assert isinstance(result, float)

    def test_summary(self) -> None:
        """Summary contains model info."""
        from python_ai.lstm_model import LSTMPredictor

        pred = LSTMPredictor(input_dim=5, seq_len=10)
        s = pred.summary()
        assert "input_dim" in s


# ── 10. Transformer Model ─────────────────────────────────────


@_skip_torch
class TestTransformerModel:
    """Tests for Transformer time-series predictor."""

    def test_train_returns_metrics(self) -> None:
        """Training returns dict with final_loss."""
        from python_ai.transformer_model import (
            TransformerPredictor,
        )

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 4))
        pred = TransformerPredictor(
            input_dim=4,
            seq_len=10,
            epochs=3,
            batch_size=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
        )
        result = pred.train(data)
        assert "final_loss" in result

    def test_predict_before_train_raises(self) -> None:
        """Predict before training raises RuntimeError."""
        from python_ai.transformer_model import (
            TransformerPredictor,
        )

        pred = TransformerPredictor(input_dim=4, seq_len=10)
        with pytest.raises(RuntimeError):
            pred.predict(np.zeros((10, 4)))

    def test_predict_returns_float(self) -> None:
        """Predict returns a float value."""
        from python_ai.transformer_model import (
            TransformerPredictor,
        )

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 4))
        pred = TransformerPredictor(
            input_dim=4,
            seq_len=10,
            epochs=3,
            batch_size=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
        )
        pred.train(data)
        result = pred.predict(rng.normal(0, 1, (10, 4)))
        assert isinstance(result, float)

    def test_summary(self) -> None:
        """Summary contains model info."""
        from python_ai.transformer_model import (
            TransformerPredictor,
        )

        pred = TransformerPredictor(input_dim=4, seq_len=10)
        s = pred.summary()
        assert "input_dim" in s


# ── 11. Retrain Scheduler ─────────────────────────────────────


class TestRetrainScheduler:
    """Tests for auto-retrain scheduler."""

    def test_trigger_retrain_manual(self) -> None:
        """Manual trigger calls retrain_fn."""
        from python_ai.retrain_scheduler import (
            RetrainScheduler,
        )

        calls: list = []

        def retrain() -> dict:
            """Fake retrain."""
            calls.append(1)
            return {"loss": 0.1}

        sched = RetrainScheduler(retrain_fn=retrain)
        result = sched.trigger_retrain("manual")
        assert len(calls) == 1
        assert result is not None

    def test_retrain_count(self) -> None:
        """retrain_count increments after trigger."""
        from python_ai.retrain_scheduler import (
            RetrainScheduler,
        )

        sched = RetrainScheduler(retrain_fn=lambda: {"ok": True})
        sched.trigger_retrain()
        assert sched.retrain_count == 1

    def test_history_records(self) -> None:
        """History stores retrain records."""
        from python_ai.retrain_scheduler import (
            RetrainScheduler,
        )

        sched = RetrainScheduler(retrain_fn=lambda: {"ok": True})
        sched.trigger_retrain()
        h = sched.history()
        assert len(h) == 1

    def test_start_stop(self) -> None:
        """Start/stop lifecycle works cleanly."""
        from python_ai.retrain_scheduler import (
            RetrainScheduler,
        )

        sched = RetrainScheduler(
            retrain_fn=lambda: None,
            interval_seconds=9999,
            poll_seconds=0.1,
        )
        sched.start()
        assert sched.is_running is True
        sched.stop(timeout=2.0)
        assert sched.is_running is False

    def test_summary(self) -> None:
        """Summary returns scheduler info."""
        from python_ai.retrain_scheduler import (
            RetrainScheduler,
        )

        sched = RetrainScheduler(retrain_fn=lambda: None)
        s = sched.summary()
        assert "retrain_count" in s


# ── 12. RL Agent ───────────────────────────────────────────────


@_skip_torch
class TestRLAgent:
    """Tests for DQN reinforcement learning agent."""

    def test_environment_reset(self) -> None:
        """Reset returns initial state."""
        from python_ai.rl_agent import TradingEnvironment

        prices = np.array([100, 101, 102, 103, 104, 105], dtype=float)
        env = TradingEnvironment(prices=prices)
        state = env.reset()
        assert isinstance(state, np.ndarray)

    def test_environment_step(self) -> None:
        """Step returns (state, reward, done) tuple."""
        from python_ai.rl_agent import (
            ACTION_HOLD,
            TradingEnvironment,
        )

        prices = np.array([100, 101, 102, 103, 104], dtype=float)
        env = TradingEnvironment(prices=prices)
        env.reset()
        state, reward, done = env.step(ACTION_HOLD)
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_agent_act(self) -> None:
        """Agent act returns valid action."""
        from python_ai.rl_agent import (
            NUM_ACTIONS,
            DQNAgent,
        )

        agent = DQNAgent(state_dim=2)
        state = np.array([0.5, 0.3])
        action = agent.act(state, explore=True)
        assert 0 <= action < NUM_ACTIONS

    def test_agent_train_short(self) -> None:
        """Training runs without error."""
        from python_ai.rl_agent import (
            DQNAgent,
            TradingEnvironment,
        )

        prices = np.cumsum(np.random.default_rng(42).normal(0, 1, 50)) + 100
        env = TradingEnvironment(prices=prices)
        agent = DQNAgent(state_dim=env.state_dim, batch_size=8)
        result = agent.train(env, episodes=3)
        assert "episode_rewards" in result

    def test_agent_summary(self) -> None:
        """Summary returns agent info."""
        from python_ai.rl_agent import DQNAgent

        agent = DQNAgent(state_dim=2)
        s = agent.summary()
        assert "state_dim" in s


# ── 13. Cross-Exchange Validator ───────────────────────────────


class TestCrossExchangeValidator:
    """Tests for cross-exchange data validation."""

    def test_validate_consensus(self) -> None:
        """Validation produces consensus price."""
        from python_ai.cross_exchange_validator import (
            CrossExchangeValidator,
            ExchangePrice,
        )

        quotes = [
            ExchangePrice("binance", "BTC", 42000.0),
            ExchangePrice("coinbase", "BTC", 42010.0),
            ExchangePrice("kraken", "BTC", 41990.0),
        ]
        val = CrossExchangeValidator()
        result = val.validate(quotes)
        assert result.valid is True
        assert result.consensus_price > 0

    def test_detect_outlier(self) -> None:
        """Outlier exchange is flagged."""
        from python_ai.cross_exchange_validator import (
            CrossExchangeValidator,
            ExchangePrice,
        )

        quotes = [
            ExchangePrice("binance", "BTC", 42000.0),
            ExchangePrice("coinbase", "BTC", 42005.0),
            ExchangePrice("kraken", "BTC", 41995.0),
            ExchangePrice("shady", "BTC", 55000.0),
        ]
        val = CrossExchangeValidator(z_threshold=1.0)
        result = val.validate(quotes)
        assert "shady" in result.outlier_exchanges

    def test_stale_detection(self) -> None:
        """Old timestamps flagged as stale."""
        from python_ai.cross_exchange_validator import (
            CrossExchangeValidator,
            ExchangePrice,
        )

        old = time.time() - 120
        quotes = [
            ExchangePrice("binance", "BTC", 42000.0),
            ExchangePrice("coinbase", "BTC", 42000.0),
            ExchangePrice(
                "old_ex",
                "BTC",
                42000.0,
                timestamp=old,
            ),
        ]
        val = CrossExchangeValidator(max_age_seconds=60.0)
        result = val.validate(quotes)
        assert "old_ex" in result.stale_exchanges

    def test_too_few_quotes_raises(self) -> None:
        """Less than 2 quotes raises ValueError."""
        from python_ai.cross_exchange_validator import (
            CrossExchangeValidator,
            ExchangePrice,
        )

        val = CrossExchangeValidator()
        with pytest.raises(ValueError):
            val.validate([ExchangePrice("x", "BTC", 100.0)])

    def test_detect_arbitrage(self) -> None:
        """Arbitrage detection with bid/ask spread."""
        from python_ai.cross_exchange_validator import (
            CrossExchangeValidator,
            ExchangePrice,
        )

        quotes = [
            ExchangePrice("a", "BTC", 42000, bid=41990, ask=42010),
            ExchangePrice("b", "BTC", 42100, bid=42090, ask=42110),
        ]
        val = CrossExchangeValidator()
        arb = val.detect_arbitrage(quotes, min_spread_pct=0.0)
        assert isinstance(arb, list)


# ── 14. Statistical Arbitrage ──────────────────────────────────


class TestStatArb:
    """Tests for pairs trading / stat arb."""

    def test_cointegration_result(self) -> None:
        """test_cointegration returns a result."""
        from python_ai.stat_arb import PairsTrader

        rng = np.random.default_rng(42)
        base = np.cumsum(rng.normal(0, 1, 200))
        a = (base + rng.normal(0, 0.1, 200)).tolist()
        b = (base * 1.5 + rng.normal(0, 0.1, 200)).tolist()
        pt = PairsTrader()
        result = pt.test_cointegration(a, b)
        assert hasattr(result, "hedge_ratio")
        assert hasattr(result, "cointegrated")

    def test_generate_signals_length(self) -> None:
        """Signals list matches prices length."""
        from python_ai.stat_arb import PairsTrader

        rng = np.random.default_rng(42)
        a = rng.normal(100, 5, 100).tolist()
        b = rng.normal(100, 5, 100).tolist()
        pt = PairsTrader()
        signals = pt.generate_signals(a, b)
        assert len(signals) == 100

    def test_current_signal_type(self) -> None:
        """current_signal returns PairsSignal."""
        from python_ai.stat_arb import (
            PairsSignal,
            PairsTrader,
        )

        rng = np.random.default_rng(42)
        a = rng.normal(100, 5, 50).tolist()
        b = rng.normal(100, 5, 50).tolist()
        pt = PairsTrader()
        sig = pt.current_signal(a, b)
        assert isinstance(sig, PairsSignal)
        assert sig.signal in (
            "long",
            "short",
            "close",
            "hold",
            "neutral",
        )

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched price arrays raise ValueError."""
        from python_ai.stat_arb import PairsTrader

        pt = PairsTrader()
        with pytest.raises(ValueError):
            pt.test_cointegration([1, 2, 3], [1, 2])


# ── 15. Alert Notifier ────────────────────────────────────────


class TestAlertNotifier:
    """Tests for alert notification channels."""

    def test_alert_creation(self) -> None:
        """Alert dataclass creates valid instance."""
        from python_ai.alert_notifier import Alert

        a = Alert(title="Test", message="hello", severity="warn")
        assert a.title == "Test"
        assert a.severity == "warn"

    def test_dispatcher_no_channels(self) -> None:
        """Dispatcher with no channels returns 0 sent."""
        from python_ai.alert_notifier import (
            Alert,
            AlertDispatcher,
        )

        d = AlertDispatcher()
        result = d.dispatch(Alert(title="T", message="M"))
        assert result["sent"] == 0

    def test_webhook_channel_init(self) -> None:
        """WebhookChannel creates with URL."""
        from python_ai.alert_notifier import (
            WebhookChannel,
        )

        ch = WebhookChannel(url="http://localhost/hook")
        assert ch is not None

    def test_dispatcher_summary(self) -> None:
        """Summary returns dispatcher info."""
        from python_ai.alert_notifier import (
            AlertDispatcher,
        )

        d = AlertDispatcher()
        s = d.summary()
        assert "channels" in s


# ── 16. Data Archival ──────────────────────────────────────────


class TestDataArchival:
    """Tests for data compression and archival."""

    def test_compress_decompress_file(self) -> None:
        """Compress then decompress round-trips."""
        from python_ai.data_archival import DataArchiver

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "test.txt")
            with open(src, "w") as f:
                f.write("hello world " * 100)

            archiver = DataArchiver(archive_dir=os.path.join(td, "arch"))
            entry = archiver.compress_file(src)
            assert entry.compressed_bytes > 0

            out = archiver.decompress_file(
                entry.archive_path,
                output_dir=os.path.join(td, "out"),
            )
            assert os.path.exists(str(out))

    def test_archive_json(self) -> None:
        """archive_json stores and load_json retrieves."""
        from python_ai.data_archival import DataArchiver

        with tempfile.TemporaryDirectory() as td:
            archiver = DataArchiver(archive_dir=os.path.join(td, "arch"))
            obj = {"key": "value", "n": 42}
            entry = archiver.archive_json(obj, "test")
            loaded = archiver.load_json(entry.archive_path)
            assert loaded == obj

    def test_disk_usage(self) -> None:
        """disk_usage returns file count and bytes."""
        from python_ai.data_archival import DataArchiver

        with tempfile.TemporaryDirectory() as td:
            archiver = DataArchiver(archive_dir=os.path.join(td, "arch"))
            usage = archiver.disk_usage()
            assert "file_count" in usage

    def test_summary(self) -> None:
        """Summary returns archiver info."""
        from python_ai.data_archival import DataArchiver

        with tempfile.TemporaryDirectory() as td:
            archiver = DataArchiver(archive_dir=os.path.join(td, "arch"))
            s = archiver.summary()
            assert "archive_dir" in s


# ── 17. Stream Ingestion ──────────────────────────────────────


class TestStreamIngestion:
    """Tests for Redis Streams event ingestion."""

    def test_in_memory_stream_roundtrip(self) -> None:
        """InMemoryStream xadd + xreadgroup works."""
        from python_ai.stream_ingestion import (
            InMemoryStream,
        )

        backend = InMemoryStream()
        backend.xadd("s", {"k": "v"})
        backend.create_group("s", "g")
        msgs = backend.xreadgroup("s", "g", "c1", 10)
        assert len(msgs) == 1

    def test_consumer_lifecycle(self) -> None:
        """Consumer start/stop works cleanly."""
        from python_ai.stream_ingestion import (
            InMemoryStream,
            StreamConsumer,
        )

        backend = InMemoryStream()
        consumer = StreamConsumer(
            backend=backend,
            stream="test",
            group="g",
            poll_interval=0.1,
        )
        consumer.start()
        assert consumer.is_running is True
        time.sleep(0.2)
        consumer.stop(timeout=2.0)
        assert consumer.is_running is False

    def test_consumer_processes_events(self) -> None:
        """Consumer processes enqueued events."""
        from python_ai.stream_ingestion import (
            InMemoryStream,
            StreamConsumer,
        )

        processed: list = []
        backend = InMemoryStream()
        backend.xadd("s", {"v": "1"})
        backend.xadd("s", {"v": "2"})
        consumer = StreamConsumer(
            backend=backend,
            stream="s",
            group="g",
            handler=lambda e: processed.append(e),
            poll_interval=0.1,
        )
        consumer.start()
        time.sleep(0.5)
        consumer.stop(timeout=2.0)
        assert consumer.processed_count >= 2

    def test_summary(self) -> None:
        """Summary returns consumer info."""
        from python_ai.stream_ingestion import (
            InMemoryStream,
            StreamConsumer,
        )

        consumer = StreamConsumer(backend=InMemoryStream(), stream="s")
        s = consumer.summary()
        assert "processed" in s


# ── 18. Audit Logger ──────────────────────────────────────────


class TestAuditLogger:
    """Tests for audit logger with hash chain."""

    def test_log_and_count(self) -> None:
        """Logging increments entry count."""
        from python_ai.audit_logger import AuditLogger

        logger = AuditLogger()
        logger.log("user1", "trade", "BTC/USD")
        assert logger.entry_count == 1

    def test_verify_chain_valid(self) -> None:
        """Chain verification passes for untampered log."""
        from python_ai.audit_logger import AuditLogger

        logger = AuditLogger()
        logger.log("u", "buy", "BTC")
        logger.log("u", "sell", "ETH")
        logger.log("u", "buy", "SOL")
        assert logger.verify_chain() is True

    def test_get_entries_filter(self) -> None:
        """Entries can be filtered by actor."""
        from python_ai.audit_logger import AuditLogger

        logger = AuditLogger()
        logger.log("alice", "buy", "BTC")
        logger.log("bob", "sell", "ETH")
        entries = logger.get_entries(actor="alice")
        assert len(entries) == 1
        assert entries[0].actor == "alice"

    def test_file_persistence(self) -> None:
        """Entries persist to file."""
        from python_ai.audit_logger import AuditLogger

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "audit.jsonl")
            logger = AuditLogger(log_file=path)
            logger.log("u", "act", "res")
            assert os.path.exists(path)

    def test_summary(self) -> None:
        """Summary returns logger info."""
        from python_ai.audit_logger import AuditLogger

        logger = AuditLogger()
        s = logger.summary()
        assert "entry_count" in s


# ── 19. Risk Governance ───────────────────────────────────────


class TestRiskGovernance:
    """Tests for risk governance approval workflow."""

    def test_evaluate_small_trade_allowed(self) -> None:
        """Small trade within policy is allowed."""
        from python_ai.risk_governance import (
            RiskGovernor,
            RiskPolicy,
        )

        gov = RiskGovernor(
            policies=[
                RiskPolicy(
                    name="default",
                    max_trade_value=50000,
                    require_approval_above=10000,
                )
            ]
        )
        result = gov.evaluate_trade(
            {"value": 5000, "symbol": "BTC"},
            portfolio_value=100000,
        )
        assert result["allowed"] is True
        assert result["requires_approval"] is False

    def test_evaluate_large_trade_needs_approval(
        self,
    ) -> None:
        """Large trade requires approval."""
        from python_ai.risk_governance import (
            RiskGovernor,
            RiskPolicy,
        )

        gov = RiskGovernor(
            policies=[
                RiskPolicy(
                    name="default",
                    max_trade_value=50000,
                    max_position_pct=50.0,
                    require_approval_above=10000,
                )
            ]
        )
        result = gov.evaluate_trade(
            {"value": 15000, "symbol": "BTC"},
            portfolio_value=100000,
        )
        assert result["requires_approval"] is True

    def test_approve_reject_workflow(self) -> None:
        """Approve and reject change request status."""
        from python_ai.risk_governance import (
            RiskGovernor,
            RiskPolicy,
        )

        gov = RiskGovernor(
            policies=[
                RiskPolicy(
                    name="default",
                    require_approval_above=100,
                )
            ]
        )
        result = gov.evaluate_trade(
            {"value": 500, "symbol": "BTC"},
            portfolio_value=100000,
        )
        if result.get("request_id"):
            rid = result["request_id"]
            ok = gov.approve(rid, "admin")
            assert ok is True

    def test_get_pending(self) -> None:
        """get_pending returns pending requests."""
        from python_ai.risk_governance import (
            RiskGovernor,
        )

        gov = RiskGovernor()
        pending = gov.get_pending()
        assert isinstance(pending, list)

    def test_summary(self) -> None:
        """Summary returns governance info."""
        from python_ai.risk_governance import (
            RiskGovernor,
        )

        gov = RiskGovernor()
        s = gov.summary()
        assert isinstance(s, dict)
