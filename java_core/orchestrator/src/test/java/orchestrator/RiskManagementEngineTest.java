package orchestrator;

import orchestrator.config.OrchestratorConfig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for RiskManagementEngine.
 *
 * Tests all 6 risk checks, circuit breaker logic, drawdown protection,
 * daily P&L tracking, and edge cases.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("RiskManagementEngine")
class RiskManagementEngineTest {

    private RiskManagementEngine engine;

    @BeforeEach
    void setUp() {
        // confidence=0.7, maxVol=2.0, maxPosition=0.1, maxDrawdown=0.05, dailyLoss=500
        engine = new RiskManagementEngine(0.7, 2.0, 0.1, 0.05, 500.0);
        engine.setEquity(10000.0);
    }

    // ---- Confidence threshold tests ----

    @Nested
    @DisplayName("Confidence threshold checks")
    class ConfidenceTests {
        @Test
        @DisplayName("Reject low-confidence BUY signal")
        void rejectLowConfidence() {
            var prediction = new ApiClient.PredictionResult("BUY", 0.5, 0.8);
            var decision = engine.evaluate(prediction, 1.0);
            assertFalse(decision.isApproved());
            assertTrue(decision.getReason().contains("LOW_CONFIDENCE"));
        }

        @Test
        @DisplayName("Approve high-confidence BUY signal")
        void approveHighConfidence() {
            var prediction = new ApiClient.PredictionResult("BUY", 0.85, 0.9);
            var decision = engine.evaluate(prediction, 1.0);
            assertTrue(decision.isApproved());
        }

        @Test
        @DisplayName("Approve exactly-at-threshold confidence")
        void approveExactThreshold() {
            var prediction = new ApiClient.PredictionResult("BUY", 0.7, 0.5);
            var decision = engine.evaluate(prediction, 1.0);
            assertTrue(decision.isApproved());
        }

        @Test
        @DisplayName("Reject confidence just below threshold")
        void rejectJustBelowThreshold() {
            var prediction = new ApiClient.PredictionResult("BUY", 0.699, 0.5);
            var decision = engine.evaluate(prediction, 1.0);
            assertFalse(decision.isApproved());
        }
    }

    // ---- Volatility cap tests ----

    @Nested
    @DisplayName("Volatility cap checks")
    class VolatilityTests {
        @Test
        @DisplayName("Reject trade when volatility exceeds cap")
        void rejectHighVolatility() {
            var prediction = new ApiClient.PredictionResult("BUY", 0.9, 0.8);
            var decision = engine.evaluate(prediction, 3.0);
            assertFalse(decision.isApproved());
            assertTrue(decision.getReason().contains("HIGH_VOLATILITY"));
        }

        @Test
        @DisplayName("Approve trade with low volatility")
        void approveLowVolatility() {
            var prediction = new ApiClient.PredictionResult("BUY", 0.9, 0.8);
            var decision = engine.evaluate(prediction, 1.5);
            assertTrue(decision.isApproved());
        }
    }

    // ---- Signal validation tests ----

    @Nested
    @DisplayName("Signal validation checks")
    class SignalTests {
        @Test
        @DisplayName("Accept BUY signal")
        void acceptBuy() {
            var prediction = new ApiClient.PredictionResult("BUY", 0.9, 0.8);
            var decision = engine.evaluate(prediction, 1.0);
            assertTrue(decision.isApproved());
        }

        @Test
        @DisplayName("Accept SELL signal")
        void acceptSell() {
            var prediction = new ApiClient.PredictionResult("SELL", 0.9, -0.8);
            var decision = engine.evaluate(prediction, 1.0);
            assertTrue(decision.isApproved());
        }

        @Test
        @DisplayName("Pass through HOLD signal")
        void passHold() {
            var prediction = new ApiClient.PredictionResult("HOLD", 0.3, 0.0);
            var decision = engine.evaluate(prediction, 5.0); // High vol but HOLD passes
            assertTrue(decision.isApproved());
            assertTrue(decision.getReason().contains("HOLD"));
        }

        @Test
        @DisplayName("Reject invalid signal type")
        void rejectInvalidSignal() {
            var prediction = new ApiClient.PredictionResult("LONG", 0.9, 0.8);
            var decision = engine.evaluate(prediction, 1.0);
            assertFalse(decision.isApproved());
            assertTrue(decision.getReason().contains("INVALID_SIGNAL"));
        }

        @Test
        @DisplayName("Signal case-insensitivity")
        void signalCaseInsensitive() {
            var prediction = new ApiClient.PredictionResult("buy", 0.9, 0.8);
            var decision = engine.evaluate(prediction, 1.0);
            assertTrue(decision.isApproved());
        }
    }

    // ---- Circuit breaker tests ----

    @Nested
    @DisplayName("Circuit breaker")
    class CircuitBreakerTests {
        @Test
        @DisplayName("Circuit breaker trips on daily loss limit")
        void tripOnDailyLoss() {
            assertFalse(engine.isCircuitBreakerTripped());
            engine.recordTradeOutcome(-600.0);
            assertTrue(engine.isCircuitBreakerTripped());
        }

        @Test
        @DisplayName("Reject all trades after circuit breaker trips")
        void rejectAfterTrip() {
            engine.recordTradeOutcome(-600.0);
            var prediction = new ApiClient.PredictionResult("BUY", 0.99, 0.95);
            var decision = engine.evaluate(prediction, 0.5);
            assertFalse(decision.isApproved());
            assertTrue(decision.getReason().contains("CIRCUIT_BREAKER"));
        }

        @Test
        @DisplayName("Reset daily clears circuit breaker")
        void resetClears() {
            engine.recordTradeOutcome(-600.0);
            assertTrue(engine.isCircuitBreakerTripped());
            engine.resetDaily();
            assertFalse(engine.isCircuitBreakerTripped());
        }

        @Test
        @DisplayName("Multiple small losses accumulate to trip")
        void accumulateLosses() {
            for (int i = 0; i < 10; i++) {
                engine.recordTradeOutcome(-60.0);
            }
            assertTrue(engine.isCircuitBreakerTripped());
            assertEquals(-600.0, engine.getDailyPnl(), 0.01);
        }
    }

    // ---- Drawdown protection tests ----

    @Nested
    @DisplayName("Drawdown protection")
    class DrawdownTests {
        @Test
        @DisplayName("Reject trade when drawdown exceeds max")
        void rejectExcessiveDrawdown() {
            // Use high daily loss limit so circuit breaker doesn't trip first
            RiskManagementEngine customEngine = new RiskManagementEngine(
                    0.7, 2.0, 0.1, 0.05, 10000.0);
            customEngine.setEquity(10000.0);
            customEngine.recordTradeOutcome(-600.0); // equity=9400, drawdown=6%

            var prediction = new ApiClient.PredictionResult("BUY", 0.9, 0.8);
            var decision = customEngine.evaluate(prediction, 1.0);
            assertFalse(decision.isApproved());
            assertTrue(decision.getReason().contains("MAX_DRAWDOWN"));
        }

        @Test
        @DisplayName("Allow trade within drawdown limit")
        void allowWithinDrawdown() {
            RiskManagementEngine customEngine = new RiskManagementEngine(
                    0.7, 2.0, 0.1, 0.05, 10000.0);
            customEngine.setEquity(10000.0);
            customEngine.recordTradeOutcome(-300.0); // equity=9700, drawdown=3%

            var prediction = new ApiClient.PredictionResult("BUY", 0.9, 0.8);
            var decision = customEngine.evaluate(prediction, 1.0);
            assertTrue(decision.isApproved());
        }
    }

    // ---- Position sizing tests ----

    @Nested
    @DisplayName("Position sizing")
    class PositionSizeTests {
        @Test
        @DisplayName("Max position size is equity * fraction")
        void maxPositionSizeCalc() {
            engine.setEquity(10000.0);
            assertEquals(1000.0, engine.getMaxPositionSizeDollars(), 0.01);
        }

        @Test
        @DisplayName("Position size updates with equity")
        void positionSizeUpdates() {
            engine.setEquity(10000.0);
            engine.recordTradeOutcome(500.0);
            assertEquals(1050.0, engine.getMaxPositionSizeDollars(), 0.01);
        }
    }

    // ---- P&L tracking tests ----

    @Nested
    @DisplayName("P&L tracking")
    class PnlTests {
        @Test
        @DisplayName("Track daily P&L correctly")
        void trackDailyPnl() {
            engine.recordTradeOutcome(100.0);
            engine.recordTradeOutcome(-50.0);
            engine.recordTradeOutcome(200.0);
            assertEquals(250.0, engine.getDailyPnl(), 0.01);
        }

        @Test
        @DisplayName("Equity updates with P&L")
        void equityUpdates() {
            engine.setEquity(10000.0);
            engine.recordTradeOutcome(500.0);
            assertEquals(10500.0, engine.getCurrentEquity(), 0.01);
        }

        @Test
        @DisplayName("Reset daily zeros P&L but not equity")
        void resetDailyKeepsEquity() {
            engine.recordTradeOutcome(250.0);
            engine.resetDaily();
            assertEquals(0.0, engine.getDailyPnl(), 0.01);
            assertEquals(10250.0, engine.getCurrentEquity(), 0.01);
        }
    }

    // ---- Legacy backward compatibility ----

    @Nested
    @DisplayName("Legacy approveAction()")
    class LegacyTests {
        @Test
        @DisplayName("Parse JSON and approve valid trade")
        void approveValidJson() {
            String json = "{\"signal\":\"BUY\",\"confidence\":0.9}";
            assertTrue(engine.approveAction(json, 1.0));
        }

        @Test
        @DisplayName("Parse JSON and reject low confidence")
        void rejectLowConfidenceJson() {
            String json = "{\"signal\":\"BUY\",\"confidence\":0.3}";
            assertFalse(engine.approveAction(json, 1.0));
        }

        @Test
        @DisplayName("Return false for invalid JSON")
        void rejectInvalidJson() {
            assertFalse(engine.approveAction("not json", 1.0));
        }
    }

    // ---- Config-based construction ----

    @Test
    @DisplayName("Construct from OrchestratorConfig")
    void constructFromConfig() {
        OrchestratorConfig config = new OrchestratorConfig.Builder()
                .confidenceThreshold(0.8)
                .maxVolatility(1.5)
                .maxPositionSize(0.05)
                .maxDrawdownPct(0.03)
                .dailyLossLimit(300.0)
                .build();

        RiskManagementEngine configEngine = new RiskManagementEngine(config);
        configEngine.setEquity(5000.0);

        // Should reject 0.75 confidence with 0.8 threshold
        var prediction = new ApiClient.PredictionResult("BUY", 0.75, 0.5);
        assertFalse(configEngine.evaluate(prediction, 1.0).isApproved());

        // Should approve 0.85 confidence
        var prediction2 = new ApiClient.PredictionResult("BUY", 0.85, 0.5);
        assertTrue(configEngine.evaluate(prediction2, 1.0).isApproved());
    }
}
