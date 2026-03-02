package orchestrator;

import orchestrator.config.OrchestratorConfig;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Tests for AutonomousLoop.
 *
 * Uses Mockito to mock all dependencies (ApiClient, RiskManagementEngine,
 * DataFeedClient, DatabaseLogger, RedisCache). Tests the trading pipeline
 * cycle execution, start/stop lifecycle, and status reporting.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("AutonomousLoop")
class AutonomousLoopTest {

    @Mock private ApiClient apiClient;
    @Mock private RiskManagementEngine riskEngine;
    @Mock private DataFeedClient dataFeed;
    @Mock private DatabaseLogger dbLogger;
    @Mock private RedisCache cache;

    private AutonomousLoop loop;

    @BeforeEach
    void setUp() {
        loop = new AutonomousLoop(apiClient, riskEngine, dataFeed,
                dbLogger, cache, "BTC/USD", 60000);
    }

    @AfterEach
    void tearDown() {
        if (loop.isRunning()) {
            loop.stop();
        }
    }

    @Nested
    @DisplayName("Lifecycle")
    class LifecycleTests {

        @Test
        @DisplayName("Loop starts correctly")
        void startsCorrectly() {
            when(apiClient.healthCheck()).thenReturn(true);
            loop.start();
            assertTrue(loop.isRunning());
        }

        @Test
        @DisplayName("Loop stops gracefully")
        void stopsGracefully() {
            when(apiClient.healthCheck()).thenReturn(true);
            loop.start();
            assertTrue(loop.isRunning());
            loop.stop();
            assertFalse(loop.isRunning());
        }

        @Test
        @DisplayName("Double start is ignored")
        void doubleStartIgnored() {
            when(apiClient.healthCheck()).thenReturn(true);
            loop.start();
            loop.start(); // Should not throw
            assertTrue(loop.isRunning());
        }

        @Test
        @DisplayName("Stop without start is safe")
        void stopWithoutStart() {
            assertDoesNotThrow(() -> loop.stop());
        }

        @Test
        @DisplayName("Starts even when health check fails")
        void startsWithUnhealthyBackend() {
            when(apiClient.healthCheck()).thenReturn(false);
            loop.start();
            assertTrue(loop.isRunning());
        }
    }

    @Nested
    @DisplayName("Cycle execution")
    class CycleTests {

        @Test
        @DisplayName("Execute cycle with approved BUY trade")
        void executeBuyTrade() throws Exception {
            // Arrange
            Map<String, List<Double>> ohlcv = Map.of(
                    "open", List.of(100.0), "high", List.of(105.0),
                    "low", List.of(95.0), "close", List.of(102.0),
                    "volume", List.of(5000.0)
            );
            Map<String, Double> features = Map.of("rsi", 65.0, "macd", 0.5);
            var prediction = new ApiClient.PredictionResult("BUY", 0.9, 0.8);
            var decision = new RiskManagementEngine.RiskDecision(true, "All checks passed");

            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(ohlcv);
            when(dataFeed.getCurrentVolatility("BTC/USD")).thenReturn(1.0);
            when(apiClient.computeFeatures(eq("BTC/USD"), any())).thenReturn(features);
            when(apiClient.predict(features)).thenReturn(prediction);
            when(riskEngine.evaluate(prediction, 1.0)).thenReturn(decision);
            when(riskEngine.getMaxPositionSizeDollars()).thenReturn(1000.0);
            when(apiClient.sendFeedback(any(), anyDouble())).thenReturn(true);

            // Act
            loop.executeCycle();

            // Assert
            assertEquals(1, loop.getCycleCount());
            assertEquals("BUY", loop.getLastAction());
            assertEquals(0.9, loop.getLastConfidence(), 0.01);
            assertEquals(1, loop.getTradeHistory().size());
            verify(dbLogger, atLeast(1)).logAction(anyString(), anyString());
        }

        @Test
        @DisplayName("Execute cycle with rejected trade")
        void executeRejectedTrade() throws Exception {
            Map<String, List<Double>> ohlcv = Map.of(
                    "close", List.of(100.0), "open", List.of(100.0),
                    "high", List.of(100.0), "low", List.of(100.0),
                    "volume", List.of(1000.0)
            );
            Map<String, Double> features = Map.of("rsi", 30.0);
            var prediction = new ApiClient.PredictionResult("BUY", 0.5, 0.3);
            var decision = new RiskManagementEngine.RiskDecision(
                    false, "LOW_CONFIDENCE: too low");

            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(ohlcv);
            when(dataFeed.getCurrentVolatility("BTC/USD")).thenReturn(1.0);
            when(apiClient.computeFeatures(eq("BTC/USD"), any())).thenReturn(features);
            when(apiClient.predict(features)).thenReturn(prediction);
            when(riskEngine.evaluate(prediction, 1.0)).thenReturn(decision);

            loop.executeCycle();

            assertEquals(1, loop.getCycleCount());
            assertEquals(0, loop.getTradeHistory().size());
        }

        @Test
        @DisplayName("Skip cycle when no market data")
        void skipNoMarketData() {
            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(Map.of());

            loop.executeCycle();

            assertEquals(1, loop.getCycleCount());
            assertEquals(0, loop.getTradeHistory().size());
            verifyNoInteractions(apiClient);
        }

        @Test
        @DisplayName("Skip cycle when null market data")
        void skipNullMarketData() {
            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(null);

            loop.executeCycle();

            assertEquals(1, loop.getCycleCount());
            assertEquals(0, loop.getTradeHistory().size());
        }

        @Test
        @DisplayName("Skip cycle when features are empty")
        void skipEmptyFeatures() throws Exception {
            Map<String, List<Double>> ohlcv = Map.of(
                    "close", List.of(100.0), "open", List.of(100.0),
                    "high", List.of(100.0), "low", List.of(100.0),
                    "volume", List.of(1000.0)
            );
            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(ohlcv);
            when(dataFeed.getCurrentVolatility("BTC/USD")).thenReturn(1.0);
            when(apiClient.computeFeatures(eq("BTC/USD"), any())).thenReturn(Map.of());

            loop.executeCycle();

            assertEquals(1, loop.getCycleCount());
            assertEquals(0, loop.getTradeHistory().size());
        }

        @Test
        @DisplayName("HOLD signal results in no trade")
        void holdNoTrade() throws Exception {
            Map<String, List<Double>> ohlcv = Map.of(
                    "close", List.of(100.0), "open", List.of(100.0),
                    "high", List.of(100.0), "low", List.of(100.0),
                    "volume", List.of(1000.0)
            );
            Map<String, Double> features = Map.of("rsi", 50.0);
            var prediction = new ApiClient.PredictionResult("HOLD", 0.9, 0.0);
            var decision = new RiskManagementEngine.RiskDecision(true, "HOLD pass");

            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(ohlcv);
            when(dataFeed.getCurrentVolatility("BTC/USD")).thenReturn(1.0);
            when(apiClient.computeFeatures(eq("BTC/USD"), any())).thenReturn(features);
            when(apiClient.predict(features)).thenReturn(prediction);
            when(riskEngine.evaluate(prediction, 1.0)).thenReturn(decision);

            loop.executeCycle();

            assertEquals(0, loop.getTradeHistory().size());
        }

        @Test
        @DisplayName("Exception during cycle does not crash loop")
        void exceptionHandled() throws Exception {
            when(dataFeed.fetchOHLCV("BTC/USD"))
                    .thenThrow(new RuntimeException("Simulated failure"));

            assertDoesNotThrow(() -> loop.executeCycle());
            assertEquals(1, loop.getCycleCount());
        }
    }

    @Nested
    @DisplayName("Status reporting")
    class StatusTests {

        @Test
        @DisplayName("Status includes required fields")
        void statusContainsFields() {
            when(riskEngine.getDailyPnl()).thenReturn(0.0);
            when(riskEngine.isCircuitBreakerTripped()).thenReturn(false);

            Map<String, Object> status = loop.getStatus();
            assertTrue(status.containsKey("running"));
            assertTrue(status.containsKey("symbol"));
            assertTrue(status.containsKey("cycleCount"));
            assertTrue(status.containsKey("lastAction"));
            assertTrue(status.containsKey("totalTrades"));
            assertTrue(status.containsKey("dailyPnl"));
        }

        @Test
        @DisplayName("Status reflects current state")
        void statusReflectsState() {
            when(riskEngine.getDailyPnl()).thenReturn(150.0);
            when(riskEngine.isCircuitBreakerTripped()).thenReturn(false);

            Map<String, Object> status = loop.getStatus();
            assertEquals(false, status.get("running"));
            assertEquals("BTC/USD", status.get("symbol"));
            assertEquals(0L, status.get("cycleCount"));
        }
    }

    @Nested
    @DisplayName("Trade records")
    class TradeRecordTests {

        @Test
        @DisplayName("Trade record contains correct data")
        void tradeRecordData() throws Exception {
            Map<String, List<Double>> ohlcv = Map.of(
                    "close", List.of(100.0), "open", List.of(100.0),
                    "high", List.of(100.0), "low", List.of(100.0),
                    "volume", List.of(1000.0)
            );
            Map<String, Double> features = Map.of("rsi", 70.0);
            var prediction = new ApiClient.PredictionResult("SELL", 0.85, -0.5);
            var decision = new RiskManagementEngine.RiskDecision(true, "OK");

            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(ohlcv);
            when(dataFeed.getCurrentVolatility("BTC/USD")).thenReturn(1.0);
            when(apiClient.computeFeatures(eq("BTC/USD"), any())).thenReturn(features);
            when(apiClient.predict(features)).thenReturn(prediction);
            when(riskEngine.evaluate(prediction, 1.0)).thenReturn(decision);
            when(riskEngine.getMaxPositionSizeDollars()).thenReturn(500.0);
            when(apiClient.sendFeedback(any(), anyDouble())).thenReturn(true);

            loop.executeCycle();

            List<AutonomousLoop.TradeRecord> trades = loop.getTradeHistory();
            assertEquals(1, trades.size());
            assertEquals("SELL", trades.get(0).getAction());
            assertEquals("BTC/USD", trades.get(0).getSymbol());
            assertEquals(0.85, trades.get(0).getConfidence(), 0.01);
            assertNotNull(trades.get(0).getTimestamp());
        }

        @Test
        @DisplayName("Trade record toJson produces valid JSON")
        void tradeRecordJson() throws Exception {
            Map<String, List<Double>> ohlcv = Map.of(
                    "close", List.of(100.0), "open", List.of(100.0),
                    "high", List.of(100.0), "low", List.of(100.0),
                    "volume", List.of(1000.0)
            );
            Map<String, Double> features = Map.of("rsi", 70.0);
            var prediction = new ApiClient.PredictionResult("BUY", 0.9, 0.7);
            var decision = new RiskManagementEngine.RiskDecision(true, "OK");

            when(dataFeed.fetchOHLCV("BTC/USD")).thenReturn(ohlcv);
            when(dataFeed.getCurrentVolatility("BTC/USD")).thenReturn(1.0);
            when(apiClient.computeFeatures(eq("BTC/USD"), any())).thenReturn(features);
            when(apiClient.predict(features)).thenReturn(prediction);
            when(riskEngine.evaluate(prediction, 1.0)).thenReturn(decision);
            when(riskEngine.getMaxPositionSizeDollars()).thenReturn(500.0);
            when(apiClient.sendFeedback(any(), anyDouble())).thenReturn(true);

            loop.executeCycle();

            String json = loop.getTradeHistory().get(0).toJson();
            assertTrue(json.contains("\"action\":\"BUY\""));
            assertTrue(json.contains("\"symbol\":\"BTC/USD\""));
        }

        @Test
        @DisplayName("Trade history is immutable copy")
        void tradeHistoryImmutable() {
            List<AutonomousLoop.TradeRecord> trades = loop.getTradeHistory();
            assertThrows(UnsupportedOperationException.class,
                    () -> trades.add(null));
        }
    }

    @Nested
    @DisplayName("Config-based construction")
    class ConfigTests {

        @Test
        @DisplayName("Construct from OrchestratorConfig")
        void constructFromConfig() {
            OrchestratorConfig config = new OrchestratorConfig.Builder()
                    .tradingSymbol("ETH/USD")
                    .loopIntervalMs(30000)
                    .build();

            AutonomousLoop configLoop = new AutonomousLoop(
                    config, apiClient, riskEngine, dataFeed, dbLogger, cache);

            when(riskEngine.getDailyPnl()).thenReturn(0.0);
            when(riskEngine.isCircuitBreakerTripped()).thenReturn(false);

            Map<String, Object> status = configLoop.getStatus();
            assertEquals("ETH/USD", status.get("symbol"));
        }
    }
}
