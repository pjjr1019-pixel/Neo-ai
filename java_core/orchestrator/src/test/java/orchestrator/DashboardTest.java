package orchestrator;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Tests for Dashboard HTTP server.
 *
 * Uses mocked dependencies and a real HTTP server to verify endpoints.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("Dashboard")
class DashboardTest {

    @Mock private AutonomousLoop loop;
    @Mock private RiskManagementEngine riskEngine;
    @Mock private ApiClient apiClient;
    @Mock private DatabaseLogger dbLogger;
    @Mock private RedisCache cache;

    private Dashboard dashboard;
    private static final int TEST_PORT = 19090;

    @BeforeEach
    void setUp() throws Exception {
        dashboard = new Dashboard(TEST_PORT, loop, riskEngine,
                apiClient, dbLogger, cache);
    }

    @AfterEach
    void tearDown() {
        if (dashboard.isRunning()) {
            dashboard.stop();
        }
    }

    @Nested
    @DisplayName("Lifecycle")
    class LifecycleTests {

        @Test
        @DisplayName("Dashboard starts and reports running")
        void startsCorrectly() throws Exception {
            dashboard.start();
            assertTrue(dashboard.isRunning());
        }

        @Test
        @DisplayName("Dashboard stops cleanly")
        void stopsCorrectly() throws Exception {
            dashboard.start();
            assertTrue(dashboard.isRunning());
            dashboard.stop();
            assertFalse(dashboard.isRunning());
        }

        @Test
        @DisplayName("Stop without start does not throw")
        void stopWithoutStart() {
            assertDoesNotThrow(() -> dashboard.stop());
        }
    }

    @Nested
    @DisplayName("HTTP endpoints")
    class EndpointTests {

        @Test
        @DisplayName("/status returns 200 with JSON")
        void statusEndpoint() throws Exception {
            when(loop.getStatus()).thenReturn(Map.of(
                    "running", true,
                    "symbol", "BTC/USD",
                    "cycleCount", 5L,
                    "lastAction", "BUY",
                    "lastConfidence", 0.85,
                    "lastPrediction", 0.7,
                    "totalTrades", 3,
                    "dailyPnl", 150.0,
                    "circuitBreaker", false
            ));

            dashboard.start();

            String response = httpGet("/status");
            assertNotNull(response);
            assertTrue(response.contains("BTC/USD") || response.contains("running"));
        }

        @Test
        @DisplayName("/trades returns trade history")
        void tradesEndpoint() throws Exception {
            when(loop.getTradeHistory()).thenReturn(List.of());

            dashboard.start();

            String response = httpGet("/trades");
            assertNotNull(response);
            assertTrue(response.contains("totalTrades"));
        }

        @Test
        @DisplayName("/risk returns risk state")
        void riskEndpoint() throws Exception {
            when(riskEngine.getDailyPnl()).thenReturn(100.0);
            when(riskEngine.getCurrentEquity()).thenReturn(10100.0);
            when(riskEngine.isCircuitBreakerTripped()).thenReturn(false);
            when(riskEngine.getMaxPositionSizeDollars()).thenReturn(1010.0);

            dashboard.start();

            String response = httpGet("/risk");
            assertNotNull(response);
            assertTrue(response.contains("dailyPnl"));
        }

        @Test
        @DisplayName("/health returns health check")
        void healthEndpoint() throws Exception {
            when(apiClient.healthCheck()).thenReturn(true);
            when(dbLogger.isAvailable()).thenReturn(false);
            when(cache.isAvailable()).thenReturn(false);
            when(loop.isRunning()).thenReturn(true);

            dashboard.start();

            String response = httpGet("/health");
            assertNotNull(response);
            assertTrue(response.contains("status"));
        }

        @Test
        @DisplayName("/logs returns recent logs")
        void logsEndpoint() throws Exception {
            when(dbLogger.getRecentLogs(100)).thenReturn(
                    List.of("[2024-01-01] TEST — test entry"));

            dashboard.start();

            String response = httpGet("/logs");
            assertNotNull(response);
            assertTrue(response.contains("count"));
        }

        @Test
        @DisplayName("/metrics returns Prometheus format")
        void metricsEndpoint() throws Exception {
            when(loop.getStatus()).thenReturn(Map.of(
                    "running", true,
                    "cycleCount", 10L,
                    "totalTrades", 5,
                    "dailyPnl", 200.0,
                    "lastConfidence", 0.9,
                    "circuitBreaker", false
            ));

            dashboard.start();

            String response = httpGet("/metrics");
            assertNotNull(response);
            assertTrue(response.contains("neo_cycle_count"));
            assertTrue(response.contains("neo_total_trades"));
        }
    }

    // ---- HTTP helper ----

    private String httpGet(String path) throws Exception {
        URL url = new URL("http://localhost:" + TEST_PORT + path);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(3000);
        conn.setReadTimeout(3000);

        java.io.InputStream stream;
        try {
            stream = conn.getInputStream();
        } catch (java.io.IOException e) {
            // Non-2xx responses (e.g. 503) use error stream
            stream = conn.getErrorStream();
            if (stream == null) throw e;
        }

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            return sb.toString();
        } finally {
            conn.disconnect();
        }
    }
}
