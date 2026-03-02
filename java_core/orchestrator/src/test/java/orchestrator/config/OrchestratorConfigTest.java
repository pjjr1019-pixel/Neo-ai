package orchestrator.config;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for OrchestratorConfig.
 *
 * Tests the Builder pattern, default values, validation rules,
 * and config loading from defaults. Environment variable resolution
 * is tested indirectly (properties file + defaults).
 */
@DisplayName("OrchestratorConfig")
class OrchestratorConfigTest {

    @Nested
    @DisplayName("Builder defaults")
    class DefaultsTests {

        @Test
        @DisplayName("Default config has sensible values")
        void defaultValues() {
            OrchestratorConfig config = new OrchestratorConfig.Builder().build();

            assertEquals("http://localhost:8000", config.getApiBaseUrl());
            assertEquals(30000, config.getApiTimeoutMs());
            assertEquals(3, config.getApiMaxRetries());
            assertEquals("", config.getApiAuthToken());
            assertEquals(0.7, config.getConfidenceThreshold(), 0.001);
            assertEquals(2.0, config.getMaxVolatility(), 0.001);
            assertEquals(0.1, config.getMaxPositionSize(), 0.001);
            assertEquals(0.05, config.getMaxDrawdownPct(), 0.001);
            assertEquals(500.0, config.getDailyLossLimit(), 0.001);
            assertEquals(60000, config.getLoopIntervalMs());
            assertEquals("BTC/USD", config.getTradingSymbol());
            assertEquals("", config.getDbUrl());
            assertEquals("localhost", config.getRedisHost());
            assertEquals(6379, config.getRedisPort());
            assertEquals(300, config.getRedisTtlSeconds());
            assertEquals(9090, config.getDashboardPort());
        }
    }

    @Nested
    @DisplayName("Builder customization")
    class CustomizationTests {

        @Test
        @DisplayName("All builder methods work correctly")
        void allBuilderMethods() {
            OrchestratorConfig config = new OrchestratorConfig.Builder()
                    .apiBaseUrl("http://api.example.com:9000")
                    .apiTimeoutMs(5000)
                    .apiMaxRetries(5)
                    .apiAuthToken("my-secret-token")
                    .confidenceThreshold(0.85)
                    .maxVolatility(1.5)
                    .maxPositionSize(0.05)
                    .maxDrawdownPct(0.03)
                    .dailyLossLimit(1000.0)
                    .loopIntervalMs(30000)
                    .tradingSymbol("ETH/USD")
                    .dbUrl("jdbc:postgresql://db:5432/mydb")
                    .dbUser("admin")
                    .dbPassword("secret")
                    .dbPoolSize(10)
                    .redisHost("redis.example.com")
                    .redisPort(6380)
                    .redisPassword("redis-pass")
                    .redisTtlSeconds(600)
                    .dashboardPort(8080)
                    .build();

            assertEquals("http://api.example.com:9000", config.getApiBaseUrl());
            assertEquals(5000, config.getApiTimeoutMs());
            assertEquals(5, config.getApiMaxRetries());
            assertEquals("my-secret-token", config.getApiAuthToken());
            assertEquals(0.85, config.getConfidenceThreshold(), 0.001);
            assertEquals(1.5, config.getMaxVolatility(), 0.001);
            assertEquals(0.05, config.getMaxPositionSize(), 0.001);
            assertEquals(0.03, config.getMaxDrawdownPct(), 0.001);
            assertEquals(1000.0, config.getDailyLossLimit(), 0.001);
            assertEquals(30000, config.getLoopIntervalMs());
            assertEquals("ETH/USD", config.getTradingSymbol());
            assertEquals("jdbc:postgresql://db:5432/mydb", config.getDbUrl());
            assertEquals("admin", config.getDbUser());
            assertEquals("secret", config.getDbPassword());
            assertEquals(10, config.getDbPoolSize());
            assertEquals("redis.example.com", config.getRedisHost());
            assertEquals(6380, config.getRedisPort());
            assertEquals("redis-pass", config.getRedisPassword());
            assertEquals(600, config.getRedisTtlSeconds());
            assertEquals(8080, config.getDashboardPort());
        }
    }

    @Nested
    @DisplayName("Validation")
    class ValidationTests {

        @Test
        @DisplayName("Null apiBaseUrl throws NPE")
        void nullApiBaseUrlThrows() {
            assertThrows(NullPointerException.class,
                    () -> new OrchestratorConfig.Builder()
                            .apiBaseUrl(null)
                            .build());
        }

        @Test
        @DisplayName("Zero timeout throws IllegalArgumentException")
        void zeroTimeoutThrows() {
            assertThrows(IllegalArgumentException.class,
                    () -> new OrchestratorConfig.Builder()
                            .apiTimeoutMs(0)
                            .build());
        }

        @Test
        @DisplayName("Negative timeout throws IllegalArgumentException")
        void negativeTimeoutThrows() {
            assertThrows(IllegalArgumentException.class,
                    () -> new OrchestratorConfig.Builder()
                            .apiTimeoutMs(-100)
                            .build());
        }

        @Test
        @DisplayName("Confidence threshold > 1 throws")
        void confidenceAboveOneThrows() {
            assertThrows(IllegalArgumentException.class,
                    () -> new OrchestratorConfig.Builder()
                            .confidenceThreshold(1.5)
                            .build());
        }

        @Test
        @DisplayName("Confidence threshold < 0 throws")
        void confidenceBelowZeroThrows() {
            assertThrows(IllegalArgumentException.class,
                    () -> new OrchestratorConfig.Builder()
                            .confidenceThreshold(-0.1)
                            .build());
        }

        @Test
        @DisplayName("Confidence threshold boundary 0.0 is valid")
        void confidenceZeroValid() {
            assertDoesNotThrow(
                    () -> new OrchestratorConfig.Builder()
                            .confidenceThreshold(0.0)
                            .build());
        }

        @Test
        @DisplayName("Confidence threshold boundary 1.0 is valid")
        void confidenceOneValid() {
            assertDoesNotThrow(
                    () -> new OrchestratorConfig.Builder()
                            .confidenceThreshold(1.0)
                            .build());
        }
    }

    @Nested
    @DisplayName("toString()")
    class ToStringTests {

        @Test
        @DisplayName("toString contains key fields")
        void toStringContent() {
            OrchestratorConfig config = new OrchestratorConfig.Builder()
                    .apiBaseUrl("http://localhost:8000")
                    .tradingSymbol("BTC/USD")
                    .build();

            String str = config.toString();
            assertTrue(str.contains("http://localhost:8000"));
            assertTrue(str.contains("BTC/USD"));
            assertTrue(str.contains("OrchestratorConfig"));
        }
    }

    @Nested
    @DisplayName("Static load()")
    class LoadTests {

        @Test
        @DisplayName("Load returns valid config with defaults")
        void loadReturnsDefaults() {
            OrchestratorConfig config = OrchestratorConfig.load();
            assertNotNull(config);
            assertNotNull(config.getApiBaseUrl());
            assertNotNull(config.getTradingSymbol());
            assertTrue(config.getApiTimeoutMs() > 0);
        }
    }
}
