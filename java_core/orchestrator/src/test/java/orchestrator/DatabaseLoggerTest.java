package orchestrator;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DatabaseLogger.
 *
 * Tests no-op mode (when no DB is configured), action logging,
 * and graceful degradation behavior. Does NOT require a running
 * database — tests the offline/disabled code paths.
 */
@DisplayName("DatabaseLogger")
class DatabaseLoggerTest {

    private DatabaseLogger logger;

    @BeforeEach
    void setUp() {
        // Create a no-op logger (no database configured)
        logger = new DatabaseLogger();
    }

    @Nested
    @DisplayName("No-op mode")
    class NoOpTests {

        @Test
        @DisplayName("Logger reports unavailable when no DB")
        void reportsUnavailable() {
            assertFalse(logger.isAvailable());
        }

        @Test
        @DisplayName("Logging does not throw when unavailable")
        void logDoesNotThrow() {
            assertDoesNotThrow(() -> logger.logAction("TEST", "test details"));
        }

        @Test
        @DisplayName("Log count returns -1 when unavailable")
        void logCountUnavailable() {
            assertEquals(-1, logger.getLogCount());
        }

        @Test
        @DisplayName("Recent logs returns empty list when unavailable")
        void recentLogsEmpty() {
            assertTrue(logger.getRecentLogs(10).isEmpty());
        }

        @Test
        @DisplayName("Close does not throw when no connection")
        void closeDoesNotThrow() {
            assertDoesNotThrow(() -> logger.close());
        }
    }

    @Nested
    @DisplayName("Graceful degradation")
    class DegradationTests {

        @Test
        @DisplayName("Empty URL creates disabled logger")
        void emptyUrlDisabled() {
            DatabaseLogger emptyLogger = new DatabaseLogger("", "", "", 1);
            assertFalse(emptyLogger.isAvailable());
        }

        @Test
        @DisplayName("Null URL creates disabled logger")
        void nullUrlDisabled() {
            DatabaseLogger nullLogger = new DatabaseLogger(null, "", "", 1);
            assertFalse(nullLogger.isAvailable());
        }

        @Test
        @DisplayName("Invalid URL creates disabled logger")
        void invalidUrlDisabled() {
            DatabaseLogger badLogger = new DatabaseLogger(
                    "jdbc:postgresql://nonexistent:5432/nope", "user", "pass", 1);
            assertFalse(badLogger.isAvailable());
        }

        @Test
        @DisplayName("Multiple log calls on disabled logger are safe")
        void multipleLogs() {
            for (int i = 0; i < 100; i++) {
                final int idx = i;
                assertDoesNotThrow(
                        () -> logger.logAction("BATCH_" + idx, "detail " + idx));
            }
        }
    }

    @Nested
    @DisplayName("API contract")
    class ApiContractTests {

        @Test
        @DisplayName("logAction accepts null details")
        void logNullDetails() {
            assertDoesNotThrow(() -> logger.logAction("TEST", null));
        }

        @Test
        @DisplayName("getRecentLogs with zero limit returns empty")
        void zeroLimitReturnsEmpty() {
            assertTrue(logger.getRecentLogs(0).isEmpty());
        }

        @Test
        @DisplayName("Close is idempotent")
        void closeIdempotent() {
            assertDoesNotThrow(() -> {
                logger.close();
                logger.close();
                logger.close();
            });
        }
    }
}
