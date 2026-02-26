package orchestrator;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DatabaseLoggerTest {
    @Test
    void testLogActionNoException() {
        DatabaseLogger logger = new DatabaseLogger(
            "jdbc:postgresql://localhost:5432/neoai_db", "neoai", "neoai123");
        assertDoesNotThrow(() -> logger.logAction("TEST", "Integration test log entry"));
    }
}
