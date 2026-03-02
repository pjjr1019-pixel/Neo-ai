package orchestrator;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ApiClient.PredictionResult.
 *
 * ApiClient itself requires a live HTTP server, so unit tests focus on
 * the PredictionResult inner class and constructor behavior. Full API
 * integration tests are handled separately.
 */
@DisplayName("ApiClient")
class ApiClientTest {

    @Nested
    @DisplayName("PredictionResult")
    class PredictionResultTests {

        @Test
        @DisplayName("Constructor stores all fields correctly")
        void constructorStoresFields() {
            var result = new ApiClient.PredictionResult("BUY", 0.95, 0.78);
            assertEquals("BUY", result.getAction());
            assertEquals(0.95, result.getConfidence(), 0.0001);
            assertEquals(0.78, result.getPrediction(), 0.0001);
        }

        @Test
        @DisplayName("toJson produces valid JSON string")
        void toJsonValid() {
            var result = new ApiClient.PredictionResult("SELL", 0.85, -0.5);
            String json = result.toJson();
            assertTrue(json.contains("\"action\":\"SELL\""));
            assertTrue(json.contains("\"confidence\":"));
            assertTrue(json.contains("\"prediction\":"));
        }

        @Test
        @DisplayName("toString is human-readable")
        void toStringReadable() {
            var result = new ApiClient.PredictionResult("HOLD", 0.5, 0.0);
            String str = result.toString();
            assertTrue(str.contains("HOLD"));
            assertTrue(str.contains("PredictionResult"));
        }

        @Test
        @DisplayName("Zero confidence is valid")
        void zeroConfidence() {
            var result = new ApiClient.PredictionResult("HOLD", 0.0, 0.0);
            assertEquals(0.0, result.getConfidence(), 0.0001);
        }

        @Test
        @DisplayName("Negative prediction is valid")
        void negativePrediction() {
            var result = new ApiClient.PredictionResult("SELL", 0.9, -1.5);
            assertEquals(-1.5, result.getPrediction(), 0.0001);
        }
    }

    @Nested
    @DisplayName("Constructor validation")
    class ConstructorTests {

        @Test
        @DisplayName("Construct with valid base URL")
        void constructValid() {
            assertDoesNotThrow(
                    () -> new ApiClient("http://localhost:8000", 5000, 3, ""));
        }

        @Test
        @DisplayName("Construct with null auth token defaults to empty")
        void nullAuthToken() {
            ApiClient client = new ApiClient("http://localhost:8000", 5000, 3, null);
            // Should not throw — null token treated as no-auth
            assertNotNull(client);
        }

        @Test
        @DisplayName("Health check returns false when server unreachable")
        void healthCheckFails() {
            ApiClient client = new ApiClient("http://localhost:59999", 1000, 0, "");
            assertFalse(client.healthCheck());
        }
    }
}
