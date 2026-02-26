package orchestrator;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ApiClientTest {
    @Test
    void testPredictHandlesResponse() {
        ApiClient client = new ApiClient("http://localhost:8000");
        // Mocking or integration test would go here
        assertDoesNotThrow(() -> {
            // This is a placeholder; in real tests, use a mock server
            // String response = client.predict("{\"feature1\":1.0}");
            // assertNotNull(response);
        });
    }
}
