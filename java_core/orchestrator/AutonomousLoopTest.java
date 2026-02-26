package orchestrator;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AutonomousLoopTest {
    @Test
    void testRunLoopNoException() {
        ApiClient apiClient = new ApiClient("http://localhost:8000");
        RiskManagementEngine riskEngine = new RiskManagementEngine(0.7, 2.0);
        AutonomousLoop loop = new AutonomousLoop(apiClient, riskEngine);
        assertDoesNotThrow(() -> {
            // Placeholder: would mock ApiClient and RiskManagementEngine in real test
            loop.runLoop("{\"feature1\":1.0}", 1.0);
        });
    }
}
