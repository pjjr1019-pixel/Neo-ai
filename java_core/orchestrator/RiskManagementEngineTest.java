package orchestrator;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class RiskManagementEngineTest {
    @Test
    void testApproveActionConfidence() {
        RiskManagementEngine engine = new RiskManagementEngine(0.7, 2.0);
        String prediction = "{\"confidence\": 0.8}";
        assertTrue(engine.approveAction(prediction, 1.0));
        assertFalse(engine.approveAction(prediction, 3.0));
        String lowConfidence = "{\"confidence\": 0.5}";
        assertFalse(engine.approveAction(lowConfidence, 1.0));
    }
}
