package orchestrator;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class OrchestratorMainTest {
    @Test
    void testMainRuns() {
        assertDoesNotThrow(() -> OrchestratorMain.main(new String[]{}));
    }
}
