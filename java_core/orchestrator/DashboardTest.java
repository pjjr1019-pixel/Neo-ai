package orchestrator;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DashboardTest {
    @Test
    void testDisplaySignal() {
        Dashboard dashboard = new Dashboard();
        assertDoesNotThrow(() -> dashboard.displaySignal("BUY", 0.9));
    }

    @Test
    void testDisplayModelVersion() {
        Dashboard dashboard = new Dashboard();
        assertDoesNotThrow(() -> dashboard.displayModelVersion("v1.0.0"));
    }
}
