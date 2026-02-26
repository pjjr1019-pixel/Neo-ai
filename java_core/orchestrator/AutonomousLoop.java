package orchestrator;

import java.io.IOException;

public class AutonomousLoop {
    private final ApiClient apiClient;
    private final RiskManagementEngine riskEngine;

    public AutonomousLoop(ApiClient apiClient, RiskManagementEngine riskEngine) {
        this.apiClient = apiClient;
        this.riskEngine = riskEngine;
    }

    public void runLoop(String featuresJson, double currentVolatility) {
        try {
            String prediction = apiClient.predict(featuresJson);
            boolean approved = riskEngine.approveAction(prediction, currentVolatility);
            if (approved) {
                // Execute action (placeholder)
                System.out.println("Action executed: " + prediction);
            } else {
                System.out.println("Action rejected by risk engine.");
            }
            // Log all steps (placeholder)
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
