package orchestrator;

public class OrchestratorMain {
    public static void main(String[] args) {
        String baseUrl = "http://localhost:8000"; // FastAPI endpoint
        ApiClient apiClient = new ApiClient(baseUrl);
        RiskManagementEngine riskEngine = new RiskManagementEngine(0.7, 2.0); // Example thresholds
        AutonomousLoop loop = new AutonomousLoop(apiClient, riskEngine);
        Dashboard dashboard = new Dashboard();

        // Example features and volatility
        String featuresJson = "{\"feature1\": 1.0, \"feature2\": 2.0}";
        double currentVolatility = 1.5;

        // Run the autonomous loop
        loop.runLoop(featuresJson, currentVolatility);
        // Display dashboard info (placeholder)
        dashboard.displaySignal("BUY", 0.85);
        dashboard.displayModelVersion("v1.0.0");
    }
}
