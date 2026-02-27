package orchestrator;

import org.json.JSONObject;

public class RiskManagementEngine {
    private double confidenceThreshold;
    private double maxVolatility;

    public RiskManagementEngine(double confidenceThreshold, double maxVolatility) {
        this.confidenceThreshold = confidenceThreshold;
        this.maxVolatility = maxVolatility;
    }

    public boolean approveAction(String predictionJson, double currentVolatility) {
        try {
            JSONObject obj = new JSONObject(predictionJson);
            double confidence = obj.optDouble("confidence", 0.0);
            if (confidence < confidenceThreshold) {
                return false;
            }
            if (currentVolatility > maxVolatility) {
                return false;
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    // Add more risk checks as needed
}
