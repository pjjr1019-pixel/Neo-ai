package orchestrator;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ApiClient for Python-Java Orchestrator Integration.
 *
 * Connects to FastAPI backend endpoints for ML model predictions,
 * feature computation, and metrics queries.
 */
public class ApiClient {
    private final String baseUrl;
    private final HttpClient client;
    private final ObjectMapper mapper;

    public ApiClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.client = HttpClient.newHttpClient();
        this.mapper = new ObjectMapper();
    }

    /**
     * Compute features from raw OHLCV price data.
     *
     * This calls the Python data pipeline via REST endpoint to convert
     * OHLCV data into normalized feature vectors (f0-f9).
     *
     * @param symbol Trading symbol (e.g., 'BTC/USD')
     * @param priceData OHLCV price data with open, high, low, close,
     *            volume
     * @return Feature dict with f0-f9 keys
     * @throws IOException if API call fails
     * @throws InterruptedException if request is interrupted
     */
    public Map<String, Double> computeFeatures(
        String symbol, Map<String, List<Double>> priceData
    ) throws IOException, InterruptedException {
        ObjectNode payload = mapper.createObjectNode();
        payload.put("symbol", symbol);

        ObjectNode ohlcvNode = mapper.createObjectNode();
        for (Map.Entry<String, List<Double>> entry :
            priceData.entrySet()) {
            ohlcvNode.putPOJO(
                entry.getKey(),
                entry.getValue()
            );
        }
        payload.set("ohlcv_data", ohlcvNode);

        String requestBody = mapper.writeValueAsString(payload);
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/compute-features"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new IOException(
                "Feature computation failed: " + response.body()
            );
        }

        JsonNode responseNode = mapper.readTree(response.body());
        Map<String, Double> features = new HashMap<>();
        for (String key : new String[]{"f0", "f1", "f2", "f3", "f4",
            "f5", "f6", "f7", "f8", "f9"}) {
            if (responseNode.has(key)) {
                features.put(key, responseNode.get(key)
                    .asDouble());
            }
        }
        return features;
    }

    /**
     * Predict with computed features.
     *
     * Calls ML model via /predict endpoint with normalized features.
     * Returns prediction, confidence, and trading signal.
     *
     * @param features Feature dict from computeFeatures()
     * @return JSON response with prediction, confidence, signal
     * @throws IOException if API call fails
     * @throws InterruptedException if request is interrupted
     */
    public String predict(Map<String, Double> features)
        throws IOException, InterruptedException {
        ObjectNode payload = mapper.createObjectNode();
        ObjectNode featuresNode = mapper.createObjectNode();
        for (Map.Entry<String, Double> entry : features.entrySet()) {
            featuresNode.put(entry.getKey(), entry.getValue());
        }
        payload.set("features", featuresNode);

        String requestBody = mapper.writeValueAsString(payload);
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/predict"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new IOException(
                "Prediction failed: " + response.body()
            );
        }

        return response.body();
    }

    /**
     * Get model explanations (feature importances).
     *
     * Returns feature importance scores from ensemble model.
     *
     * @return JSON response with feature importances
     * @throws IOException if API call fails
     * @throws InterruptedException if request is interrupted
     */
    public String explain() throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/explain"))
            .header("Content-Type", "application/json")
            .GET()
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new IOException(
                "Explanation failed: " + response.body()
            );
        }

        return response.body();
    }

    /**
     * Get system metrics.
     *
     * Returns request counts and system health information.
     *
     * @return JSON response with metrics
     * @throws IOException if API call fails
     * @throws InterruptedException if request is interrupted
     */
    public String getMetrics() throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/metrics"))
            .header("Content-Type", "application/json")
            .GET()
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new IOException(
                "Metrics query failed: " + response.body()
            );
        }

        return response.body();
    }

    /**
     * Health check endpoint.
     *
     * Verify Python backend is running.
     *
     * @return True if backend is responsive
     * @throws IOException if API call fails
     * @throws InterruptedException if request is interrupted
     */
    public boolean healthCheck()
        throws IOException, InterruptedException {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/"))
                .GET()
                .build();

            HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
            );

            return response.statusCode() == 200;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }
}
