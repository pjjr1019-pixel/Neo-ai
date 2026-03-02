package orchestrator;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import orchestrator.config.OrchestratorConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * REST API client for communicating with the Python FastAPI backend.
 *
 * Provides methods for feature computation, prediction, explanation,
 * metrics retrieval, health checks, and model feedback. Includes
 * configurable timeouts, retry logic with exponential backoff,
 * and optional JWT authentication.
 */
public class ApiClient {
    private static final Logger LOG = LoggerFactory.getLogger(ApiClient.class);

    private final String baseUrl;
    private final HttpClient client;
    private final ObjectMapper mapper;
    private final int maxRetries;
    private final String authToken;

    /**
     * Create an ApiClient with explicit parameters.
     *
     * @param baseUrl    FastAPI base URL (e.g., http://localhost:8000)
     * @param timeoutMs  HTTP request timeout in milliseconds
     * @param maxRetries Maximum retry attempts on failure
     * @param authToken  JWT/API key for authentication (empty string = no auth)
     */
    public ApiClient(String baseUrl, int timeoutMs, int maxRetries, String authToken) {
        this.baseUrl = baseUrl;
        this.maxRetries = maxRetries;
        this.authToken = authToken != null ? authToken : "";
        this.mapper = new ObjectMapper();
        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(timeoutMs))
                .build();
        LOG.info("ApiClient initialized: baseUrl={}, timeout={}ms, maxRetries={}",
                baseUrl, timeoutMs, maxRetries);
    }

    /**
     * Create an ApiClient from OrchestratorConfig.
     *
     * @param config Configuration object
     */
    public ApiClient(OrchestratorConfig config) {
        this(config.getApiBaseUrl(), config.getApiTimeoutMs(),
                config.getApiMaxRetries(), config.getApiAuthToken());
    }

    /**
     * Compute features from raw OHLCV price data.
     *
     * Calls the Python data pipeline via /compute-features to convert
     * OHLCV data into normalized feature vectors.
     *
     * @param symbol    Trading symbol (e.g., "BTC/USD")
     * @param priceData OHLCV data with keys: open, high, low, close, volume
     * @return Feature map with named feature keys and double values
     * @throws IOException          if API call fails after retries
     * @throws InterruptedException if request is interrupted
     */
    public Map<String, Double> computeFeatures(String symbol, Map<String, List<Double>> priceData)
            throws IOException, InterruptedException {

        ObjectNode payload = mapper.createObjectNode();
        payload.put("symbol", symbol);
        ObjectNode ohlcvNode = mapper.createObjectNode();
        for (Map.Entry<String, List<Double>> entry : priceData.entrySet()) {
            ohlcvNode.putPOJO(entry.getKey(), entry.getValue());
        }
        payload.set("ohlcv_data", ohlcvNode);

        String responseBody = postWithRetry("/compute-features", payload);
        JsonNode responseNode = mapper.readTree(responseBody);

        Map<String, Double> features = new HashMap<>();
        Iterator<String> fieldNames = responseNode.fieldNames();
        while (fieldNames.hasNext()) {
            String key = fieldNames.next();
            JsonNode value = responseNode.get(key);
            if (value.isNumber()) {
                features.put(key, value.asDouble());
            }
        }

        LOG.debug("Computed {} features for {}", features.size(), symbol);
        return features;
    }

    /**
     * Run model prediction with computed features.
     *
     * Calls /predict endpoint and returns the full prediction result
     * including action, confidence, and signal.
     *
     * @param features Feature map from computeFeatures()
     * @return Prediction result containing action, confidence, signal
     * @throws IOException          if API call fails after retries
     * @throws InterruptedException if request is interrupted
     */
    public PredictionResult predict(Map<String, Double> features)
            throws IOException, InterruptedException {

        ObjectNode payload = mapper.createObjectNode();
        ObjectNode featuresNode = mapper.createObjectNode();
        for (Map.Entry<String, Double> entry : features.entrySet()) {
            featuresNode.put(entry.getKey(), entry.getValue());
        }
        payload.set("features", featuresNode);

        String responseBody = postWithRetry("/predict", payload);
        JsonNode responseNode = mapper.readTree(responseBody);

        JsonNode signalNode = responseNode.path("signal");
        String action = (signalNode.isMissingNode() || signalNode.isNull()) ? "HOLD" : signalNode.asText();
        double confidence = responseNode.path("confidence").asDouble(0.0);
        double prediction = responseNode.path("prediction").asDouble(0.0);

        PredictionResult result = new PredictionResult(action, confidence, prediction);
        LOG.info("Prediction: action={}, confidence={:.4f}", action, confidence);
        return result;
    }

    /**
     * Send feedback for online learning (model retraining).
     *
     * Calls /learn endpoint to provide actual outcome data
     * so the model can improve over time.
     *
     * @param features Feature map that was used for prediction
     * @param actualOutcome The actual return/outcome observed
     * @return true if feedback was accepted
     * @throws IOException          if API call fails after retries
     * @throws InterruptedException if request is interrupted
     */
    public boolean sendFeedback(Map<String, Double> features, double actualOutcome)
            throws IOException, InterruptedException {

        ObjectNode payload = mapper.createObjectNode();
        ObjectNode featuresNode = mapper.createObjectNode();
        for (Map.Entry<String, Double> entry : features.entrySet()) {
            featuresNode.put(entry.getKey(), entry.getValue());
        }
        payload.set("features", featuresNode);
        payload.put("actual", actualOutcome);

        String responseBody = postWithRetry("/learn", payload);
        LOG.info("Feedback sent: actualOutcome={}", actualOutcome);
        return responseBody != null;
    }

    /**
     * Get model explanations (feature importances).
     *
     * @return JSON string with feature importance scores
     * @throws IOException          if API call fails after retries
     * @throws InterruptedException if request is interrupted
     */
    public String explain() throws IOException, InterruptedException {
        String response = getWithRetry("/explain");
        LOG.debug("Explanation retrieved");
        return response;
    }

    /**
     * Get system metrics.
     *
     * @return JSON string with request counts and system health
     * @throws IOException          if API call fails after retries
     * @throws InterruptedException if request is interrupted
     */
    public String getMetrics() throws IOException, InterruptedException {
        return getWithRetry("/metrics");
    }

    /**
     * Health check — verify the Python backend is running.
     *
     * @return true if backend responds with HTTP 200
     */
    public boolean healthCheck() {
        try {
            HttpRequest request = buildRequest("/health", "GET", null);
            HttpResponse<String> response = client.send(request,
                    HttpResponse.BodyHandlers.ofString());
            boolean healthy = response.statusCode() == 200;
            LOG.debug("Health check: {}", healthy ? "OK" : "FAILED");
            return healthy;
        } catch (Exception e) {
            LOG.warn("Health check failed: {}", e.getMessage());
            return false;
        }
    }

    // ---- Internal HTTP helpers with retry logic ----

    private String postWithRetry(String path, ObjectNode payload)
            throws IOException, InterruptedException {
        String body = mapper.writeValueAsString(payload);
        return executeWithRetry(path, "POST", body);
    }

    private String getWithRetry(String path)
            throws IOException, InterruptedException {
        return executeWithRetry(path, "GET", null);
    }

    private String executeWithRetry(String path, String method, String body)
            throws IOException, InterruptedException {
        IOException lastException = null;

        for (int attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                HttpRequest request = buildRequest(path, method, body);
                HttpResponse<String> response = client.send(request,
                        HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() >= 200 && response.statusCode() < 300) {
                    return response.body();
                }

                lastException = new IOException(String.format(
                        "%s %s failed (HTTP %d): %s",
                        method, path, response.statusCode(), response.body()));

                // Don't retry client errors (4xx) — they won't succeed on retry
                if (response.statusCode() >= 400 && response.statusCode() < 500) {
                    throw lastException;
                }

            } catch (IOException e) {
                lastException = e;
                if (attempt < maxRetries) {
                    long backoffMs = (long) Math.pow(2, attempt) * 1000;
                    LOG.warn("Attempt {}/{} failed for {} {}: {}. Retrying in {}ms",
                            attempt + 1, maxRetries + 1, method, path,
                            e.getMessage(), backoffMs);
                    Thread.sleep(backoffMs);
                }
            }
        }

        LOG.error("All {} attempts failed for {} {}", maxRetries + 1, method, path);
        throw lastException;
    }

    private HttpRequest buildRequest(String path, String method, String body) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(30));

        // Add auth header if token is configured
        if (!authToken.isEmpty()) {
            builder.header("Authorization", "Bearer " + authToken);
        }

        return switch (method) {
            case "POST" -> builder.POST(HttpRequest.BodyPublishers.ofString(
                    body != null ? body : "")).build();
            case "PUT" -> builder.PUT(HttpRequest.BodyPublishers.ofString(
                    body != null ? body : "")).build();
            default -> builder.GET().build();
        };
    }

    /**
     * Structured prediction result from the AI model.
     */
    public static class PredictionResult {
        private final String action;
        private final double confidence;
        private final double prediction;

        public PredictionResult(String action, double confidence, double prediction) {
            this.action = action;
            this.confidence = confidence;
            this.prediction = prediction;
        }

        public String getAction() { return action; }
        public double getConfidence() { return confidence; }
        public double getPrediction() { return prediction; }

        /**
         * Convert prediction result to JSON string for logging/storage.
         *
         * @return JSON representation of this prediction
         */
        public String toJson() {
            return String.format(
                    "{\"action\":\"%s\",\"confidence\":%.6f,\"prediction\":%.6f}",
                    action, confidence, prediction);
        }

        @Override
        public String toString() {
            return String.format("PredictionResult{action='%s', confidence=%.4f, prediction=%.4f}",
                    action, confidence, prediction);
        }
    }
}
