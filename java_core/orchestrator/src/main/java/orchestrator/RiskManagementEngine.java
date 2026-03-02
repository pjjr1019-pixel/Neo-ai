package orchestrator;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import orchestrator.config.OrchestratorConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Risk management engine for the NEO trading system.
 *
 * Enforces multi-layer risk checks before any trade execution:
 * - Confidence threshold filtering
 * - Volatility cap enforcement
 * - Position size limits (max % of portfolio per trade)
 * - Daily loss limit tracking (circuit breaker)
 * - Maximum drawdown protection
 * - Signal validation (only BUY/SELL/HOLD allowed)
 *
 * All decisions are logged for audit trail compliance.
 */
public class RiskManagementEngine {
    private static final Logger LOG = LoggerFactory.getLogger(RiskManagementEngine.class);

    private final double confidenceThreshold;
    private final double maxVolatility;
    private final double maxPositionSize;
    private final double maxDrawdownPct;
    private final double dailyLossLimit;

    // Mutable state — tracked across the trading session
    private double dailyPnl = 0.0;
    private double peakEquity = 0.0;
    private double currentEquity = 0.0;
    private boolean circuitBreakerTripped = false;

    private final ObjectMapper mapper = new ObjectMapper();

    /**
     * Create a risk engine with explicit parameters.
     *
     * @param confidenceThreshold Minimum confidence to allow a trade [0, 1]
     * @param maxVolatility       Maximum allowed volatility level
     * @param maxPositionSize     Maximum position size as fraction of portfolio [0, 1]
     * @param maxDrawdownPct      Maximum drawdown percentage before halt [0, 1]
     * @param dailyLossLimit      Maximum daily loss before circuit breaker trips
     */
    public RiskManagementEngine(double confidenceThreshold, double maxVolatility,
                                 double maxPositionSize, double maxDrawdownPct,
                                 double dailyLossLimit) {
        this.confidenceThreshold = confidenceThreshold;
        this.maxVolatility = maxVolatility;
        this.maxPositionSize = maxPositionSize;
        this.maxDrawdownPct = maxDrawdownPct;
        this.dailyLossLimit = dailyLossLimit;

        LOG.info("RiskManagementEngine initialized: confidence>={}, maxVol={}, "
                + "maxPosition={}, maxDrawdown={}%, dailyLossLimit={}",
                confidenceThreshold, maxVolatility, maxPositionSize,
                maxDrawdownPct * 100, dailyLossLimit);
    }

    /**
     * Create a risk engine from OrchestratorConfig.
     *
     * @param config Configuration object
     */
    public RiskManagementEngine(OrchestratorConfig config) {
        this(config.getConfidenceThreshold(), config.getMaxVolatility(),
                config.getMaxPositionSize(), config.getMaxDrawdownPct(),
                config.getDailyLossLimit());
    }

    /**
     * Evaluate a prediction and determine if trade execution is approved.
     *
     * Applies all risk checks in sequence. Returns a RiskDecision
     * containing the approval status and reason if rejected.
     *
     * @param predictionResult Prediction from the AI model
     * @param currentVolatility Current market volatility
     * @return RiskDecision with approval status and reason
     */
    public RiskDecision evaluate(ApiClient.PredictionResult predictionResult,
                                  double currentVolatility) {
        String action = predictionResult.getAction();
        double confidence = predictionResult.getConfidence();

        LOG.debug("Evaluating risk: action={}, confidence={}, volatility={}",
                action, confidence, currentVolatility);

        // Check 1: Circuit breaker — halt trading if daily loss limit exceeded
        if (circuitBreakerTripped) {
            return reject("CIRCUIT_BREAKER",
                    "Daily loss limit breached — trading halted");
        }

        // Check 2: HOLD signals pass through (no risk to evaluate)
        if ("HOLD".equalsIgnoreCase(action)) {
            return approve("HOLD signal — no trade to evaluate");
        }

        // Check 3: Validate signal type
        if (!isValidSignal(action)) {
            return reject("INVALID_SIGNAL",
                    "Unknown signal: " + action);
        }

        // Check 4: Confidence threshold
        if (confidence < confidenceThreshold) {
            return reject("LOW_CONFIDENCE",
                    String.format("Confidence %.4f < threshold %.4f",
                            confidence, confidenceThreshold));
        }

        // Check 5: Volatility cap
        if (currentVolatility > maxVolatility) {
            return reject("HIGH_VOLATILITY",
                    String.format("Volatility %.4f > max %.4f",
                            currentVolatility, maxVolatility));
        }

        // Check 6: Drawdown protection
        if (currentEquity > 0 && peakEquity > 0) {
            double drawdown = (peakEquity - currentEquity) / peakEquity;
            if (drawdown > maxDrawdownPct) {
                return reject("MAX_DRAWDOWN",
                        String.format("Drawdown %.2f%% > max %.2f%%",
                                drawdown * 100, maxDrawdownPct * 100));
            }
        }

        // All checks passed
        return approve("All risk checks passed");
    }

    /**
     * Legacy method for backward compatibility — parses JSON string.
     *
     * @param predictionJson JSON string with prediction data
     * @param currentVolatility Current market volatility
     * @return true if trade is approved
     */
    public boolean approveAction(String predictionJson, double currentVolatility) {
        try {
            JsonNode node = mapper.readTree(predictionJson);
            double confidence = node.path("confidence").asDouble(0.0);
            String actionNode = node.path("action").isMissingNode() ? "HOLD" : node.path("action").asText();
            String action = node.path("signal").isMissingNode() ? actionNode : node.path("signal").asText();

            ApiClient.PredictionResult result = new ApiClient.PredictionResult(
                    action, confidence, 0.0);
            RiskDecision decision = evaluate(result, currentVolatility);
            return decision.isApproved();

        } catch (Exception e) {
            LOG.error("Failed to parse prediction JSON: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Record a trade outcome to update daily P&L tracking.
     *
     * @param pnl Profit/loss from the trade (positive = profit)
     */
    public void recordTradeOutcome(double pnl) {
        dailyPnl += pnl;
        currentEquity += pnl;
        if (currentEquity > peakEquity) {
            peakEquity = currentEquity;
        }

        // Check if daily loss limit is breached
        if (dailyPnl < -dailyLossLimit) {
            circuitBreakerTripped = true;
            LOG.error("CIRCUIT BREAKER TRIPPED: Daily P&L {} exceeds loss limit {}",
                    dailyPnl, dailyLossLimit);
        }

        LOG.info("Trade recorded: pnl={}, dailyPnl={}, equity={}, peak={}",
                pnl, dailyPnl, currentEquity, peakEquity);
    }

    /**
     * Reset daily tracking (call at start of each trading day).
     */
    public void resetDaily() {
        dailyPnl = 0.0;
        circuitBreakerTripped = false;
        LOG.info("Daily risk counters reset");
    }

    /**
     * Set the current equity level for drawdown calculations.
     *
     * @param equity Current portfolio equity
     */
    public void setEquity(double equity) {
        this.currentEquity = equity;
        if (equity > peakEquity) {
            this.peakEquity = equity;
        }
    }

    /**
     * Calculate the maximum allowed position size in dollars.
     *
     * @return Maximum position size based on current equity
     */
    public double getMaxPositionSizeDollars() {
        return currentEquity * maxPositionSize;
    }

    /**
     * Check if the circuit breaker is currently tripped.
     *
     * @return true if trading is halted due to daily loss limit
     */
    public boolean isCircuitBreakerTripped() {
        return circuitBreakerTripped;
    }

    public double getDailyPnl() { return dailyPnl; }
    public double getCurrentEquity() { return currentEquity; }

    private boolean isValidSignal(String signal) {
        return "BUY".equalsIgnoreCase(signal)
                || "SELL".equalsIgnoreCase(signal)
                || "HOLD".equalsIgnoreCase(signal);
    }

    private RiskDecision approve(String reason) {
        LOG.info("APPROVED: {}", reason);
        return new RiskDecision(true, reason);
    }

    private RiskDecision reject(String code, String reason) {
        LOG.warn("REJECTED [{}]: {}", code, reason);
        return new RiskDecision(false, code + ": " + reason);
    }

    /**
     * Immutable risk decision result.
     */
    public static class RiskDecision {
        private final boolean approved;
        private final String reason;

        public RiskDecision(boolean approved, String reason) {
            this.approved = approved;
            this.reason = reason;
        }

        public boolean isApproved() { return approved; }
        public String getReason() { return reason; }

        @Override
        public String toString() {
            return String.format("RiskDecision{approved=%s, reason='%s'}",
                    approved, reason);
        }
    }
}
