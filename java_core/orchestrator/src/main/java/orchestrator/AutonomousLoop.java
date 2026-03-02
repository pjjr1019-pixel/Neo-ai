package orchestrator;

import orchestrator.config.OrchestratorConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Autonomous trading loop for the NEO orchestrator.
 *
 * Implements the full trading pipeline cycle:
 * 1. Fetch market data (via DataFeedClient)
 * 2. Compute features (via ApiClient → Python backend)
 * 3. Get AI prediction (via ApiClient → /predict)
 * 4. Evaluate risk (via RiskManagementEngine)
 * 5. Execute trade (paper or live)
 * 6. Log all steps (via DatabaseLogger)
 * 7. Send feedback for online learning (via ApiClient → /learn)
 *
 * Runs on a configurable schedule with graceful shutdown support.
 */
public class AutonomousLoop {
    private static final Logger LOG = LoggerFactory.getLogger(AutonomousLoop.class);

    private final ApiClient apiClient;
    private final RiskManagementEngine riskEngine;
    private final DataFeedClient dataFeed;
    private final DatabaseLogger dbLogger;
    private final RedisCache cache;
    private final String symbol;
    private final long intervalMs;

    private final ScheduledExecutorService scheduler;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicLong cycleCount = new AtomicLong(0);
    private ScheduledFuture<?> scheduledTask;

    // Trading state
    private String lastAction = "HOLD";
    private double lastPrediction = 0.0;
    private double lastConfidence = 0.0;
    private Map<String, Double> lastFeatures = new HashMap<>();
    private final List<TradeRecord> tradeHistory = new ArrayList<>();

    /**
     * Create an autonomous loop with all dependencies.
     *
     * @param apiClient  API client for Python backend communication
     * @param riskEngine Risk management engine for trade approval
     * @param dataFeed   Market data feed for price data
     * @param dbLogger   Database logger for audit trail
     * @param cache      Redis cache for state/features
     * @param symbol     Trading symbol (e.g., "BTC/USD")
     * @param intervalMs Loop interval in milliseconds
     */
    public AutonomousLoop(ApiClient apiClient, RiskManagementEngine riskEngine,
                          DataFeedClient dataFeed, DatabaseLogger dbLogger,
                          RedisCache cache, String symbol, long intervalMs) {
        this.apiClient = apiClient;
        this.riskEngine = riskEngine;
        this.dataFeed = dataFeed;
        this.dbLogger = dbLogger;
        this.cache = cache;
        this.symbol = symbol;
        this.intervalMs = intervalMs;
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "neo-trading-loop");
            t.setDaemon(true);
            return t;
        });

        LOG.info("AutonomousLoop created: symbol={}, interval={}ms", symbol, intervalMs);
    }

    /**
     * Create an autonomous loop from OrchestratorConfig.
     *
     * @param config   Configuration
     * @param apiClient API client
     * @param riskEngine Risk engine
     * @param dataFeed Data feed
     * @param dbLogger Database logger
     * @param cache    Redis cache
     */
    public AutonomousLoop(OrchestratorConfig config, ApiClient apiClient,
                          RiskManagementEngine riskEngine, DataFeedClient dataFeed,
                          DatabaseLogger dbLogger, RedisCache cache) {
        this(apiClient, riskEngine, dataFeed, dbLogger, cache,
                config.getTradingSymbol(), config.getLoopIntervalMs());
    }

    /**
     * Start the autonomous trading loop.
     *
     * The loop runs on a fixed schedule. Each cycle executes the
     * full trading pipeline. A health check is performed before
     * starting.
     */
    public void start() {
        if (running.compareAndSet(false, true)) {
            LOG.info("Starting autonomous trading loop for {} (interval={}ms)",
                    symbol, intervalMs);

            // Check backend health before starting
            if (!apiClient.healthCheck()) {
                LOG.warn("Python backend health check failed — starting loop anyway "
                        + "(will retry on each cycle)");
            }

            scheduledTask = scheduler.scheduleAtFixedRate(
                    this::executeCycle,
                    0, intervalMs, TimeUnit.MILLISECONDS
            );
        } else {
            LOG.warn("Loop is already running");
        }
    }

    /**
     * Stop the autonomous trading loop gracefully.
     *
     * Waits for the current cycle to finish before stopping.
     */
    public void stop() {
        if (running.compareAndSet(true, false)) {
            LOG.info("Stopping autonomous trading loop...");
            if (scheduledTask != null) {
                scheduledTask.cancel(false);
            }
            scheduler.shutdown();
            try {
                if (!scheduler.awaitTermination(30, TimeUnit.SECONDS)) {
                    scheduler.shutdownNow();
                    LOG.warn("Forced shutdown after timeout");
                }
            } catch (InterruptedException e) {
                scheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
            LOG.info("Trading loop stopped. Total cycles: {}, trades: {}",
                    cycleCount.get(), tradeHistory.size());
        }
    }

    /**
     * Execute a single trading cycle.
     *
     * This is the core pipeline: fetch → features → predict → risk → execute → log → feedback.
     */
    void executeCycle() {
        long cycleNum = cycleCount.incrementAndGet();
        Instant cycleStart = Instant.now();
        LOG.info("=== Cycle {} started at {} ===", cycleNum, cycleStart);

        try {
            // Step 1: Fetch latest market data
            Map<String, List<Double>> ohlcvData = dataFeed.fetchOHLCV(symbol);
            if (ohlcvData == null || ohlcvData.isEmpty()) {
                LOG.warn("No market data available for {} — skipping cycle", symbol);
                logStep(cycleNum, "SKIP", "No market data available");
                return;
            }
            double currentVolatility = dataFeed.getCurrentVolatility(symbol);
            LOG.debug("Fetched {} bars for {}, volatility={:.4f}",
                    ohlcvData.getOrDefault("close", List.of()).size(),
                    symbol, currentVolatility);

            // Step 2: Compute features via Python backend
            Map<String, Double> features = apiClient.computeFeatures(symbol, ohlcvData);
            if (features.isEmpty()) {
                LOG.warn("Feature computation returned empty — skipping cycle");
                logStep(cycleNum, "SKIP", "Empty features");
                return;
            }
            lastFeatures = features;

            // Cache features in Redis
            cache.setJson("features:" + symbol, features.toString());

            // Step 3: Get AI prediction
            ApiClient.PredictionResult prediction = apiClient.predict(features);
            lastAction = prediction.getAction();
            lastConfidence = prediction.getConfidence();
            lastPrediction = prediction.getPrediction();

            // Cache prediction
            cache.setJson("prediction:" + symbol, prediction.toJson());

            // Step 4: Risk evaluation
            RiskManagementEngine.RiskDecision decision =
                    riskEngine.evaluate(prediction, currentVolatility);

            // Step 5: Execute or skip based on risk decision
            if (decision.isApproved() && !"HOLD".equalsIgnoreCase(prediction.getAction())) {
                executeTrade(cycleNum, prediction, features);
            } else {
                LOG.info("No trade: action={}, approved={}, reason={}",
                        prediction.getAction(), decision.isApproved(), decision.getReason());
                logStep(cycleNum, "NO_TRADE", decision.getReason());
            }

            // Step 6: Log cycle summary
            dbLogger.logAction("CYCLE_COMPLETE",
                    String.format("cycle=%d, action=%s, confidence=%.4f, approved=%s",
                            cycleNum, prediction.getAction(), prediction.getConfidence(),
                            decision.isApproved()));

        } catch (Exception e) {
            LOG.error("Cycle {} failed: {}", cycleNum, e.getMessage(), e);
            logStep(cycleNum, "ERROR", e.getMessage());
        }

        long elapsed = Instant.now().toEpochMilli() - cycleStart.toEpochMilli();
        LOG.info("=== Cycle {} completed in {}ms ===", cycleNum, elapsed);
    }

    /**
     * Execute a paper trade and record the outcome.
     */
    private void executeTrade(long cycleNum, ApiClient.PredictionResult prediction,
                              Map<String, Double> features) {
        TradeRecord trade = new TradeRecord(
                cycleNum, Instant.now(), symbol,
                prediction.getAction(), prediction.getConfidence(),
                riskEngine.getMaxPositionSizeDollars()
        );
        tradeHistory.add(trade);

        LOG.info("TRADE EXECUTED: {} {} (confidence={:.4f}, size={:.2f})",
                prediction.getAction(), symbol,
                prediction.getConfidence(), trade.positionSize);

        dbLogger.logAction("TRADE_EXECUTED", trade.toJson());

        // Step 7: Async feedback for online learning (non-blocking)
        try {
            apiClient.sendFeedback(features, prediction.getPrediction());
        } catch (Exception e) {
            LOG.warn("Feedback send failed (non-critical): {}", e.getMessage());
        }
    }

    private void logStep(long cycleNum, String action, String details) {
        try {
            dbLogger.logAction(action, String.format("cycle=%d: %s", cycleNum, details));
        } catch (Exception e) {
            LOG.warn("Failed to log step: {}", e.getMessage());
        }
    }

    // ---- Status accessors ----

    public boolean isRunning() { return running.get(); }
    public long getCycleCount() { return cycleCount.get(); }
    public String getLastAction() { return lastAction; }
    public double getLastConfidence() { return lastConfidence; }
    public List<TradeRecord> getTradeHistory() { return List.copyOf(tradeHistory); }

    /**
     * Get a status summary for dashboard display.
     *
     * @return Map of status key-value pairs
     */
    public Map<String, Object> getStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("running", running.get());
        status.put("symbol", symbol);
        status.put("cycleCount", cycleCount.get());
        status.put("lastAction", lastAction);
        status.put("lastConfidence", lastConfidence);
        status.put("lastPrediction", lastPrediction);
        status.put("totalTrades", tradeHistory.size());
        status.put("dailyPnl", riskEngine.getDailyPnl());
        status.put("circuitBreaker", riskEngine.isCircuitBreakerTripped());
        return status;
    }

    /**
     * Trade execution record for audit trail.
     */
    public static class TradeRecord {
        final long cycleNum;
        final Instant timestamp;
        final String symbol;
        final String action;
        final double confidence;
        final double positionSize;

        TradeRecord(long cycleNum, Instant timestamp, String symbol,
                    String action, double confidence, double positionSize) {
            this.cycleNum = cycleNum;
            this.timestamp = timestamp;
            this.symbol = symbol;
            this.action = action;
            this.confidence = confidence;
            this.positionSize = positionSize;
        }

        public String toJson() {
            return String.format(
                    "{\"cycle\":%d,\"timestamp\":\"%s\",\"symbol\":\"%s\","
                    + "\"action\":\"%s\",\"confidence\":%.6f,\"positionSize\":%.2f}",
                    cycleNum, timestamp, symbol, action, confidence, positionSize);
        }

        public long getCycleNum() { return cycleNum; }
        public Instant getTimestamp() { return timestamp; }
        public String getSymbol() { return symbol; }
        public String getAction() { return action; }
        public double getConfidence() { return confidence; }
        public double getPositionSize() { return positionSize; }
    }
}
