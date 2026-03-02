package orchestrator;

import orchestrator.config.OrchestratorConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main entry point for the NEO Java Orchestrator.
 *
 * Initializes all components (API client, risk engine, data feed,
 * database logger, Redis cache, dashboard) from configuration,
 * starts the autonomous trading loop, and handles graceful shutdown.
 *
 * Configuration is loaded from environment variables and/or a
 * properties file (see OrchestratorConfig).
 *
 * Usage:
 *   java -jar orchestrator.jar
 *
 * Environment variables:
 *   NEO_API_BASE_URL       — Python FastAPI backend URL
 *   NEO_CONFIDENCE_THRESHOLD — Min confidence for trades
 *   NEO_MAX_VOLATILITY     — Max volatility cap
 *   NEO_TRADING_SYMBOL     — Symbol to trade (e.g., BTC/USD)
 *   NEO_LOOP_INTERVAL_MS   — Trading cycle interval
 *   NEO_DB_URL             — PostgreSQL JDBC URL
 *   NEO_DB_USER            — Database username
 *   NEO_DB_PASSWORD        — Database password
 *   NEO_REDIS_HOST         — Redis hostname
 *   NEO_DASHBOARD_PORT     — Dashboard HTTP port
 */
public class OrchestratorMain {
    private static final Logger LOG = LoggerFactory.getLogger(OrchestratorMain.class);

    private OrchestratorConfig config;
    private ApiClient apiClient;
    private RiskManagementEngine riskEngine;
    private DataFeedClient dataFeed;
    private DatabaseLogger dbLogger;
    private RedisCache cache;
    private AuthManager authManager;
    private AutonomousLoop loop;
    private Dashboard dashboard;

    /**
     * Application entry point.
     *
     * @param args Command-line arguments (unused — config via env vars)
     */
    public static void main(String[] args) {
        LOG.info("============================================");
        LOG.info("  NEO Orchestrator v1.0.0 starting...");
        LOG.info("============================================");

        OrchestratorMain orchestrator = new OrchestratorMain();
        try {
            orchestrator.initialize();
            orchestrator.start();
        } catch (Exception e) {
            LOG.error("Fatal error during startup: {}", e.getMessage(), e);
            System.exit(1);
        }
    }

    /**
     * Initialize all orchestrator components.
     */
    public void initialize() {
        LOG.info("Loading configuration...");
        config = OrchestratorConfig.load();

        LOG.info("Initializing components...");

        // Authentication
        authManager = new AuthManager(config.getApiAuthToken(), "");

        // API client for Python backend
        apiClient = new ApiClient(config);

        // Risk management engine
        riskEngine = new RiskManagementEngine(config);
        riskEngine.setEquity(10000.0); // Default starting equity

        // Market data feed (simulated for now)
        dataFeed = new DataFeedClient("simulated", 100.0, 0.02, 50);

        // Database logger (graceful degradation if DB unavailable)
        dbLogger = new DatabaseLogger(config);

        // Redis cache (graceful degradation if Redis unavailable)
        cache = new RedisCache(
                config.getRedisHost(),
                config.getRedisPort(),
                config.getRedisPassword(),
                config.getRedisTtlSeconds()
        );

        // Autonomous trading loop
        loop = new AutonomousLoop(config, apiClient, riskEngine,
                dataFeed, dbLogger, cache);

        // Dashboard
        dashboard = new Dashboard(config.getDashboardPort(), loop,
                riskEngine, apiClient, dbLogger, cache);

        LOG.info("All components initialized successfully");
        logComponentStatus();
    }

    /**
     * Start the orchestrator (loop + dashboard).
     */
    public void start() {
        // Register shutdown hook for graceful cleanup
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            LOG.info("Shutdown signal received — cleaning up...");
            stop();
        }, "neo-shutdown"));

        // Start dashboard
        try {
            dashboard.start();
        } catch (Exception e) {
            LOG.warn("Dashboard failed to start: {}", e.getMessage());
        }

        // Backend health check
        if (apiClient.healthCheck()) {
            LOG.info("Python backend is healthy — starting trading loop");
        } else {
            LOG.warn("Python backend is NOT reachable — "
                    + "loop will start but predictions will fail until backend is up");
        }

        // Start trading loop
        loop.start();

        LOG.info("============================================");
        LOG.info("  NEO Orchestrator is RUNNING");
        LOG.info("  Symbol: {}", config.getTradingSymbol());
        LOG.info("  Dashboard: http://localhost:{}", config.getDashboardPort());
        LOG.info("============================================");

        // Keep main thread alive
        try {
            Thread.currentThread().join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            LOG.info("Main thread interrupted — shutting down");
        }
    }

    /**
     * Stop all components gracefully.
     */
    public void stop() {
        LOG.info("Stopping orchestrator components...");

        if (loop != null) {
            loop.stop();
        }
        if (dashboard != null) {
            dashboard.stop();
        }
        if (cache != null) {
            cache.close();
        }
        if (dbLogger != null) {
            dbLogger.close();
        }

        LOG.info("All components stopped");
    }

    /**
     * Log the status of all components.
     */
    private void logComponentStatus() {
        LOG.info("Component status:");
        LOG.info("  API Client:    {} ({})", "initialized", config.getApiBaseUrl());
        LOG.info("  Risk Engine:   {} (confidence>={}, maxVol={})",
                "initialized", config.getConfidenceThreshold(), config.getMaxVolatility());
        LOG.info("  Data Feed:     {}", dataFeed.isConnected() ? "connected" : "disconnected");
        LOG.info("  Database:      {}", dbLogger.isAvailable() ? "connected" : "disabled");
        LOG.info("  Redis Cache:   {}", cache.isAvailable() ? "connected" : "disabled");
        LOG.info("  Auth:          {}", authManager.isAuthEnabled() ? "enabled" : "disabled");
    }

    // Accessors for testing
    OrchestratorConfig getConfig() { return config; }
    ApiClient getApiClient() { return apiClient; }
    RiskManagementEngine getRiskEngine() { return riskEngine; }
    AutonomousLoop getLoop() { return loop; }
    Dashboard getDashboard() { return dashboard; }
}
