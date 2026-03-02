package orchestrator.config;

import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Centralized configuration for the NEO Orchestrator.
 *
 * Loads settings from environment variables, system properties,
 * or a properties file (in that priority order). Validates all
 * required configuration on construction.
 */
public class OrchestratorConfig {
    private static final Logger LOG = LoggerFactory.getLogger(OrchestratorConfig.class);
    private static final String PROPS_FILE = "orchestrator.properties";

    // API
    private final String apiBaseUrl;
    private final int apiTimeoutMs;
    private final int apiMaxRetries;
    private final String apiAuthToken;

    // Risk management
    private final double confidenceThreshold;
    private final double maxVolatility;
    private final double maxPositionSize;
    private final double maxDrawdownPct;
    private final double dailyLossLimit;

    // Autonomous loop
    private final long loopIntervalMs;
    private final String tradingSymbol;

    // Database
    private final String dbUrl;
    private final String dbUser;
    private final String dbPassword;
    private final int dbPoolSize;

    // Redis
    private final String redisHost;
    private final int redisPort;
    private final String redisPassword;
    private final int redisTtlSeconds;

    // Dashboard
    private final int dashboardPort;

    private OrchestratorConfig(Builder builder) {
        this.apiBaseUrl = builder.apiBaseUrl;
        this.apiTimeoutMs = builder.apiTimeoutMs;
        this.apiMaxRetries = builder.apiMaxRetries;
        this.apiAuthToken = builder.apiAuthToken;
        this.confidenceThreshold = builder.confidenceThreshold;
        this.maxVolatility = builder.maxVolatility;
        this.maxPositionSize = builder.maxPositionSize;
        this.maxDrawdownPct = builder.maxDrawdownPct;
        this.dailyLossLimit = builder.dailyLossLimit;
        this.loopIntervalMs = builder.loopIntervalMs;
        this.tradingSymbol = builder.tradingSymbol;
        this.dbUrl = builder.dbUrl;
        this.dbUser = builder.dbUser;
        this.dbPassword = builder.dbPassword;
        this.dbPoolSize = builder.dbPoolSize;
        this.redisHost = builder.redisHost;
        this.redisPort = builder.redisPort;
        this.redisPassword = builder.redisPassword;
        this.redisTtlSeconds = builder.redisTtlSeconds;
        this.dashboardPort = builder.dashboardPort;
    }

    // ---- Getters ----

    public String getApiBaseUrl() { return apiBaseUrl; }
    public int getApiTimeoutMs() { return apiTimeoutMs; }
    public int getApiMaxRetries() { return apiMaxRetries; }
    public String getApiAuthToken() { return apiAuthToken; }
    public double getConfidenceThreshold() { return confidenceThreshold; }
    public double getMaxVolatility() { return maxVolatility; }
    public double getMaxPositionSize() { return maxPositionSize; }
    public double getMaxDrawdownPct() { return maxDrawdownPct; }
    public double getDailyLossLimit() { return dailyLossLimit; }
    public long getLoopIntervalMs() { return loopIntervalMs; }
    public String getTradingSymbol() { return tradingSymbol; }
    public String getDbUrl() { return dbUrl; }
    public String getDbUser() { return dbUser; }
    public String getDbPassword() { return dbPassword; }
    public int getDbPoolSize() { return dbPoolSize; }
    public String getRedisHost() { return redisHost; }
    public int getRedisPort() { return redisPort; }
    public String getRedisPassword() { return redisPassword; }
    public int getRedisTtlSeconds() { return redisTtlSeconds; }
    public int getDashboardPort() { return dashboardPort; }

    /**
     * Load configuration from environment variables and properties file.
     *
     * Environment variables take precedence over properties file values.
     * Properties file values take precedence over defaults.
     *
     * @return Configured OrchestratorConfig instance
     */
    public static OrchestratorConfig load() {
        Properties props = loadProperties();

        Builder builder = new Builder()
            .apiBaseUrl(resolve("NEO_API_BASE_URL", props, "http://localhost:8000"))
            .apiTimeoutMs(resolveInt("NEO_API_TIMEOUT_MS", props, 30000))
            .apiMaxRetries(resolveInt("NEO_API_MAX_RETRIES", props, 3))
            .apiAuthToken(resolve("NEO_API_AUTH_TOKEN", props, ""))
            .confidenceThreshold(resolveDouble("NEO_CONFIDENCE_THRESHOLD", props, 0.7))
            .maxVolatility(resolveDouble("NEO_MAX_VOLATILITY", props, 2.0))
            .maxPositionSize(resolveDouble("NEO_MAX_POSITION_SIZE", props, 0.1))
            .maxDrawdownPct(resolveDouble("NEO_MAX_DRAWDOWN_PCT", props, 0.05))
            .dailyLossLimit(resolveDouble("NEO_DAILY_LOSS_LIMIT", props, 500.0))
            .loopIntervalMs(resolveLong("NEO_LOOP_INTERVAL_MS", props, 60000))
            .tradingSymbol(resolve("NEO_TRADING_SYMBOL", props, "BTC/USD"))
            .dbUrl(resolve("NEO_DB_URL", props, ""))
            .dbUser(resolve("NEO_DB_USER", props, ""))
            .dbPassword(resolve("NEO_DB_PASSWORD", props, ""))
            .dbPoolSize(resolveInt("NEO_DB_POOL_SIZE", props, 5))
            .redisHost(resolve("NEO_REDIS_HOST", props, "localhost"))
            .redisPort(resolveInt("NEO_REDIS_PORT", props, 6379))
            .redisPassword(resolve("NEO_REDIS_PASSWORD", props, ""))
            .redisTtlSeconds(resolveInt("NEO_REDIS_TTL_SECONDS", props, 300))
            .dashboardPort(resolveInt("NEO_DASHBOARD_PORT", props, 9090));

        OrchestratorConfig config = builder.build();
        LOG.info("Configuration loaded: apiBaseUrl={}, symbol={}, loopInterval={}ms",
                config.apiBaseUrl, config.tradingSymbol, config.loopIntervalMs);
        return config;
    }

    private static Properties loadProperties() {
        Properties props = new Properties();
        try (InputStream is = OrchestratorConfig.class.getClassLoader()
                .getResourceAsStream(PROPS_FILE)) {
            if (is != null) {
                props.load(is);
                LOG.debug("Loaded {} properties from {}", props.size(), PROPS_FILE);
            }
        } catch (IOException e) {
            LOG.warn("Could not load {}: {}", PROPS_FILE, e.getMessage());
        }
        return props;
    }

    private static String resolve(String envKey, Properties props, String defaultVal) {
        String env = System.getenv(envKey);
        if (env != null && !env.isBlank()) return env;
        String prop = props.getProperty(envKey);
        if (prop != null && !prop.isBlank()) return prop;
        return defaultVal;
    }

    private static int resolveInt(String envKey, Properties props, int defaultVal) {
        String val = resolve(envKey, props, null);
        if (val == null) return defaultVal;
        try {
            return Integer.parseInt(val);
        } catch (NumberFormatException e) {
            LOG.warn("Invalid integer for {}: '{}', using default {}", envKey, val, defaultVal);
            return defaultVal;
        }
    }

    private static long resolveLong(String envKey, Properties props, long defaultVal) {
        String val = resolve(envKey, props, null);
        if (val == null) return defaultVal;
        try {
            return Long.parseLong(val);
        } catch (NumberFormatException e) {
            LOG.warn("Invalid long for {}: '{}', using default {}", envKey, val, defaultVal);
            return defaultVal;
        }
    }

    private static double resolveDouble(String envKey, Properties props, double defaultVal) {
        String val = resolve(envKey, props, null);
        if (val == null) return defaultVal;
        try {
            return Double.parseDouble(val);
        } catch (NumberFormatException e) {
            LOG.warn("Invalid double for {}: '{}', using default {}", envKey, val, defaultVal);
            return defaultVal;
        }
    }

    @Override
    public String toString() {
        return "OrchestratorConfig{"
            + "apiBaseUrl='" + apiBaseUrl + '\''
            + ", tradingSymbol='" + tradingSymbol + '\''
            + ", loopIntervalMs=" + loopIntervalMs
            + ", confidenceThreshold=" + confidenceThreshold
            + ", maxVolatility=" + maxVolatility
            + ", dashboardPort=" + dashboardPort
            + '}';
    }

    /**
     * Builder for OrchestratorConfig.
     */
    public static class Builder {
        private String apiBaseUrl = "http://localhost:8000";
        private int apiTimeoutMs = 30000;
        private int apiMaxRetries = 3;
        private String apiAuthToken = "";
        private double confidenceThreshold = 0.7;
        private double maxVolatility = 2.0;
        private double maxPositionSize = 0.1;
        private double maxDrawdownPct = 0.05;
        private double dailyLossLimit = 500.0;
        private long loopIntervalMs = 60000;
        private String tradingSymbol = "BTC/USD";
        private String dbUrl = "";
        private String dbUser = "";
        private String dbPassword = "";
        private int dbPoolSize = 5;
        private String redisHost = "localhost";
        private int redisPort = 6379;
        private String redisPassword = "";
        private int redisTtlSeconds = 300;
        private int dashboardPort = 9090;

        public Builder apiBaseUrl(String v) { this.apiBaseUrl = v; return this; }
        public Builder apiTimeoutMs(int v) { this.apiTimeoutMs = v; return this; }
        public Builder apiMaxRetries(int v) { this.apiMaxRetries = v; return this; }
        public Builder apiAuthToken(String v) { this.apiAuthToken = v; return this; }
        public Builder confidenceThreshold(double v) { this.confidenceThreshold = v; return this; }
        public Builder maxVolatility(double v) { this.maxVolatility = v; return this; }
        public Builder maxPositionSize(double v) { this.maxPositionSize = v; return this; }
        public Builder maxDrawdownPct(double v) { this.maxDrawdownPct = v; return this; }
        public Builder dailyLossLimit(double v) { this.dailyLossLimit = v; return this; }
        public Builder loopIntervalMs(long v) { this.loopIntervalMs = v; return this; }
        public Builder tradingSymbol(String v) { this.tradingSymbol = v; return this; }
        public Builder dbUrl(String v) { this.dbUrl = v; return this; }
        public Builder dbUser(String v) { this.dbUser = v; return this; }
        public Builder dbPassword(String v) { this.dbPassword = v; return this; }
        public Builder dbPoolSize(int v) { this.dbPoolSize = v; return this; }
        public Builder redisHost(String v) { this.redisHost = v; return this; }
        public Builder redisPort(int v) { this.redisPort = v; return this; }
        public Builder redisPassword(String v) { this.redisPassword = v; return this; }
        public Builder redisTtlSeconds(int v) { this.redisTtlSeconds = v; return this; }
        public Builder dashboardPort(int v) { this.dashboardPort = v; return this; }

        public OrchestratorConfig build() {
            Objects.requireNonNull(apiBaseUrl, "apiBaseUrl must not be null");
            if (apiTimeoutMs <= 0) throw new IllegalArgumentException("apiTimeoutMs must be > 0");
            if (confidenceThreshold < 0 || confidenceThreshold > 1) {
                throw new IllegalArgumentException("confidenceThreshold must be in [0, 1]");
            }
            return new OrchestratorConfig(this);
        }
    }
}
