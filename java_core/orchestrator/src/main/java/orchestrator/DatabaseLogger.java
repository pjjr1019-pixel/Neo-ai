package orchestrator;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import orchestrator.config.OrchestratorConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Database logger for the NEO orchestrator audit trail.
 *
 * Uses HikariCP connection pooling for efficient database access.
 * Logs all trading actions, risk decisions, cycle events, and errors
 * to the orchestrator_logs table. Supports table auto-creation and
 * log querying for dashboard display.
 *
 * Thread-safe: connection pool handles concurrency.
 */
public class DatabaseLogger {
    private static final Logger LOG = LoggerFactory.getLogger(DatabaseLogger.class);

    private HikariDataSource dataSource;
    private final boolean available;

    /**
     * Create a database logger with HikariCP connection pooling.
     *
     * @param url      JDBC URL (e.g., jdbc:postgresql://localhost:5432/neoai_db)
     * @param user     Database username
     * @param password Database password
     * @param poolSize Maximum connection pool size
     */
    public DatabaseLogger(String url, String user, String password, int poolSize) {
        boolean connected = false;

        if (url == null || url.isEmpty()) {
            LOG.warn("No database URL configured — database logging disabled");
            this.available = false;
            return;
        }

        try {
            HikariConfig config = new HikariConfig();
            config.setJdbcUrl(url);
            config.setUsername(user);
            config.setPassword(password);
            config.setMaximumPoolSize(poolSize);
            config.setMinimumIdle(1);
            config.setConnectionTimeout(5000);
            config.setIdleTimeout(300000);
            config.setMaxLifetime(600000);
            config.setPoolName("neo-db-pool");

            this.dataSource = new HikariDataSource(config);

            // Auto-create table if it doesn't exist
            ensureTable();
            connected = true;
            LOG.info("DatabaseLogger connected: url={}, poolSize={}", url, poolSize);

        } catch (Exception e) {
            LOG.warn("Database not available — logging disabled: {}", e.getMessage());
        }

        this.available = connected;
    }

    /**
     * Create a database logger from OrchestratorConfig.
     *
     * @param config Configuration object
     */
    public DatabaseLogger(OrchestratorConfig config) {
        this(config.getDbUrl(), config.getDbUser(),
                config.getDbPassword(), config.getDbPoolSize());
    }

    /**
     * Create a no-op database logger (when no DB is configured).
     */
    public DatabaseLogger() {
        this("", "", "", 1);
    }

    /**
     * Log a trading action with details.
     *
     * @param action  Action type (e.g., "TRADE_EXECUTED", "CYCLE_COMPLETE", "ERROR")
     * @param details Detailed information about the action (JSON or free text)
     */
    public void logAction(String action, String details) {
        if (!available) {
            LOG.info("[DB-OFFLINE] {} — {}", action, details);
            return;
        }

        String sql = "INSERT INTO orchestrator_logs (action, details, created_at) "
                + "VALUES (?, ?, ?)";
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, action);
            pstmt.setString(2, details);
            pstmt.setTimestamp(3, Timestamp.from(Instant.now()));
            pstmt.executeUpdate();
            LOG.debug("Logged: {} — {}", action, details);
        } catch (SQLException e) {
            LOG.error("Failed to log action '{}': {}", action, e.getMessage());
        }
    }

    /**
     * Query recent log entries.
     *
     * @param limit Maximum number of entries to return
     * @return List of log entry strings (newest first)
     */
    public List<String> getRecentLogs(int limit) {
        List<String> logs = new ArrayList<>();
        if (!available) return logs;

        String sql = "SELECT action, details, created_at FROM orchestrator_logs "
                + "ORDER BY created_at DESC LIMIT ?";
        try (Connection conn = dataSource.getConnection();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, limit);
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    logs.add(String.format("[%s] %s — %s",
                            rs.getTimestamp("created_at"),
                            rs.getString("action"),
                            rs.getString("details")));
                }
            }
        } catch (SQLException e) {
            LOG.error("Failed to query logs: {}", e.getMessage());
        }
        return logs;
    }

    /**
     * Get total number of log entries.
     *
     * @return Count of log entries, or -1 if unavailable
     */
    public long getLogCount() {
        if (!available) return -1;
        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(
                     "SELECT COUNT(*) FROM orchestrator_logs")) {
            if (rs.next()) return rs.getLong(1);
        } catch (SQLException e) {
            LOG.error("Failed to count logs: {}", e.getMessage());
        }
        return -1;
    }

    /**
     * Check if database is connected.
     *
     * @return true if database is available
     */
    public boolean isAvailable() {
        return available;
    }

    /**
     * Close the connection pool.
     */
    public void close() {
        if (dataSource != null && !dataSource.isClosed()) {
            dataSource.close();
            LOG.info("Database connection pool closed");
        }
    }

    /**
     * Create the orchestrator_logs table if it doesn't exist.
     */
    private void ensureTable() throws SQLException {
        String sql = "CREATE TABLE IF NOT EXISTS orchestrator_logs ("
                + "id SERIAL PRIMARY KEY, "
                + "action VARCHAR(100) NOT NULL, "
                + "details TEXT, "
                + "created_at TIMESTAMP NOT NULL DEFAULT NOW()"
                + ")";
        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement()) {
            stmt.execute(sql);
            LOG.debug("Table orchestrator_logs verified/created");
        }
    }
}
