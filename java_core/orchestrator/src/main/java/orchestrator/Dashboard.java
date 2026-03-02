package orchestrator;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Lightweight HTTP dashboard for the NEO orchestrator.
 *
 * Exposes REST endpoints for monitoring the trading loop status,
 * viewing recent trades, checking risk engine state, and querying
 * system health. Uses the built-in JDK HttpServer (no external
 * framework needed).
 *
 * Endpoints:
 * - GET /status       — Current loop status and trading state
 * - GET /trades       — Recent trade history
 * - GET /risk         — Risk engine state (P&L, circuit breaker)
 * - GET /health       — System health check (API, DB, Redis)
 * - GET /logs         — Recent log entries
 * - GET /metrics      — Prometheus-compatible metrics
 */
public class Dashboard {
    private static final Logger LOG = LoggerFactory.getLogger(Dashboard.class);

    private final int port;
    private final AutonomousLoop loop;
    private final RiskManagementEngine riskEngine;
    private final ApiClient apiClient;
    private final DatabaseLogger dbLogger;
    private final RedisCache cache;
    private final ObjectMapper mapper = new ObjectMapper();

    private HttpServer server;
    private boolean running = false;

    /**
     * Create a dashboard server.
     *
     * @param port       HTTP port to listen on
     * @param loop       Autonomous trading loop
     * @param riskEngine Risk management engine
     * @param apiClient  API client for health checks
     * @param dbLogger   Database logger for log queries
     * @param cache      Redis cache for health checks
     */
    public Dashboard(int port, AutonomousLoop loop,
                     RiskManagementEngine riskEngine, ApiClient apiClient,
                     DatabaseLogger dbLogger, RedisCache cache) {
        this.port = port;
        this.loop = loop;
        this.riskEngine = riskEngine;
        this.apiClient = apiClient;
        this.dbLogger = dbLogger;
        this.cache = cache;
    }

    /**
     * Start the dashboard HTTP server.
     *
     * @throws IOException if the server cannot bind to the port
     */
    public void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress(port), 0);

        server.createContext("/status", new StatusHandler());
        server.createContext("/trades", new TradesHandler());
        server.createContext("/risk", new RiskHandler());
        server.createContext("/health", new HealthHandler());
        server.createContext("/logs", new LogsHandler());
        server.createContext("/metrics", new MetricsHandler());

        server.setExecutor(null);
        server.start();
        running = true;

        LOG.info("Dashboard started on http://localhost:{}", port);
        LOG.info("Endpoints: /status, /trades, /risk, /health, /logs, /metrics");
    }

    /**
     * Stop the dashboard server.
     */
    public void stop() {
        if (server != null) {
            server.stop(2);
            running = false;
            LOG.info("Dashboard stopped");
        }
    }

    public boolean isRunning() { return running; }

    // ---- HTTP Handlers ----

    private class StatusHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                respond(exchange, 405, "{\"error\":\"Method not allowed\"}");
                return;
            }

            Map<String, Object> status = loop.getStatus();
            ObjectNode json = mapper.createObjectNode();
            status.forEach((k, v) -> {
                if (v instanceof Boolean) json.put(k, (Boolean) v);
                else if (v instanceof Number) json.put(k, ((Number) v).doubleValue());
                else json.put(k, String.valueOf(v));
            });
            json.put("timestamp", Instant.now().toString());

            respond(exchange, 200, mapper.writerWithDefaultPrettyPrinter()
                    .writeValueAsString(json));
        }
    }

    private class TradesHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                respond(exchange, 405, "{\"error\":\"Method not allowed\"}");
                return;
            }

            List<AutonomousLoop.TradeRecord> trades = loop.getTradeHistory();
            ArrayNode tradesArray = mapper.createArrayNode();
            // Show last 50 trades
            int start = Math.max(0, trades.size() - 50);
            for (int i = start; i < trades.size(); i++) {
                AutonomousLoop.TradeRecord t = trades.get(i);
                ObjectNode node = mapper.createObjectNode();
                node.put("cycle", t.getCycleNum());
                node.put("timestamp", t.getTimestamp().toString());
                node.put("symbol", t.getSymbol());
                node.put("action", t.getAction());
                node.put("confidence", t.getConfidence());
                node.put("positionSize", t.getPositionSize());
                tradesArray.add(node);
            }

            ObjectNode response = mapper.createObjectNode();
            response.put("totalTrades", trades.size());
            response.set("trades", tradesArray);
            respond(exchange, 200, mapper.writerWithDefaultPrettyPrinter()
                    .writeValueAsString(response));
        }
    }

    private class RiskHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                respond(exchange, 405, "{\"error\":\"Method not allowed\"}");
                return;
            }

            ObjectNode json = mapper.createObjectNode();
            json.put("dailyPnl", riskEngine.getDailyPnl());
            json.put("currentEquity", riskEngine.getCurrentEquity());
            json.put("circuitBreakerTripped", riskEngine.isCircuitBreakerTripped());
            json.put("maxPositionSizeDollars", riskEngine.getMaxPositionSizeDollars());
            json.put("timestamp", Instant.now().toString());

            respond(exchange, 200, mapper.writerWithDefaultPrettyPrinter()
                    .writeValueAsString(json));
        }
    }

    private class HealthHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            ObjectNode json = mapper.createObjectNode();

            boolean apiHealthy = apiClient.healthCheck();
            boolean dbHealthy = dbLogger.isAvailable();
            boolean cacheHealthy = cache.isAvailable();
            boolean loopHealthy = loop.isRunning();
            boolean allHealthy = apiHealthy && dbHealthy && loopHealthy;

            json.put("status", allHealthy ? "healthy" : "degraded");
            json.put("api", apiHealthy ? "connected" : "disconnected");
            json.put("database", dbHealthy ? "connected" : "disconnected");
            json.put("cache", cacheHealthy ? "connected" : "disconnected");
            json.put("tradingLoop", loopHealthy ? "running" : "stopped");
            json.put("timestamp", Instant.now().toString());

            int statusCode = allHealthy ? 200 : 503;
            respond(exchange, statusCode, mapper.writerWithDefaultPrettyPrinter()
                    .writeValueAsString(json));
        }
    }

    private class LogsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                respond(exchange, 405, "{\"error\":\"Method not allowed\"}");
                return;
            }

            List<String> logs = dbLogger.getRecentLogs(100);
            ArrayNode logsArray = mapper.createArrayNode();
            logs.forEach(logsArray::add);

            ObjectNode response = mapper.createObjectNode();
            response.put("count", logs.size());
            response.set("logs", logsArray);
            respond(exchange, 200, mapper.writerWithDefaultPrettyPrinter()
                    .writeValueAsString(response));
        }
    }

    private class MetricsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // Prometheus-compatible text format
            StringBuilder sb = new StringBuilder();
            Map<String, Object> status = loop.getStatus();

            sb.append("# HELP neo_cycle_count Total trading cycles executed\n");
            sb.append("# TYPE neo_cycle_count counter\n");
            sb.append("neo_cycle_count ").append(status.get("cycleCount")).append("\n");

            sb.append("# HELP neo_total_trades Total trades executed\n");
            sb.append("# TYPE neo_total_trades counter\n");
            sb.append("neo_total_trades ").append(status.get("totalTrades")).append("\n");

            sb.append("# HELP neo_daily_pnl Daily profit and loss\n");
            sb.append("# TYPE neo_daily_pnl gauge\n");
            sb.append("neo_daily_pnl ").append(status.get("dailyPnl")).append("\n");

            sb.append("# HELP neo_last_confidence Last prediction confidence\n");
            sb.append("# TYPE neo_last_confidence gauge\n");
            sb.append("neo_last_confidence ").append(status.get("lastConfidence")).append("\n");

            sb.append("# HELP neo_circuit_breaker Circuit breaker status (1=tripped)\n");
            sb.append("# TYPE neo_circuit_breaker gauge\n");
            sb.append("neo_circuit_breaker ")
                    .append(Boolean.TRUE.equals(status.get("circuitBreaker")) ? 1 : 0)
                    .append("\n");

            sb.append("# HELP neo_loop_running Trading loop running status (1=running)\n");
            sb.append("# TYPE neo_loop_running gauge\n");
            sb.append("neo_loop_running ")
                    .append(Boolean.TRUE.equals(status.get("running")) ? 1 : 0)
                    .append("\n");

            exchange.getResponseHeaders().set("Content-Type",
                    "text/plain; version=0.0.4; charset=utf-8");
            byte[] bytes = sb.toString().getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        }
    }

    /**
     * Send a JSON response.
     */
    private void respond(HttpExchange exchange, int statusCode, String body)
            throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(statusCode, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }
}
