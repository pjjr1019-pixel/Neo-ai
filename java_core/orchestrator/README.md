# NEO Java Orchestrator Core

The Java orchestrator for the NEO Hybrid AI trading system. Manages the autonomous trading loop, risk management, market data feeds, caching, database logging, authentication, and a real-time HTTP dashboard — all wired to the Python FastAPI AI backend.

## Architecture

```
OrchestratorMain (entry point)
├── OrchestratorConfig      — centralized configuration (env vars / properties / defaults)
├── AutonomousLoop           — scheduled fetch → features → predict → risk → execute → log → feedback
│   ├── DataFeedClient       — simulated OHLCV market data (geometric Brownian motion)
│   ├── ApiClient            — REST client for Python AI backend (/predict, /compute-features, /learn)
│   ├── RiskManagementEngine — 6-layer risk checks, circuit breaker, drawdown protection
│   ├── RedisCache           — Jedis pool with TTL, graceful degradation
│   └── DatabaseLogger       — HikariCP PostgreSQL audit trail, auto table creation
├── AuthManager              — JWT validation, HMAC-SHA256 signing/verification
└── Dashboard                — JDK HttpServer with 6 REST endpoints + Prometheus metrics
```

## Components

| Component | Description |
|---|---|
| **OrchestratorConfig** | Builder pattern config from `NEO_*` env vars → `orchestrator.properties` → defaults |
| **ApiClient** | REST client with retry + exponential backoff, JWT auth, configurable timeout |
| **RiskManagementEngine** | Confidence, volatility, signal, circuit breaker, drawdown, position sizing checks |
| **AutonomousLoop** | `ScheduledExecutorService`-based trading pipeline with trade history |
| **DataFeedClient** | Thread-safe simulated OHLCV feed with configurable volatility |
| **RedisCache** | Jedis connection pool, `neo:` key prefix, graceful degradation |
| **DatabaseLogger** | HikariCP-pooled PostgreSQL logging with `orchestrator_logs` table |
| **AuthManager** | Token validation, HMAC signing/verification, constant-time comparison |
| **Dashboard** | HTTP endpoints: `/status`, `/trades`, `/risk`, `/health`, `/logs`, `/metrics` |
| **OrchestratorMain** | Wires all components, shutdown hooks, health checks |

## Prerequisites

- **Java 17+** (OpenJDK Temurin recommended)
- **Maven 3.6+**
- **Python AI backend** running at `http://localhost:8000` (optional — loop starts regardless)
- **PostgreSQL** at `localhost:5432/neoai_db` (optional — degrades gracefully)
- **Redis** at `localhost:6379` (optional — degrades gracefully)

## Build & Run

```bash
# Compile
mvn clean compile

# Run tests (117 tests)
mvn clean test

# Package JAR
mvn clean package

# Run orchestrator
java -jar target/orchestrator-1.0.0.jar
```

## Configuration

All settings can be overridden via environment variables or `src/main/resources/orchestrator.properties`:

| Environment Variable | Default | Description |
|---|---|---|
| `NEO_API_BASE_URL` | `http://localhost:8000` | Python AI backend URL |
| `NEO_API_TOKEN` | *(none)* | JWT auth token |
| `NEO_API_SECRET` | *(none)* | HMAC signing secret |
| `NEO_CONFIDENCE_THRESHOLD` | `0.7` | Minimum prediction confidence |
| `NEO_MAX_VOLATILITY` | `2.0` | Maximum allowed volatility |
| `NEO_MAX_DRAWDOWN_PCT` | `5.0` | Max drawdown before halt (%) |
| `NEO_DAILY_LOSS_LIMIT` | `500.0` | Circuit breaker loss limit |
| `NEO_MAX_POSITION_SIZE` | `10000.0` | Maximum position size |
| `NEO_LOOP_INTERVAL_SEC` | `60` | Seconds between trading cycles |
| `NEO_TRADING_SYMBOL` | `BTC/USD` | Trading pair |
| `NEO_DB_URL` | `jdbc:postgresql://localhost:5432/neoai_db` | Database URL |
| `NEO_DB_USER` | `neo` | Database username |
| `NEO_DB_PASSWORD` | `neo_password` | Database password |
| `NEO_REDIS_HOST` | `localhost` | Redis host |
| `NEO_REDIS_PORT` | `6379` | Redis port |
| `NEO_REDIS_TTL` | `300` | Cache TTL in seconds |
| `NEO_DASHBOARD_PORT` | `8081` | Dashboard HTTP port |

## Dashboard Endpoints

| Endpoint | Description |
|---|---|
| `GET /status` | Loop state (running, cycle count, trades, P&L) |
| `GET /trades` | Last 50 trade records as JSON array |
| `GET /risk` | Risk engine state (P&L, circuit breaker, drawdown) |
| `GET /health` | Component health (API, DB, Redis, loop) — returns 200/503 |
| `GET /logs` | Recent database audit log entries |
| `GET /metrics` | Prometheus text format metrics |

## Testing

117 unit tests across 8 test classes using JUnit 5 + Mockito:

- **ApiClientTest** — PredictionResult logic, constructor validation
- **AuthManagerTest** — Token validation, no-auth mode, HMAC signing/verification
- **AutonomousLoopTest** — Full cycle mocking (7 scenarios), lifecycle, status, trade records
- **DashboardTest** — Real HTTP server on test port, all 6 endpoints
- **DatabaseLoggerTest** — No-op mode, graceful degradation, API contract
- **DataFeedClientTest** — OHLCV generation, volatility, thread safety
- **OrchestratorConfigTest** — Builder defaults, customization, validation, load
- **RiskManagementEngineTest** — Confidence, volatility, signals, circuit breaker, drawdown, P&L

## Project Structure

```
java_core/orchestrator/
├── pom.xml
├── src/
│   ├── main/
│   │   ├── java/orchestrator/
│   │   │   ├── config/
│   │   │   │   └── OrchestratorConfig.java
│   │   │   ├── ApiClient.java
│   │   │   ├── AuthManager.java
│   │   │   ├── AutonomousLoop.java
│   │   │   ├── Dashboard.java
│   │   │   ├── DatabaseLogger.java
│   │   │   ├── DataFeedClient.java
│   │   │   ├── OrchestratorMain.java
│   │   │   ├── RedisCache.java
│   │   │   └── RiskManagementEngine.java
│   │   └── resources/
│   │       ├── logback.xml
│   │       └── orchestrator.properties
│   └── test/
│       ├── java/orchestrator/
│       │   ├── config/
│       │   │   └── OrchestratorConfigTest.java
│       │   ├── ApiClientTest.java
│       │   ├── AuthManagerTest.java
│       │   ├── AutonomousLoopTest.java
│       │   ├── DashboardTest.java
│       │   ├── DatabaseLoggerTest.java
│       │   ├── DataFeedClientTest.java
│       │   └── RiskManagementEngineTest.java
│       └── resources/
│           └── logback-test.xml
└── target/
```
