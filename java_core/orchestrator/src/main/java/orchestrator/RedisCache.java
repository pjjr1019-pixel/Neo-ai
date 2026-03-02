package orchestrator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.exceptions.JedisException;

import java.time.Duration;

/**
 * Redis cache client for the NEO orchestrator.
 *
 * Provides caching for feature vectors, predictions, and trading state
 * using Jedis connection pooling. Supports TTL-based expiration,
 * key prefixing, and graceful degradation when Redis is unavailable.
 *
 * All operations are fault-tolerant — cache misses/failures are logged
 * but never throw exceptions to the caller.
 */
public class RedisCache {
    private static final Logger LOG = LoggerFactory.getLogger(RedisCache.class);
    private static final String KEY_PREFIX = "neo:";

    private final JedisPool pool;
    private final int defaultTtlSeconds;
    private final boolean available;

    /**
     * Create a Redis cache with connection pooling.
     *
     * @param host     Redis server hostname
     * @param port     Redis server port
     * @param password Redis password (empty string = no auth)
     * @param ttlSeconds Default TTL for cached entries
     */
    public RedisCache(String host, int port, String password, int ttlSeconds) {
        this.defaultTtlSeconds = ttlSeconds;
        JedisPool tempPool = null;
        boolean connected = false;

        try {
            JedisPoolConfig poolConfig = new JedisPoolConfig();
            poolConfig.setMaxTotal(10);
            poolConfig.setMaxIdle(5);
            poolConfig.setMinIdle(1);
            poolConfig.setMaxWait(Duration.ofMillis(2000));
            poolConfig.setTestOnBorrow(true);

            if (password != null && !password.isEmpty()) {
                tempPool = new JedisPool(poolConfig, host, port, 2000, password);
            } else {
                tempPool = new JedisPool(poolConfig, host, port, 2000);
            }

            // Test connection
            try (Jedis jedis = tempPool.getResource()) {
                jedis.ping();
                connected = true;
                LOG.info("Redis cache connected: {}:{} (TTL={}s)", host, port, ttlSeconds);
            }
        } catch (JedisException e) {
            LOG.warn("Redis not available at {}:{} — cache disabled: {}",
                    host, port, e.getMessage());
        }

        this.pool = tempPool;
        this.available = connected;
    }

    /**
     * Create a Redis cache with defaults (localhost:6379, no password, 300s TTL).
     */
    public RedisCache() {
        this("localhost", 6379, "", 300);
    }

    /**
     * Store a string value with the default TTL.
     *
     * @param key   Cache key (will be auto-prefixed with "neo:")
     * @param value Value to store
     */
    public void set(String key, String value) {
        set(key, value, defaultTtlSeconds);
    }

    /**
     * Store a string value with a specific TTL.
     *
     * @param key        Cache key
     * @param value      Value to store
     * @param ttlSeconds Time-to-live in seconds
     */
    public void set(String key, String value, int ttlSeconds) {
        if (!available) return;
        try (Jedis jedis = pool.getResource()) {
            jedis.setex(KEY_PREFIX + key, ttlSeconds, value);
            LOG.debug("Cache SET: {} (TTL={}s)", key, ttlSeconds);
        } catch (JedisException e) {
            LOG.warn("Cache SET failed for {}: {}", key, e.getMessage());
        }
    }

    /**
     * Store a JSON string with the default TTL.
     *
     * @param key  Cache key
     * @param json JSON string to store
     */
    public void setJson(String key, String json) {
        set(key, json, defaultTtlSeconds);
    }

    /**
     * Retrieve a cached value.
     *
     * @param key Cache key
     * @return Cached value, or null if not found or cache unavailable
     */
    public String get(String key) {
        if (!available) return null;
        try (Jedis jedis = pool.getResource()) {
            String value = jedis.get(KEY_PREFIX + key);
            LOG.debug("Cache {}: {}", value != null ? "HIT" : "MISS", key);
            return value;
        } catch (JedisException e) {
            LOG.warn("Cache GET failed for {}: {}", key, e.getMessage());
            return null;
        }
    }

    /**
     * Delete a cached key.
     *
     * @param key Cache key to delete
     * @return true if key was deleted
     */
    public boolean delete(String key) {
        if (!available) return false;
        try (Jedis jedis = pool.getResource()) {
            long deleted = jedis.del(KEY_PREFIX + key);
            LOG.debug("Cache DEL: {} (deleted={})", key, deleted > 0);
            return deleted > 0;
        } catch (JedisException e) {
            LOG.warn("Cache DEL failed for {}: {}", key, e.getMessage());
            return false;
        }
    }

    /**
     * Check if a key exists in cache.
     *
     * @param key Cache key
     * @return true if key exists
     */
    public boolean exists(String key) {
        if (!available) return false;
        try (Jedis jedis = pool.getResource()) {
            return jedis.exists(KEY_PREFIX + key);
        } catch (JedisException e) {
            return false;
        }
    }

    /**
     * Check if Redis is connected and available.
     *
     * @return true if cache is operational
     */
    public boolean isAvailable() {
        return available;
    }

    /**
     * Close the connection pool.
     */
    public void close() {
        if (pool != null && !pool.isClosed()) {
            pool.close();
            LOG.info("Redis connection pool closed");
        }
    }
}
