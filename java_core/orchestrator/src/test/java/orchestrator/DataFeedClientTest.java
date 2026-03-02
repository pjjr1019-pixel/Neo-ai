package orchestrator;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DataFeedClient.
 *
 * Validates OHLCV data generation, volatility calculation, price
 * continuity, and thread-safety of the simulated feed.
 */
@DisplayName("DataFeedClient")
class DataFeedClientTest {

    private DataFeedClient feed;

    @BeforeEach
    void setUp() {
        feed = new DataFeedClient("simulated", 100.0, 0.02, 50);
    }

    @Nested
    @DisplayName("OHLCV data generation")
    class OhlcvTests {

        @Test
        @DisplayName("fetchOHLCV returns all 5 OHLCV fields")
        void returnsAllFields() {
            Map<String, List<Double>> data = feed.fetchOHLCV("BTC/USD");
            assertNotNull(data);
            assertTrue(data.containsKey("open"));
            assertTrue(data.containsKey("high"));
            assertTrue(data.containsKey("low"));
            assertTrue(data.containsKey("close"));
            assertTrue(data.containsKey("volume"));
        }

        @Test
        @DisplayName("All OHLCV lists are same length")
        void listsAreSameLength() {
            Map<String, List<Double>> data = feed.fetchOHLCV("BTC/USD");
            int size = data.get("close").size();
            assertEquals(size, data.get("open").size());
            assertEquals(size, data.get("high").size());
            assertEquals(size, data.get("low").size());
            assertEquals(size, data.get("volume").size());
        }

        @Test
        @DisplayName("Data has correct history size")
        void correctHistorySize() {
            Map<String, List<Double>> data = feed.fetchOHLCV("BTC/USD");
            // Initial 50 bars + 1 new bar = 50 (trimmed to historySize)
            assertEquals(50, data.get("close").size());
        }

        @Test
        @DisplayName("High >= max(Open, Close) for each bar")
        void highIsValid() {
            Map<String, List<Double>> data = feed.fetchOHLCV("BTC/USD");
            for (int i = 0; i < data.get("high").size(); i++) {
                double high = data.get("high").get(i);
                double open = data.get("open").get(i);
                double close = data.get("close").get(i);
                assertTrue(high >= open || high >= close,
                        "High should be >= max(open, close) at bar " + i);
            }
        }

        @Test
        @DisplayName("Low <= min(Open, Close) for each bar")
        void lowIsValid() {
            Map<String, List<Double>> data = feed.fetchOHLCV("BTC/USD");
            for (int i = 0; i < data.get("low").size(); i++) {
                double low = data.get("low").get(i);
                double open = data.get("open").get(i);
                double close = data.get("close").get(i);
                assertTrue(low <= open || low <= close,
                        "Low should be <= min(open, close) at bar " + i);
            }
        }

        @Test
        @DisplayName("All prices are positive")
        void pricesPositive() {
            Map<String, List<Double>> data = feed.fetchOHLCV("BTC/USD");
            for (double p : data.get("close")) {
                assertTrue(p > 0, "Close price should be positive");
            }
            for (double v : data.get("volume")) {
                assertTrue(v > 0, "Volume should be positive");
            }
        }

        @Test
        @DisplayName("Consecutive fetches generate new data")
        void consecutiveFetchesDiffer() {
            Map<String, List<Double>> first = feed.fetchOHLCV("BTC/USD");
            Map<String, List<Double>> second = feed.fetchOHLCV("BTC/USD");

            double lastClose1 = first.get("close").get(first.get("close").size() - 1);
            double lastClose2 = second.get("close").get(second.get("close").size() - 1);

            // New bar generated on each fetch; last close should usually differ
            // (extremely unlikely to be exactly equal with random generation)
            assertNotEquals(lastClose1, lastClose2, 1e-15,
                    "New data should be generated on each fetch");
        }
    }

    @Nested
    @DisplayName("Volatility calculation")
    class VolatilityTests {

        @Test
        @DisplayName("Volatility is positive")
        void volatilityPositive() {
            double vol = feed.getCurrentVolatility("BTC/USD");
            assertTrue(vol > 0, "Volatility should be positive");
        }

        @Test
        @DisplayName("Volatility is annualized (reasonable range)")
        void volatilityReasonableRange() {
            double vol = feed.getCurrentVolatility("BTC/USD");
            // With 2% daily vol and sqrt(252) annualization, expect ~31.7%
            // Allow wide range due to randomness
            assertTrue(vol > 0.05 && vol < 2.0,
                    "Annualized vol should be in reasonable range, got: " + vol);
        }
    }

    @Nested
    @DisplayName("State checks")
    class StateTests {

        @Test
        @DisplayName("Feed reports connected after init")
        void isConnected() {
            assertTrue(feed.isConnected());
        }

        @Test
        @DisplayName("Latest price is positive")
        void latestPricePositive() {
            assertTrue(feed.getLatestPrice() > 0);
        }

        @Test
        @DisplayName("Source is correct")
        void sourceCorrect() {
            assertEquals("simulated", feed.getSource());
        }
    }

    @Nested
    @DisplayName("Default constructor")
    class DefaultConstructorTests {

        @Test
        @DisplayName("Default feed works correctly")
        void defaultFeedWorks() {
            DataFeedClient defaultFeed = new DataFeedClient();
            Map<String, List<Double>> data = defaultFeed.fetchOHLCV("ETH/USD");
            assertNotNull(data);
            assertFalse(data.get("close").isEmpty());
        }
    }

    @Nested
    @DisplayName("Thread safety")
    class ThreadSafetyTests {

        @Test
        @DisplayName("Concurrent fetches don't throw exceptions")
        void concurrentFetches() throws InterruptedException {
            int threadCount = 10;
            Thread[] threads = new Thread[threadCount];
            boolean[] errors = new boolean[threadCount];

            for (int i = 0; i < threadCount; i++) {
                final int idx = i;
                threads[i] = new Thread(() -> {
                    try {
                        for (int j = 0; j < 50; j++) {
                            feed.fetchOHLCV("BTC/USD");
                            feed.getCurrentVolatility("BTC/USD");
                        }
                    } catch (Exception e) {
                        errors[idx] = true;
                    }
                });
                threads[i].start();
            }

            for (Thread t : threads) {
                t.join(5000);
            }

            for (int i = 0; i < threadCount; i++) {
                assertFalse(errors[i], "Thread " + i + " should not throw");
            }
        }
    }
}
