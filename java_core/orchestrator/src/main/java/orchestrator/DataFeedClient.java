package orchestrator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Market data feed client for the NEO orchestrator.
 *
 * Provides OHLCV market data from configurable sources. Currently
 * supports a simulated data feed for testing and paper trading.
 * Designed to be extended with REST/WebSocket/Kafka sources.
 *
 * Thread-safe: all public methods are synchronized.
 */
public class DataFeedClient {
    private static final Logger LOG = LoggerFactory.getLogger(DataFeedClient.class);

    private final String source;
    private double currentPrice;
    private final double volatility;
    private final int historySize;
    private final Random random;
    private final List<double[]> priceHistory; // [open, high, low, close, volume]

    /**
     * Create a data feed client.
     *
     * @param source       Data source identifier ("simulated", "binance", etc.)
     * @param initialPrice Starting price for simulated feed
     * @param volatility   Daily volatility for simulated feed (std dev of returns)
     * @param historySize  Number of historical bars to maintain
     */
    public DataFeedClient(String source, double initialPrice,
                          double volatility, int historySize) {
        this.source = source;
        this.currentPrice = initialPrice;
        this.volatility = volatility;
        this.historySize = historySize;
        this.random = new Random();
        this.priceHistory = new ArrayList<>();

        // Seed with initial history
        for (int i = 0; i < historySize; i++) {
            generateNextBar();
        }

        LOG.info("DataFeedClient initialized: source={}, price={}, volatility={}, history={}",
                source, initialPrice, volatility, historySize);
    }

    /**
     * Create a simulated data feed with defaults.
     */
    public DataFeedClient() {
        this("simulated", 100.0, 0.02, 50);
    }

    /**
     * Fetch OHLCV data for a symbol.
     *
     * Returns the latest historical bars as lists of open, high,
     * low, close, and volume values.
     *
     * @param symbol Trading symbol (used for logging; simulated feed ignores it)
     * @return Map with keys "open", "high", "low", "close", "volume"
     */
    public synchronized Map<String, List<Double>> fetchOHLCV(String symbol) {
        // Generate a new bar for each fetch
        generateNextBar();

        Map<String, List<Double>> result = new HashMap<>();
        List<Double> opens = new ArrayList<>();
        List<Double> highs = new ArrayList<>();
        List<Double> lows = new ArrayList<>();
        List<Double> closes = new ArrayList<>();
        List<Double> volumes = new ArrayList<>();

        for (double[] bar : priceHistory) {
            opens.add(bar[0]);
            highs.add(bar[1]);
            lows.add(bar[2]);
            closes.add(bar[3]);
            volumes.add(bar[4]);
        }

        result.put("open", opens);
        result.put("high", highs);
        result.put("low", lows);
        result.put("close", closes);
        result.put("volume", volumes);

        LOG.debug("Fetched {} OHLCV bars for {} (source={})",
                priceHistory.size(), symbol, source);
        return result;
    }

    /**
     * Get the current estimated volatility for a symbol.
     *
     * Computes realized volatility from recent price returns.
     *
     * @param symbol Trading symbol
     * @return Estimated volatility (annualized std dev of returns)
     */
    public synchronized double getCurrentVolatility(String symbol) {
        if (priceHistory.size() < 2) {
            return volatility;
        }

        // Compute realized volatility from last 20 returns
        int lookback = Math.min(20, priceHistory.size() - 1);
        double[] returns = new double[lookback];
        for (int i = 0; i < lookback; i++) {
            int idx = priceHistory.size() - 1 - i;
            double close = priceHistory.get(idx)[3];
            double prevClose = priceHistory.get(idx - 1)[3];
            returns[i] = Math.log(close / prevClose);
        }

        // Standard deviation of log returns
        double mean = 0;
        for (double r : returns) mean += r;
        mean /= returns.length;

        double sumSq = 0;
        for (double r : returns) sumSq += (r - mean) * (r - mean);
        double stdDev = Math.sqrt(sumSq / returns.length);

        // Annualize: multiply by sqrt(252) for daily data
        return stdDev * Math.sqrt(252);
    }

    /**
     * Get the latest closing price.
     *
     * @return Most recent close price, or 0 if no data
     */
    public synchronized double getLatestPrice() {
        if (priceHistory.isEmpty()) return 0.0;
        return priceHistory.get(priceHistory.size() - 1)[3];
    }

    /**
     * Check if the data feed is connected and receiving data.
     *
     * @return true if data is available
     */
    public boolean isConnected() {
        return !priceHistory.isEmpty();
    }

    /**
     * Get the data source identifier.
     *
     * @return Source name (e.g., "simulated", "binance")
     */
    public String getSource() {
        return source;
    }

    /**
     * Generate a simulated OHLCV bar using geometric Brownian motion.
     */
    private void generateNextBar() {
        double returnPct = random.nextGaussian() * volatility;
        double newPrice = currentPrice * (1 + returnPct);

        double open = currentPrice;
        double close = newPrice;
        double high = Math.max(open, close) * (1 + Math.abs(random.nextGaussian() * volatility / 2));
        double low = Math.min(open, close) * (1 - Math.abs(random.nextGaussian() * volatility / 2));
        double volume = 1000 + random.nextDouble() * 9000;

        priceHistory.add(new double[]{open, high, low, close, volume});
        currentPrice = newPrice;

        // Trim to max history size
        while (priceHistory.size() > historySize) {
            priceHistory.remove(0);
        }
    }
}
