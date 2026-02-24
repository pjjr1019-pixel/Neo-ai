
package data_ingestion;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Scanner;
import java.util.logging.Logger;

/**
 * RealTimeDataFetcher fetches data from a given API endpoint.
 * Handles network errors gracefully and logs all operations.
 */
public class RealTimeDataFetcher {
    private static final Logger logger = Logger.getLogger("RealTimeDataFetcher");

    /**
     * Fetch data from the given API URL.
     * @param apiUrl The API endpoint URL.
     * @return The response data as a string, or error JSON if failed.
     */
    public static String fetchData(String apiUrl) {
        StringBuilder result = new StringBuilder();
        try {
            URL url = new URL(apiUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            int responseCode = conn.getResponseCode();
            logger.info("Fetching data from: " + apiUrl + " | Response: " + responseCode);
            if (responseCode == 200) {
                try (Scanner scanner = new Scanner(conn.getInputStream())) {
                    while (scanner.hasNext()) {
                        result.append(scanner.nextLine());
                    }
                }
                logger.info("Data fetched successfully.");
            } else {
                logger.warning("Failed to fetch data. Response code: " + responseCode);
                result.append("{\"error\": \"Failed to fetch data. Response code: " + responseCode + "\"}");
            }
        } catch (IOException e) {
            logger.severe("Error fetching data: " + e.getMessage());
            result.append("{\"error\": \"Exception: " + e.getMessage() + "\"}");
        }
        return result.toString();
    }

    /**
     * Main method for standalone testing.
     * @param args Command-line arguments (not used).
     */
    public static void main(String[] args) {
        String sampleApi = "https://api.coindesk.com/v1/bpi/currentprice.json";
        String data = fetchData(sampleApi);
        logger.info("Sample data: " + data);
    }
}
