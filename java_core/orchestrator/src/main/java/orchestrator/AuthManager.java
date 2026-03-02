package orchestrator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Base64;

/**
 * Authentication and security manager for the NEO orchestrator.
 *
 * Handles JWT token validation, API key verification, and request
 * signing (HMAC-SHA256). Validates tokens before allowing API calls
 * to the Python backend.
 *
 * Note: Named AuthManager to avoid conflict with the deprecated
 * java.lang.SecurityManager class in Java 17+.
 */
public class AuthManager {
    private static final Logger LOG = LoggerFactory.getLogger(AuthManager.class);
    private static final String HMAC_ALGORITHM = "HmacSHA256";

    private final String apiToken;
    private final String hmacSecret;
    private final boolean authEnabled;

    /**
     * Create an auth manager with token and HMAC secret.
     *
     * @param apiToken   JWT or API key for authentication (empty = disabled)
     * @param hmacSecret Secret key for HMAC request signing (empty = disabled)
     */
    public AuthManager(String apiToken, String hmacSecret) {
        this.apiToken = apiToken != null ? apiToken : "";
        this.hmacSecret = hmacSecret != null ? hmacSecret : "";
        this.authEnabled = !this.apiToken.isEmpty();

        if (authEnabled) {
            LOG.info("AuthManager initialized with API token authentication");
        } else {
            LOG.warn("AuthManager initialized WITHOUT authentication — "
                    + "all requests will be unauthenticated");
        }
    }

    /**
     * Create an auth manager with no authentication (for development).
     */
    public AuthManager() {
        this("", "");
    }

    /**
     * Validate an API token or JWT.
     *
     * Checks if the provided token matches the configured API token.
     * In production, this should validate JWT signature, expiry, etc.
     *
     * @param token Token to validate
     * @return true if the token is valid
     */
    public boolean validateToken(String token) {
        if (!authEnabled) {
            LOG.debug("Auth disabled — allowing request");
            return true;
        }

        if (token == null || token.isEmpty()) {
            LOG.warn("Empty token rejected");
            return false;
        }

        // Strip "Bearer " prefix if present
        String cleanToken = token.startsWith("Bearer ")
                ? token.substring(7) : token;

        boolean valid = apiToken.equals(cleanToken);
        if (!valid) {
            LOG.warn("Invalid token rejected");
        }
        return valid;
    }

    /**
     * Sign a request payload with HMAC-SHA256.
     *
     * Generates a signature for the given payload that can be
     * verified by the receiving service to ensure integrity.
     *
     * @param payload Request body to sign
     * @return Base64-encoded HMAC-SHA256 signature, or empty string if signing disabled
     */
    public String signRequest(String payload) {
        if (hmacSecret.isEmpty()) {
            return "";
        }

        try {
            Mac mac = Mac.getInstance(HMAC_ALGORITHM);
            SecretKeySpec keySpec = new SecretKeySpec(
                    hmacSecret.getBytes(StandardCharsets.UTF_8), HMAC_ALGORITHM);
            mac.init(keySpec);

            String timestampedPayload = Instant.now().getEpochSecond() + ":" + payload;
            byte[] hash = mac.doFinal(
                    timestampedPayload.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(hash);

        } catch (Exception e) {
            LOG.error("HMAC signing failed: {}", e.getMessage());
            return "";
        }
    }

    /**
     * Verify an HMAC signature on a received payload.
     *
     * @param payload   The payload that was signed
     * @param signature The signature to verify
     * @param timestamp The timestamp used during signing
     * @return true if the signature is valid
     */
    public boolean verifySignature(String payload, String signature, long timestamp) {
        if (hmacSecret.isEmpty()) {
            return true; // Signing not configured — allow
        }

        try {
            Mac mac = Mac.getInstance(HMAC_ALGORITHM);
            SecretKeySpec keySpec = new SecretKeySpec(
                    hmacSecret.getBytes(StandardCharsets.UTF_8), HMAC_ALGORITHM);
            mac.init(keySpec);

            String timestampedPayload = timestamp + ":" + payload;
            byte[] expectedHash = mac.doFinal(
                    timestampedPayload.getBytes(StandardCharsets.UTF_8));
            String expectedSignature = Base64.getEncoder().encodeToString(expectedHash);

            // Constant-time comparison to prevent timing attacks
            boolean valid = constantTimeEquals(expectedSignature, signature);
            if (!valid) {
                LOG.warn("HMAC signature verification failed");
            }
            return valid;

        } catch (Exception e) {
            LOG.error("HMAC verification error: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Get the authorization header value for API requests.
     *
     * @return "Bearer {token}" or empty string if no auth configured
     */
    public String getAuthHeader() {
        if (!authEnabled) return "";
        return "Bearer " + apiToken;
    }

    /**
     * Check if authentication is enabled.
     *
     * @return true if a token is configured
     */
    public boolean isAuthEnabled() {
        return authEnabled;
    }

    /**
     * Constant-time string comparison to prevent timing attacks.
     */
    private boolean constantTimeEquals(String a, String b) {
        if (a.length() != b.length()) return false;
        int result = 0;
        for (int i = 0; i < a.length(); i++) {
            result |= a.charAt(i) ^ b.charAt(i);
        }
        return result == 0;
    }
}
