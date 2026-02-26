package orchestrator;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;

public class DatabaseLogger {
    private final String url;
    private final String user;
    private final String password;

    public DatabaseLogger(String url, String user, String password) {
        this.url = url;
        this.user = user;
        this.password = password;
    }

    public void logAction(String action, String details) {
        String sql = "INSERT INTO orchestrator_logs (action, details, timestamp) VALUES (?, ?, NOW())";
        try (Connection conn = DriverManager.getConnection(url, user, password);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, action);
            pstmt.setString(2, details);
            pstmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
