# Sample Python PostgreSQL Connection
import psycopg2
conn = psycopg2.connect(
    dbname='neoai_db', user='neoai', password='neoai123', host='localhost', port=5432
)
print('PostgreSQL connection successful!')
conn.close()

# Sample Python Redis Connection
import redis
r = redis.Redis(host='localhost', port=6379)
print('Redis connection successful!')
r.close()

---
# Sample Java PostgreSQL Connection
// Requires: org.postgresql.Driver
import java.sql.*;
Connection conn = DriverManager.getConnection(
    "jdbc:postgresql://localhost:5432/neoai_db", "neoai", "neoai123"
);
System.out.println("PostgreSQL connection successful!");
conn.close();

# Sample Java Redis Connection
// Requires: Jedis library
import redis.clients.jedis.Jedis;
Jedis jedis = new Jedis("localhost", 6379);
System.out.println("Redis connection successful!");
jedis.close();

---
# Verification
- Run these snippets to verify connectivity.
- Update docs as integration evolves.