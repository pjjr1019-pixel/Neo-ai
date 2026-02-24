# NEO Hybrid AI — Feature Engineering Test & Integration

## Test Script (Python)
import pandas as pd

# Sample data
data = pd.DataFrame({
    'price': [100, 110, 120, 115, 125],
    'volume': [200, 210, 220, 215, 225]
})

# Feature engineering
# Rolling average (window=3)
data['price_sma3'] = data['price'].rolling(window=3).mean()
# Percent change
data['price_pct_change'] = data['price'].pct_change()
# Lagged value (previous step)
data['price_lag1'] = data['price'].shift(1)

print("Engineered features:")
print(data)

# Integration: Store in PostgreSQL (pseudo-code)
# import psycopg2
# conn = psycopg2.connect(...)
# data.to_sql('features', conn, if_exists='append')

# Integration: Store in Redis (pseudo-code)
# import redis
# r = redis.Redis(...)
# r.set('features:latest', data.to_json())

---
## Logging
- Log test results and integration steps.
- Update this file as integration evolves.