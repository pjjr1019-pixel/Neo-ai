# NEO Hybrid AI — Feature Normalization Pipeline

## Overview
- Normalize features (scaling, encoding, etc.) before storing in PostgreSQL and Redis.
- Document schema and retrieval logic.

## Example (Python)
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample data
features = pd.DataFrame({
    'price': [100, 110, 120],
    'volume': [200, 210, 220]
})

scaler = StandardScaler()
normalized = scaler.fit_transform(features)
print(normalized)

---
## Storage
- Store normalized features in PostgreSQL (table: features) and Redis (key: features:latest).

## Schema Example
PostgreSQL table: features
- id SERIAL PRIMARY KEY
- timestamp TIMESTAMP
- price FLOAT
- volume FLOAT
- ...

---
Update this file as normalization logic evolves.