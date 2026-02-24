# NEO Hybrid AI — Automated Feature Engineering

## Overview
- Generate new features to improve model performance and data analysis.
- Examples: rolling averages, percent changes, lagged values.

## Example (Python)
import pandas as pd

# Sample data
data = pd.DataFrame({
    'price': [100, 110, 120, 115, 125],
    'volume': [200, 210, 220, 215, 225]
})

# Rolling average (window=3)
data['price_sma3'] = data['price'].rolling(window=3).mean()

# Percent change
data['price_pct_change'] = data['price'].pct_change()

# Lagged value (previous step)
data['price_lag1'] = data['price'].shift(1)

print(data)

---
## Documentation
- Log feature engineering steps and their impact on model/data.
- Update this file as new features are added.