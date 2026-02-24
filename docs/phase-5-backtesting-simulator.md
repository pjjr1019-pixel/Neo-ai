# NEO Hybrid AI — Backtesting Simulator Example

## Example (Python)
import pandas as pd

# Sample historical data
data = pd.DataFrame({
    'price': [100, 110, 120, 115, 125],
    'signal': ['buy', 'hold', 'buy', 'sell', 'buy']
})

# Simulate trading
capital = 1000
positions = []
for i, row in data.iterrows():
    if row['signal'] == 'buy':
        capital += row['price'] * 0.01  # Example profit
    elif row['signal'] == 'sell':
        capital -= row['price'] * 0.01  # Example loss
    positions.append(capital)

print('Final capital:', capital)
print('Positions:', positions)

---
## Logging
- Log simulation results and performance metrics
- Update this file as simulator logic evolves