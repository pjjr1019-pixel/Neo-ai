# NEO Hybrid AI — Model Performance Metrics

## Metrics
- Sharpe ratio
- Max drawdown
- Win rate

## Example (Python)
import numpy as np

# Sample returns
daily_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])

# Sharpe ratio
sharpe = np.mean(daily_returns) / np.std(daily_returns)
print('Sharpe ratio:', sharpe)

# Max drawdown
def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return np.min(drawdown)

print('Max drawdown:', max_drawdown(daily_returns))

# Win rate
win_rate = np.sum(daily_returns > 0) / len(daily_returns)
print('Win rate:', win_rate)

---
## Logging
- Log computed metrics and results
- Update this file as metric logic evolves