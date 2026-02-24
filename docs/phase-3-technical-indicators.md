# NEO Hybrid AI — Technical Indicators Computation

## Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- Volatility

## Example (Python)
import pandas as pd
import numpy as np

# Sample price data
prices = pd.Series([100, 110, 120, 115, 125])

# SMA
sma = prices.rolling(window=3).mean()
print('SMA:', sma)

# EMA
ema = prices.ewm(span=3, adjust=False).mean()
print('EMA:', ema)

# RSI
# (Simple version)
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rsi = compute_rsi(prices, window=3)
print('RSI:', rsi)

---
## Logging
- Log computed indicators for each batch.

---
Update this file as indicator logic evolves.