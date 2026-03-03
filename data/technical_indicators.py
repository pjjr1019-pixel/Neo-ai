import pandas as pd
import numpy as np

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def sma(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window).mean()

def ema(series: pd.Series, span: int = 3) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def volatility(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window).std()
