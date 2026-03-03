import pandas as pd

def add_rolling_average(df: pd.DataFrame, column: str, window: int = 3) -> pd.DataFrame:
    col_name = f"{column}_sma{window}"
    df[col_name] = df[column].rolling(window=window).mean()
    return df

def add_percent_change(df: pd.DataFrame, column: str) -> pd.DataFrame:
    col_name = f"{column}_pct_change"
    df[col_name] = df[column].pct_change()
    return df

def add_lagged(df: pd.DataFrame, column: str, lag: int = 1) -> pd.DataFrame:
    col_name = f"{column}_lag{lag}"
    df[col_name] = df[column].shift(lag)
    return df
