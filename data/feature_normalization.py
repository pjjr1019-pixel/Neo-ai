import pandas as pd

def min_max_normalize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    col_name = f"{column}_norm"
    min_val = df[column].min()
    max_val = df[column].max()
    df[col_name] = (df[column] - min_val) / (max_val - min_val)
    return df

def zscore_normalize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    col_name = f"{column}_zscore"
    mean = df[column].mean()
    std = df[column].std()
    df[col_name] = (df[column] - mean) / std
    return df
