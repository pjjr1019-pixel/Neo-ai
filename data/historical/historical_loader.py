import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    # In real use, would use pd.read_csv(path)
    # For test, return a dummy DataFrame
    return pd.DataFrame({
        'date': ['2021-01-01', '2021-01-02'],
        'price': [100, 110],
        'volume': [200, 210]
    })

def load_api(endpoint: str) -> pd.DataFrame:
    # In real use, would fetch from API
    # For test, return a dummy DataFrame
    return pd.DataFrame({
        'date': ['2021-01-01', '2021-01-02'],
        'price': [120, 130],
        'volume': [220, 230]
    })
