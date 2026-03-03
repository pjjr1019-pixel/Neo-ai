"""
Data validation using Great Expectations (placeholder).
- Acceptance: Data validation implemented and tested
"""

import pandas as pd
import great_expectations as ge

def validate_data(data):
    """
    Validate data using Great Expectations. Accepts a pandas DataFrame or list of dicts.
    Returns True if validation passes, False otherwise.
    """
    if data is None:
        return False
    # Convert to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        try:
            df = pd.DataFrame(data)
        except Exception:
            return False
    else:
        df = data
    # Create a simple expectation suite: at least 1 row, no nulls
    ge_df = ge.from_pandas(df)
    results = ge_df.expect_table_row_count_to_be_greater_than(0)
    if not results.success:
        return False
    # Example: check no nulls in any column
    for col in df.columns:
        res = ge_df.expect_column_values_to_not_be_null(col)
        if not res.success:
            return False
    return True

if __name__ == "__main__":
    print(validate_data([1,2,3]))
