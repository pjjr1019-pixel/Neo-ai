"""Data validation helpers with optional Great Expectations support."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _load_ge() -> Any | None:
    """Best-effort loader for Great Expectations.

    Returns ``None`` when GE is unavailable or incompatible with
    the current runtime so callers can fall back to pandas checks.
    """
    try:
        import great_expectations as ge  # type: ignore

        return ge
    except Exception:
        return None


def validate_data(data: Any) -> bool:
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
    ge = _load_ge()
    # Use GE only when available and compatible.
    try:
        if (
            ge is not None
            and hasattr(ge, "dataset")
            and hasattr(ge.dataset, "PandasDataset")
        ):
            ge_df = ge.dataset.PandasDataset(df)
            # At least 1 row
            results = ge_df.expect_table_row_count_to_be_greater_than(0)
            if not results.success:
                return False
            # Check no nulls in any column
            for col in df.columns:
                res = ge_df.expect_column_values_to_not_be_null(col)
                if not res.success:
                    return False
            return True
    except Exception:
        # Any GE runtime issue falls back to pandas-only validation.
        pass
    # Fallback: basic DataFrame validation
    if df.shape[0] == 0:
        return False
    if df.isnull().any().any():
        return False
    return True

if __name__ == "__main__":
    print(validate_data([1, 2, 3]))
