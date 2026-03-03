# NEO Hybrid AI — Data Validation

## Overview
- Data validation is performed using Great Expectations in `data/validation.py`.
- Validation logic ensures all ingested and processed data meets schema, type, and value constraints.
- Validation is triggered automatically in both batch and stream pipeline modes.

## Example (Python with Great Expectations)
import great_expectations as ge
import pandas as pd

data = pd.DataFrame({
    'price': [100, 110, 120, 115, 125],
    'volume': [200, 210, 220, 215, 225]
})

gdf = ge.from_pandas(data)
result = gdf.expect_column_values_to_be_between('price', 90, 130)
print(result)

## Integration
- Validation results are logged and can trigger alerts if data fails expectations.
- All validation steps are tested in `tests/test_validation_ge.py`.

---
## Documentation
- Document all validation rules and update this file as validation logic evolves.
