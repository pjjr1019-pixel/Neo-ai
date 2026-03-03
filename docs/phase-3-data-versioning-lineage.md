## Data Lineage and Validation Policies


*Add diagrams and further details as implementation progresses.*
# Data validation is performed using Great Expectations in `data/validation.py`.
# Lineage tracking is automated using OpenLineage in `data/lineage_tracker.py`.
# All validation and lineage logic are tested in `tests/test_validation_ge.py` and `tests/test_lineage_openlineage.py`.

# Advanced test coverage includes edge cases and failure scenarios for both validation and lineage modules.

# Diagrams and further details will be added as implementation progresses.
# NEO Hybrid AI — Data Versioning & Lineage

## Overview
- Keep a record of every dataset ingested or created (like saving different versions of a file).
- Use tools (like DVC) or a custom metadata table to track when and how each dataset was made or changed.
- Write down where the data came from and what steps were taken to process it.
- This helps you always know which data was used, and makes it easy to repeat or check your work later.

## Example (Python with DVC)
# Initialize DVC in your project
# dvc init
# Track a dataset
# dvc add data/raw/historical_data.csv
# Commit changes
# git add . && git commit -m "Add versioned historical data"

## Example (Custom Metadata Table)
PostgreSQL table: data_lineage
- id SERIAL PRIMARY KEY
- dataset_name TEXT
- version TEXT
- source TEXT
- processing_steps TEXT
- created_at TIMESTAMP

---
## Documentation
- Log all dataset versions and processing steps.
- Update this file as lineage tracking evolves.