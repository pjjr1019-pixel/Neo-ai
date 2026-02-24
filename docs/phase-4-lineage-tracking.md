# NEO Hybrid AI — Automated Data/Model Lineage Tracking

## Overview
- Use MLflow/DVC for tracking model and data changes
- Ensure all model/data changes are reproducible

## Example (MLflow)
import mlflow
mlflow.start_run()
mlflow.log_param("model_type", "LSTM")
mlflow.log_metric("accuracy", 0.92)
mlflow.log_artifact("model.pth")
mlflow.end_run()

## Example (DVC)
# dvc add data/processed/features.csv
# dvc add models/model.pth

---
## Documentation
- Log all lineage tracking actions and results
- Update this file as tracking logic evolves.