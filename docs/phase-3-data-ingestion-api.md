## Data Pipeline Operational Modes

The data pipeline supports two modes:
- **Batch**: Processes data in discrete chunks.
- **Stream**: Processes data as it arrives in real time.

Switching logic is handled by the `DataPipeline` class in data/pipeline.py. The mode is set at initialization and determines which ingestion method is used.

Example:
```python
pipeline = DataPipeline(mode="batch")
pipeline.ingest([...])
pipeline = DataPipeline(mode="stream")
pipeline.ingest([...])
```
# NEO Hybrid AI — Data Ingestion Service API

## Service: RealTimeDataFetcher (Java)
- Fetches real-time data from external APIs.
- Logs all fetch actions, errors, and sample data.

### API Example
- Endpoint: (to be defined, e.g., /fetch?url=...)
- Method: GET
- Input: API URL
- Output: Raw data (JSON, CSV, etc.)

### Logging
- All fetch actions and errors are logged.
- Sample data is logged for verification.

---
Update this file as the service evolves. Document endpoints, schemas, and integration logic.