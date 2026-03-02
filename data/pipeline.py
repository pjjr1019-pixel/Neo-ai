"""
Data pipeline supporting both streaming and batch ingestion modes.
- Acceptance: Pipeline supports both modes, switching logic documented
"""
class DataPipeline:
    def __init__(self, mode="batch"):
        assert mode in ("batch", "stream"), "Mode must be 'batch' or 'stream'"
        self.mode = mode
    def ingest(self, data):
        if self.mode == "batch":
            return self._batch_ingest(data)
        else:
            return self._stream_ingest(data)
    def _batch_ingest(self, data):
        # Placeholder for batch ingestion logic
        return f"Batch ingested {len(data)} records"
    def _stream_ingest(self, data):
        # Placeholder for streaming ingestion logic
        return f"Stream ingested {len(data)} records"

if __name__ == "__main__":
    pipeline = DataPipeline(mode="batch")
    print(pipeline.ingest([1,2,3]))
    pipeline = DataPipeline(mode="stream")
    print(pipeline.ingest([1,2,3]))
