import unittest
from data.pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    def test_batch_mode(self):
        pipeline = DataPipeline(mode="batch")
        result = pipeline.ingest([1,2,3])
        self.assertIn("Batch ingested", result)
    def test_stream_mode(self):
        pipeline = DataPipeline(mode="stream")
        result = pipeline.ingest([1,2,3])
        self.assertIn("Stream ingested", result)
    def test_invalid_mode(self):
        with self.assertRaises(AssertionError):
            DataPipeline(mode="invalid")

if __name__ == "__main__":
    unittest.main()
