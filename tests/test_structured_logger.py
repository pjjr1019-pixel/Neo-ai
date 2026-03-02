import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neo_logging.structured_logger import StructuredLogger

class TestStructuredLogger(unittest.TestCase):
    def test_info_log(self):
        logger = StructuredLogger("test_logger")
        # Should not raise
        logger.info("Test info log")
    def test_error_log(self):
        logger = StructuredLogger("test_logger")
        # Should not raise
        logger.error("Test error log")

if __name__ == "__main__":
    unittest.main()
