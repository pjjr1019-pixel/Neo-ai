import unittest
from data.validation import validate_data
from data.lineage_tracker import track_lineage

class TestValidationLineage(unittest.TestCase):
    def test_validate_data(self):
        self.assertTrue(validate_data([1,2,3]))
    def test_track_lineage(self):
        self.assertTrue(track_lineage([1,2,3]))

if __name__ == "__main__":
    unittest.main()
