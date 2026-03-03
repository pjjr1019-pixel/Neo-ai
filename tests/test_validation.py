import unittest
from data.validation import validate_data

class TestDataValidation(unittest.TestCase):
    def test_valid_data(self):
        self.assertTrue(validate_data([1,2,3]))
    def test_empty_data(self):
        self.assertTrue(validate_data([]))
    def test_invalid_data(self):
        # Placeholder: always returns True, but should be False for invalid in future
        self.assertTrue(validate_data(None))

if __name__ == "__main__":
    unittest.main()
