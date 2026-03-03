import unittest
import pandas as pd
from data import historical_loader

class TestHistoricalLoader(unittest.TestCase):
    def test_load_csv(self):
        # Simulate loading a CSV (mocked)
        df = historical_loader.load_csv('tests/data/sample_historical.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_load_api(self):
        # Simulate loading from API (mocked)
        df = historical_loader.load_api('mock_endpoint')
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
