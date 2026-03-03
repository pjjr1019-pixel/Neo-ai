import unittest
import pandas as pd
from data import feature_engineering

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'price': [100, 110, 120, 115, 125],
            'volume': [200, 210, 220, 215, 225]
        })

    def test_rolling_average(self):
        result = feature_engineering.add_rolling_average(self.data.copy(), 'price', window=3)
        self.assertIn('price_sma3', result.columns)
        self.assertAlmostEqual(result['price_sma3'].iloc[2], 110)

    def test_percent_change(self):
        result = feature_engineering.add_percent_change(self.data.copy(), 'price')
        self.assertIn('price_pct_change', result.columns)
        self.assertAlmostEqual(result['price_pct_change'].iloc[1], 0.1)

    def test_lagged_value(self):
        result = feature_engineering.add_lagged(self.data.copy(), 'price', lag=1)
        self.assertIn('price_lag1', result.columns)
        self.assertTrue(pd.isna(result['price_lag1'].iloc[0]))
        self.assertEqual(result['price_lag1'].iloc[1], 100)

if __name__ == '__main__':
    unittest.main()
