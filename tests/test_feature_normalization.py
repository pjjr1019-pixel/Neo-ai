import unittest
import pandas as pd
from data import feature_normalization

class TestFeatureNormalization(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

    def test_min_max_normalization(self):
        result = feature_normalization.min_max_normalize(self.data.copy(), 'feature1')
        self.assertTrue(result['feature1_norm'].min() == 0)
        self.assertTrue(result['feature1_norm'].max() == 1)

    def test_zscore_normalization(self):
        result = feature_normalization.zscore_normalize(self.data.copy(), 'feature2')
        self.assertAlmostEqual(result['feature2_zscore'].mean(), 0, places=6)
        self.assertAlmostEqual(result['feature2_zscore'].std(), 1, places=6)

if __name__ == '__main__':
    unittest.main()
