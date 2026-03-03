import unittest
import pandas as pd
from data import technical_indicators

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'close': [100, 110, 120, 115, 125],
            'high': [105, 115, 125, 120, 130],
            'low': [95, 105, 115, 110, 120]
        })

    def test_rsi(self):
        rsi = technical_indicators.rsi(self.data['close'])
        self.assertEqual(len(rsi), len(self.data['close']))

    def test_macd(self):
        macd, signal = technical_indicators.macd(self.data['close'])
        self.assertEqual(len(macd), len(self.data['close']))
        self.assertEqual(len(signal), len(self.data['close']))

    def test_sma(self):
        sma = technical_indicators.sma(self.data['close'], window=3)
        self.assertEqual(len(sma), len(self.data['close']))

    def test_ema(self):
        ema = technical_indicators.ema(self.data['close'], span=3)
        self.assertEqual(len(ema), len(self.data['close']))

    def test_volatility(self):
        vol = technical_indicators.volatility(self.data['close'], window=3)
        self.assertEqual(len(vol), len(self.data['close']))

if __name__ == '__main__':
    unittest.main()
