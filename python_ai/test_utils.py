import numpy as np


def simulate_trading(prices, signals):
    capital = 1000
    for price, signal in zip(prices, signals):
        if signal == 'buy':
            capital += price * 0.01
        elif signal == 'sell':
            capital -= price * 0.01
    return capital


def sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns)


def select_best_model(metrics):
    return max(metrics, key=lambda x: x['score'])


def test_simulate_trading():
    prices = [100, 110, 120]
    signals = ['buy', 'hold', 'sell']
    result = simulate_trading(prices, signals)
    assert result > 0


def test_sharpe_ratio():
    returns = np.array([0.01, -0.02, 0.03])
    ratio = sharpe_ratio(returns)
    assert isinstance(ratio, float)


def test_select_best_model():
    metrics = [{'score': 0.8}, {'score': 0.9}, {'score': 0.85}]
    best = select_best_model(metrics)
    assert best['score'] == 0.9
