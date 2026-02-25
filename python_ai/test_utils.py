import numpy as np
import pytest


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


@pytest.mark.parametrize(
    "prices, signals, expected",
    [
        ([100, 110, 120], ['buy', 'hold', 'sell'], 1000 + 100*0.01 - 120*0.01),
        ([0, 0, 0], ['buy', 'sell', 'buy'], 1000),  # edge: zero prices
        ([100], ['buy'], 1000 + 1),  # single buy
        ([100], ['sell'], 1000 - 1),  # single sell
        ([], [], 1000),  # edge: empty input
    ]
)
def test_simulate_trading(prices, signals, expected):
    result = simulate_trading(prices, signals)
    assert result == expected


@pytest.mark.parametrize(
    "returns, expected_type",
    [
        (np.array([0.01, -0.02, 0.03]), float),
        (np.array([0.0, 0.0, 0.0]), float),  # edge: zero returns
        (np.array([1e6, -1e6]), float),  # large values
    ]
)
def test_sharpe_ratio(returns, expected_type):
    ratio = sharpe_ratio(returns)
    assert isinstance(ratio, expected_type)


@pytest.mark.parametrize(
    "metrics, expected_score",
    [
        ([{'score': 0.8}, {'score': 0.9}, {'score': 0.85}], 0.9),
        ([{'score': -1}, {'score': 0}], 0),  # edge: negative score
        ([{'score': 1}], 1),  # single model
    ]
)
def test_select_best_model(metrics, expected_score):
    best = select_best_model(metrics)
    assert best['score'] == expected_score


def test_sharpe_ratio_zero_division():
    # Edge: std is zero, should return nan and raise a warning
    import warnings
    import numpy as np
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sharpe_ratio(np.array([1, 1, 1]))
        assert np.isinf(result)
        assert any(issubclass(warn.category, RuntimeWarning) for warn in w)
