"""Tests for model selection, regime detection, and significance utilities."""

from python_ai.model_selection import (
    detect_market_regime,
    paired_significance_test,
    purged_walk_forward_cv,
    regime_aware_score,
)


def test_purged_walk_forward_cv_produces_folds() -> None:
    folds = purged_walk_forward_cv([0.01, -0.02, 0.03, 0.01, 0.02], n_splits=3)
    assert len(folds) >= 2
    assert all(fold.test_size > 0 for fold in folds)


def test_regime_detection() -> None:
    assert detect_market_regime([0.01, 0.02, 0.005]) == "bullish"
    assert detect_market_regime([-0.02, -0.01, -0.003]) == "bearish"
    assert detect_market_regime([0.0001, -0.0002, 0.0001]) == "sideways"


def test_regime_aware_score() -> None:
    score = regime_aware_score(
        {"bull": [0.02, 0.01], "bear": [-0.01, -0.005], "side": [0.0]}
    )
    assert isinstance(score, float)


def test_paired_significance_test_returns_bounded_pvalue() -> None:
    t_stat, p_val = paired_significance_test(
        baseline=[0.01, 0.0, 0.02, -0.01],
        candidate=[0.02, 0.01, 0.03, -0.005],
    )
    assert isinstance(t_stat, float)
    assert 0.0 <= p_val <= 1.0
