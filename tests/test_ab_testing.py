import pytest
from python_ai.ab_testing import StrategyVariant, ABTest


def test_strategy_variant_record_and_stats():
    v = StrategyVariant("A", 0.7)
    v.record(1.0)
    v.record(2.0)
    v.record(3.0)
    assert v.name == "A"
    assert v.weight == 0.7
    assert v.mean == 2.0
    assert round(v.std, 5) == 1.0
    d = v.to_dict()
    assert d["name"] == "A"
    assert d["weight"] == 0.7
    assert d["n"] == 3
    assert d["mean"] == 2.0
    assert round(d["std"], 5) == 1.0


def test_abtest_assignment_and_recording():
    ab = ABTest("test", "A", "B", split=0.5)
    assignments = [ab.assign() for _ in range(1000)]
    # Should be roughly balanced
    a_count = assignments.count("A")
    b_count = assignments.count("B")
    assert abs(a_count - b_count) < 200
    ab.variant_a.record(1.0)
    ab.variant_b.record(2.0)
    assert ab.variant_a.mean == 1.0
    assert ab.variant_b.mean == 2.0


def test_strategy_variant_empty_stats():
    v = StrategyVariant("empty")
    assert v.mean == 0.0
    assert v.std == 0.0
    d = v.to_dict()
    assert d["n"] == 0
    assert d["mean"] == 0.0
    assert d["std"] == 0.0


def test_abtest_to_dict():
    ab = ABTest("exp", "A", "B", split=0.3)
    ab.variant_a.record(1.0)
    ab.variant_b.record(2.0)
    d_a = ab.variant_a.to_dict()
    d_b = ab.variant_b.to_dict()
    assert d_a["name"] == "A"
    assert d_b["name"] == "B"
    assert isinstance(d_a["mean"], float)
    assert isinstance(d_b["mean"], float)
