import pytest


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 1, 2),
        (0, 0, 0),
        (-1, 1, 0),
        (1e6, 1e6, 2e6),
        (-1e6, 1e6, 0),
    ],
)
def test_sample_math(a, b, expected):
    """Parametrized math test: checks addition for edge and normal cases."""
    assert a + b == expected


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0
