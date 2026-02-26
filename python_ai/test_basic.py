import pytest
from python_ai.test_fun_compliance import (
    test_flake8_compliance,
    test_all_functions_have_docstrings,
    test_line_length,
    test_no_unused_imports,
    test_no_todo_comments,
)


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
    """Test division by zero raises exception."""
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0


def test_fun_compliance_suite():
    """Run all fun compliance tests as part of the suite."""
    test_flake8_compliance()
    test_all_functions_have_docstrings()
    test_line_length()
    test_no_unused_imports()
    test_no_todo_comments()
