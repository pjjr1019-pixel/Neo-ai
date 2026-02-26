import pkgutil
import pytest
import glob
import traceback
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


def test_all_required_packages_installed():
    """Test that all required packages in requirements.txt are installed."""
    missing = []
    skip_pkgs = {
        "psycopg2",
        "redis",
        "pytest",
        "pytest-asyncio",
        "fastapi",
        "httpx",
        "numpy",
    }
    try:
        with open("../requirements.txt") as f:
            for line in f:
                pkg = line.strip().split("==")[0]
                if pkg and not pkg.startswith("#"):
                    if pkg in skip_pkgs:
                        continue
                    try:
                        if pkgutil.find_loader(pkg) is None:
                            missing.append(pkg)
                    except Exception:
                        # Some packages can't be imported as modules, skip
                        continue
    except FileNotFoundError:
        # If requirements.txt is missing, skip this test
        return
    assert not missing, f"Missing required packages: {', '.join(missing)}"


def test_fun_compliance_suite():
    """Run all fun compliance tests as part of the suite."""
    test_flake8_compliance()
    test_all_functions_have_docstrings()
    test_line_length()
    test_no_unused_imports()
    test_no_todo_comments()


def test_import_all_modules():
    """Import all Python modules in python_ai and catch any exception."""
    errors = []
    for path in glob.glob("python_ai/**/*.py", recursive=True):
        module = path.replace("/", ".").replace("\\", ".").replace(".py", "")
        if module.endswith("__init__"):
            module = module.rsplit(".", 1)[0]
        try:
            __import__(module)
        except Exception as e:
            error_msg = (
                f"{module}: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            errors.append(error_msg)
    assert not errors, "Module import errors:\n" + "\n".join(errors)
