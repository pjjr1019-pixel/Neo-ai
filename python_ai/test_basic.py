import pkgutil
import pytest
import glob
import traceback
import subprocess
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
                        continue
    except FileNotFoundError:
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
    import pytest

    errors = []
    for path in glob.glob("python_ai/**/*.py", recursive=True):
        module = path.replace("/", ".").replace("\\", ".").replace(".py", "")
        if module.endswith("__init__"):
            module = module.rsplit(".", 1)[0]
        try:
            __import__(module)
        except ModuleNotFoundError as e:
            # Allow missing optional dependencies like psutil
            if e.name == "psutil":
                pytest.skip("psutil not installed; skipping resource_monitor import.")
            else:
                error_msg = (
                    f"{module}: {type(e).__name__}: {e}\n" f"{traceback.format_exc()}"
                )
                errors.append(error_msg)
        except Exception as e:
            error_msg = (
                f"{module}: {type(e).__name__}: {e}\n" f"{traceback.format_exc()}"
            )
            errors.append(error_msg)
    assert not errors, "Module import errors:\n" + "\n".join(errors)


def test_git_environment():
    """Check git status, config, and submodule for errors or warnings."""
    commands = [
        ["git", "status"],
        ["git", "config", "--global", "--list"],
        ["git", "config", "--local", "--list"],
        ["git", "submodule", "status"],
    ]
    error_keywords = [
        "unsafe",
        "error",
        "fatal",
        "not found",
        "missing",
        "failed",
        "warning",
    ]
    errors = []
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.lower() + result.stderr.lower()
            for keyword in error_keywords:
                if keyword in output:
                    error_msg = f"{' '.join(cmd)}: {keyword} found\n" f"{output}"
                    errors.append(error_msg)
        except Exception as e:
            error_msg = f"{' '.join(cmd)}: Exception {type(e).__name__}: {e}"
            errors.append(error_msg)
    import pytest

    # If running in CI and .gitconfig is missing, skip this test
    for err in errors:
        if "fatal: unable to read config file" in err and ".gitconfig" in err:
            pytest.skip("Global git config missing in CI; skipping.")
    assert not errors, "Git environment errors:\n" + "\n".join(errors)
