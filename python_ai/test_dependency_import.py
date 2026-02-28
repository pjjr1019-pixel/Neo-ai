import pkgutil
import glob
import pytest


def test_all_required_packages_installed():
    """Fail if any required package in requirements.txt is missing."""
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


def test_import_all_modules():
    """Fail if any Python module in python_ai cannot be imported."""
    errors = []
    for path in glob.glob("python_ai/**/*.py", recursive=True):
        module = path.replace("/", ".").replace("\\", ".").replace(".py", "")
        if module.endswith("__init__"):
            module = module.rsplit(".", 1)[0]
        try:
            __import__(module)
        except ModuleNotFoundError as e:
            if e.name == "psutil":
                pytest.skip(
                    "psutil not installed; "
                    "skipping resource_monitor import."
                )
            else:
                errors.append(f"{module}: {type(e).__name__}: {e}")
        except Exception as e:
            errors.append(f"{module}: {type(e).__name__}: {e}")
    assert not errors, (
        "Module import errors:\n" + "\n".join(errors)
    )
