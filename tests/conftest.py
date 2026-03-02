"""
Shared pytest configuration and fixtures for NEO test suite.

This conftest.py ensures the project root is on sys.path
so that ``import python_ai.*`` works from the tests/ directory.

It also overrides the FastAPI auth dependency so that
test clients can hit protected endpoints without real tokens.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for absolute imports.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@pytest.fixture(autouse=True)
def _override_auth():
    """Override auth dependencies so tests bypass JWT/API-key checks.

    Yields control to the test, then restores overrides.
    """
    from python_ai.auth.dependencies import get_current_user, get_optional_user
    from python_ai.auth.models import User, UserRole
    from python_ai.fastapi_service.fastapi_service import app

    test_user = User(
        username="test_user",
        email="test@neo.ai",
        full_name="Test User",
        roles=[UserRole.ADMIN],
        disabled=False,
    )

    app.dependency_overrides[get_current_user] = lambda: test_user
    app.dependency_overrides[get_optional_user] = lambda: test_user
    yield
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_optional_user, None)
