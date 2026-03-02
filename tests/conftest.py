"""
Shared pytest configuration and fixtures for NEO test suite.

This conftest.py ensures the project root is on sys.path
so that ``import python_ai.*`` works from the tests/ directory.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for absolute imports.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
