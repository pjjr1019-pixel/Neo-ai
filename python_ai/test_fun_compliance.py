"""
Fun Compliance Tests for NEO Hybrid AI
Ensures style, documentation, and maintainability with a smile!
"""

import subprocess
import sys
import ast
import os
from python_ai.fun_compliance_config import FORBIDDEN_WORD


def test_flake8_compliance():
    """All Python files must be flake8-compliant."""
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "python_ai/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Flake8 errors found."


def test_all_functions_have_docstrings():
    """Every function in python_ai/ must have a docstring."""
    for root, _, files in os.walk("python_ai"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            assert ast.get_docstring(
                                node
                            ), f"No docstring in {file}:{node.name}"


def test_line_length():
    """All lines must be <=79 characters."""
    for root, _, files in os.walk("python_ai"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        assert len(line.rstrip()) <= 79, f"{file}:{i} exceeds 79 chars"


def test_no_unused_imports():
    """No unused imports allowed (flake8 F401)."""
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "--select=F401", "python_ai/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Unused imports found."


def test_no_todo_comments():
    """No forbidden comments allowed anywhere in codebase."""
    for root, _, files in os.walk("python_ai"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        is_empty = line.strip() == ""
                        is_docstring = line.strip() in {'"""', "'''"}
                        if is_empty or is_docstring:
                            continue
                        assert (
                            FORBIDDEN_WORD not in line
                        ), f"Forbidden comment in {file}"
