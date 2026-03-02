import ast
import glob


def test_all_functions_have_docstrings():
    """Fail if any function or class is missing a docstring."""
    missing = []
    for path in glob.glob("python_ai/**/*.py", recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        missing.append(f"{path}:{node.name}")
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)


def test_line_length():
    """Fail if any line exceeds 79 characters."""
    too_long = []
    for path in glob.glob("python_ai/**/*.py", recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if len(line.rstrip()) > 79:
                    too_long.append(f"{path}:{i}: {line.strip()}")
    assert not too_long, "Lines too long:\n" + "\n".join(too_long)
