
import os


def test_ci_env_variables():
    """Check for required CI environment variables."""
    required_vars = ["CI", "GITHUB_ACTIONS", "PYTHONPATH"]
    missing = [var for var in required_vars if var not in os.environ]
    assert not missing, f"Missing CI environment variables: {missing}"
