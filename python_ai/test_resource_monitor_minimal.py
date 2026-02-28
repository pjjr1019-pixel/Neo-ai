import pytest
from python_ai import resource_monitor


def test_log_resource_usage_runs():
    """Test that log_resource_usage runs without raising an exception."""
    # This test just checks that the function can be called without error.
    # In real scenarios, you would mock system calls and file writes.
    try:
        resource_monitor.log_resource_usage()
    except Exception as e:
        pytest.fail(f"log_resource_usage raised an exception: {e}")
