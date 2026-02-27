"""
Master test suite for Neo project.
Runs FastAPI endpoint tests, flake8 compliance, and other checks.
Follows full coding best practices:
clear docstrings, proper imports, blank lines,
and error logging.
"""

import subprocess


def test_flake8_compliance():
    """
    Run flake8 compliance check on the Neo project.
    Asserts that all files are flake8 compliant.
    """
    result = subprocess.run([
        'python', '-m', 'flake8', 'python_ai/',
        '--count', '--show-source', '--statistics'
    ], capture_output=True, text=True)
    assert result.returncode == 0, (
        "Flake8 errors found:\n{}\n{}".format(result.stdout, result.stderr)
    )


if __name__ == "__main__":
    """
    Entry point for running all tests and logging errors.
    Follows best practices for error handling and reporting.
    """
    result = subprocess.run([
        'pytest', '--maxfail=0', '--disable-warnings', '-q'
    ], capture_output=True, text=True, timeout=120)
    log_path = 'error_logs/Errorlog_master.txt'
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("Pytest output:\n")
        log_file.write(result.stdout)
        log_file.write("\nPytest errors:\n")
        log_file.write(result.stderr)
    if result.returncode != 0:
        print(f"Errors logged to {log_path}")
    else:
        print("All tests passed. Log written.")
