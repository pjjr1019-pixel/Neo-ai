.PHONY: help install dev lint format test test-full test-parallel coverage security clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────

install:  ## Install production dependencies
	pip install -r requirements.txt

dev:  ## Install dev dependencies + pre-commit hooks
	pip install -r requirements-dev.txt
	pip install pre-commit
	pre-commit install

# ── Quality ────────────────────────────────────────────────────

lint:  ## Run all linters (black, isort, flake8, mypy, bandit)
	black --check --line-length 79 python_ai/
	isort --check-only --profile black --line-length 79 python_ai/
	flake8 python_ai/
	mypy python_ai/ --ignore-missing-imports
	bandit -r python_ai/ -c .bandit -q

format:  ## Auto-format code (black + isort)
	black --line-length 79 python_ai/
	isort --profile black --line-length 79 python_ai/

# ── Testing ────────────────────────────────────────────────────

test:  ## Run tests (stop on first failure)
	python -m pytest tests/ --tb=short --maxfail=1

test-full:  ## Run full test suite with verbose output
	python -m pytest tests/ -v --tb=short

test-parallel:  ## Run tests in parallel when pytest-xdist is installed
	python -m pytest tests/ -n auto --tb=short

coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=python_ai --cov-report=term-missing

# ── Security ───────────────────────────────────────────────────

security:  ## Run security audit (bandit + pip-audit)
	bandit -r python_ai/ -c .bandit
	pip-audit || true

# ── Cleanup ────────────────────────────────────────────────────

clean:  ## Remove caches, coverage, build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage_report.txt
