# NEO Hybrid AI System

## Overview
NEO is a modular, robust, and fully tested hybrid AI system built with FastAPI, pytest, and modern Python best practices.

## Features
- Modular FastAPI endpoints for prediction and learning
- Comprehensive test coverage (100%)
- Flake8 style compliance
- Async test support (pytest-asyncio)
- CI/CD ready

## Setup
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:
	```
	pip install -r requirements-dev.txt
	```
4. Run tests:
	```
	pytest --maxfail=1 --disable-warnings -v
	```

## API Documentation
- FastAPI auto-generates docs at `/docs` and `/redoc` when running the app.

## Project Structure
- `python_ai/fastapi_service/fastapi_service.py`: Main FastAPI app and endpoints
- `python_ai/fastapi_service/test_fastapi_service.py`: Endpoint tests
- `python_ai/test_api.py`, `test_basic.py`, `test_utils.py`, `test_benchmark_predict.py`: Additional tests and utilities
- `requirements-dev.txt`: Dev dependencies

## CI/CD
- Ensure `pytest-asyncio` is installed in CI for async test support
- Use `pip install -r requirements-dev.txt` in your workflow

## Contributing
- Follow flake8 and modularity best practices
- Add docstrings and type hints
- Write tests for new features

---
For more details, see the code and tests. For questions, open an issue or contact the maintainer.