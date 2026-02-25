# Deployment Guide

## Local Development
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
4. Run tests and linting:
   ```
   pytest --cov=python_ai --cov-report=term-missing
   flake8 python_ai
   mypy python_ai
   ```
5. Start the FastAPI server:
   ```
   uvicorn python_ai.fastapi_service.fastapi_service:app --reload
   ```

## Docker Deployment
1. Build the Docker image:
   ```
   docker build -t neo-hybrid-ai .
   ```
2. Run the container:
   ```
   docker run -p 8000:8000 neo-hybrid-ai
   ```

## Cloud Deployment
- Adapt the Dockerfile for your cloud provider (Heroku, AWS, Azure, etc.)
- Use the provided GitHub Actions workflow for CI/CD automation

## Monitoring
- Use `/metrics` endpoint for live resource stats
- Extend with Prometheus/Grafana as needed
