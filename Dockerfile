# NEO Hybrid AI Trading System - Multi-Stage Dockerfile

# Stage 1: Python Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development Image (for testing)
FROM runtime AS development

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "pytest", "tests/", "-v"]

# Stage 3: Production Image
FROM python:3.12-slim

WORKDIR /app

# Copy Python dependencies from runtime stage
COPY --from=runtime /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=runtime /usr/local/bin /usr/local/bin

# Copy application code
COPY python_ai/ ./python_ai/
COPY docs/ ./docs/
COPY README.md .
COPY pytest.ini .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check — hit the real FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run FastAPI via uvicorn in production
CMD ["uvicorn", "python_ai.fastapi_service.fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]
