# Dockerfile for NEO Hybrid AI System
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "python_ai.fastapi_service.fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]
