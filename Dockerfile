# CPU-only Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./
COPY segmentor/ ./segmentor/
COPY service/ ./service/
COPY config.yaml ./

# Install package
RUN pip install --no-cache-dir -e .[server,torch]

# Create non-root user
RUN useradd -m -u 1000 segmentor && chown -R segmentor:segmentor /app
USER segmentor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import requests; requests.get('http://localhost:8080/v1/healthz')"

EXPOSE 8080

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]