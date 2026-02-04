# Grid Trading Bot - Docker Image
# Python 3.12 slim image for minimal size and security
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - gcc: Required for some Python packages with C extensions
# - libpq-dev: PostgreSQL client library (for asyncpg)
# - curl: For healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't store pip cache (smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python environment variables
# PYTHONUNBUFFERED: Don't buffer stdout/stderr (better for Docker logs)
# PYTHONDONTWRITEBYTECODE: Don't create .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check - verify Python can import main modules
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.core import get_logger; print('ok')" || exit 1

# Default command - run the integrated trading system
CMD ["python", "run_all.py"]
