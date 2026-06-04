FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code first so pip install . can find the package
COPY pyproject.toml .
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install Python dependencies and the package itself
RUN pip install --no-cache-dir .

# Create non-root user and switch
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "scripts/train_model.py"]
