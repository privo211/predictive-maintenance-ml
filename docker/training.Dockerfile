FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY pyproject.toml README.md ./
COPY config/ ./config/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --no-index --find-links=/wheels . && rm -rf /wheels

RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "scripts/train_model.py"]
