<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e&color=00d2ff">
    <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e&color=00d2ff">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/FastAPI-0.136%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white&labelColor=1a1a2e&color=00d2ff">
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.136%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white&labelColor=1a1a2e&color=00d2ff">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/XGBoost-2.1%2B-FF6600?style=for-the-badge&logo=xgboost&logoColor=white&labelColor=1a1a2e&color=00d2ff">
    <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-2.1%2B-FF6600?style=for-the-badge&logo=xgboost&logoColor=white&labelColor=1a1a2e&color=00d2ff">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white&labelColor=1a1a2e&color=00d2ff">
    <img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white&labelColor=1a1a2e&color=00d2ff">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/pytest-8.0%2B-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white&labelColor=1a1a2e&color=00d2ff">
    <img alt="pytest" src="https://img.shields.io/badge/pytest-8.0%2B-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white&labelColor=1a1a2e&color=00d2ff">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/license-MIT-green?style=for-the-badge&logo=open-source-initiative&logoColor=white&labelColor=1a1a2e&color=00d2ff">
    <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge&logo=open-source-initiative&logoColor=white&labelColor=1a1a2e&color=00d2ff">
  </picture>
</p>

<div align="center">
  <h1>🏭 Industrial Predictive Maintenance Platform</h1>
  <p><strong>Production-grade ML system for equipment failure prediction with automated data quality gates.</strong></p>
  <p>9,800+ lines of Python | 7 test modules | 6 Docker services | 5-stage data quality firewall</p>
  <br>
  <p><em>Built with: Python · FastAPI · XGBoost · scikit-learn · Docker · PostgreSQL/TimescaleDB · Redis · MLflow · SHAP · Evidently AI · Grafana</em></p>
</div>

---

## 📋 Table of Contents

- [Why This Project Exists](#-why-this-project-exists)
- [Features](#-features)
- [Architecture](#-architecture)
- [Data Quality Firewall Deep Dive](#-data-quality-firewall-deep-dive)
- [Quick Start](#-quick-start)
- [Local Development](#-local-development-without-docker)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Testing](#-testing)
- [Design Decisions](#-design-decisions)
- [What I Learned](#-what-i-learned)
- [License](#-license)

---

## 🎯 Why This Project Exists

Industrial ML models don't fail because of bad algorithms. They fail because of bad data. Sensors drift. Networks drop packets. Maintenance teams swap hardware without updating the data pipeline. Your 99% accurate model from training becomes a random number generator in production — and nobody notices until a $400K transformer fails.

This project demonstrates **production ML engineering**, not Kaggle ML. It answers: *"How do you know your predictions are still trustworthy at 3 AM?"*

| Most ML Portfolio Projects | This Project |
|---|---|
| `model.fit()` → "95% accuracy!" → Jupyter notebook | Data quality firewall → drift monitoring → calibrated predictions → SHAP explanations → Docker deployment |
| Random train/test split | Temporal split — no future data leakage |
| No data validation | 5-stage quality gate: schema, range, timestamp, cross-channel, CUSUM drift |
| Single sensor | Multi-sensor fusion with redundant channel validation |
| "It works on my laptop" | `docker compose up` — 6 services, one command |
| Accuracy without context | Recall, precision, FPR, ROC-AUC with promotion gates |
| No tests | 7 test files (unit + integration + validation) |
| Unmaintainable notebook | Structured packages, type hints, mypy strict, ruff linting |

---

## ✨ Features

### 🔥 ML Pipeline
- **XGBoost Classifier** trained with temporal train/test splits (no future leakage)
- **5 evaluation promotion gates**: recall ≥ 0.90, FPR ≤ 0.10, ROC-AUC ≥ 0.92, precision ≥ 0.70, F1 ≥ 0.80
- **Class imbalance handling** via `scale_pos_weight`
- **Calibrated probabilities** (isotonic regression)
- **Early stopping** with 30-round patience

### 🛡️ Data Quality Firewall (5 Stages)
1. **SchemaGate** — Validates columns, dtypes, required fields
2. **RangeGate** — Per-sensor value bounds from YAML registry
3. **TimestampGate** — Monotonicity, duplicates, gaps, future timestamps
4. **CrossChannelGate** — Redundant sensor pair disagreement detection + flatline detection
5. **DriftGate** — CUSUM statistical process control for gradual calibration drift

### 📡 API
- **8 REST endpoints**: health, readiness, metrics, predict, batch-predict, explain, model list, model promote
- **Async inference** via `ThreadPoolExecutor` (non-blocking event loop)
- **Demo mode** fallback when no model is trained (heuristic predictions)
- **Request ID tracing** across all requests
- **Structured error handling** with typed exception handlers

### 🔬 Explainability
- **SHAP TreeExplainer** — exact (not approximate) feature attribution for XGBoost
- Top-K contributor ranking in API responses
- Operator-facing explanations in safety-critical context

### 📊 Monitoring
- **Evidently AI** drift detection (data drift, model drift)
- **Alert lifecycle management** with deduplication, escalation chains, cooldown
- **Prediction logging** with async PostgreSQL persistence + in-memory fallback
- **Grafana dashboard** for real-time visualization

### 🏗️ Infrastructure
- **Docker Compose** with 6 services: API, TimescaleDB, Redis, MLflow, Grafana, training worker
- **MLflow** experiment tracking and model registry with staging/promotion
- **Pydantic Settings** — type-safe, env-driven configuration
- **Structured logging** — JSON format for production, console for development

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────┐
                         │         DATA INGESTION               │
                         │  [CSV / Synthetic Generator / Redis] │
                         └──────────────┬───────────────────────┘
                                        │
                         ┌──────────────▼───────────────────────┐
                         │     DATA QUALITY FIREWALL (5 gates)  │
                         │                                      │
                         │  ┌──────────┐  ┌──────────────────┐  │
                         │  │SchemaGate│─▶│   RangeGate      │  │
                         │  └──────────┘  └────────┬─────────┘  │
                         │                         │             │
                         │  ┌──────────────────────▼──────────┐  │
                         │  │      TimestampGate              │  │
                         │  └──────────┬───────────────────────┘  │
                         │             │                         │
                         │  ┌──────────▼───────────────────────┐  │
                         │  │    CrossChannelGate              │  │
                         │  └──────────┬───────────────────────┘  │
                         │             │                         │
                         │  ┌──────────▼───────────────────────┐  │
                         │  │      DriftGate (CUSUM)           │  │
                         │  └──────────┬───────────────────────┘  │
                         └──────────────┬───────────────────────┘
                                        │
                         ┌──────────────▼───────────────────────┐
                         │      FEATURE ENGINEERING              │
                         │  [Rolling Windows + FFT + Degradation │
                         │   → sklearn Pipeline]                 │
                         └──────────────┬───────────────────────┘
                                        │
                         ┌──────────────▼───────────────────────┐
                         │    MODEL INFERENCE                    │
                         │  [XGBoost Classifier → SHAP TreeExp.  │
                         │   → Calibrated Probability]           │
                         └──────────────┬───────────────────────┘
                                        │
          ┌─────────────────────────────┬─────────────────────────────┐
          │                             │                             │
┌─────────▼──────────┐     ┌───────────▼──────────┐     ┌───────────▼──────────┐
│   FASTAPI API       │     │   MONITORING STACK   │     │   MODEL MANAGEMENT   │
│  /predict           │     │  Evidently AI Drift  │     │   MLflow Registry    │
│  /explain           │     │  Alert Manager       │     │   Version Promotion  │
│  /models            │     │  Prediction Logger   │     │   Performance Gates  │
│  /health            │     │  Grafana Dashboard   │     │                      │
└─────────┬──────────┘     └───────────┬──────────┘     └───────────┬──────────┘
          │                             │                             │
          └─────────────────────────────┼─────────────────────────────┘
                                        │
                         ┌──────────────▼───────────────────────┐
                         │      DATA PERSISTENCE                │
                         │  [TimescaleDB / Redis Cache /        │
                         │   Feature Store / Logs]              │
                         └──────────────────────────────────────┘
```

### Pipeline Flow

```
Raw Sensor Data
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ DATA QUALITY GATE                                                    │
│  SchemaGate ──▶ RangeGate ──▶ TimestampGate ──▶ CrossGate ──▶ Drift │
│                                                                     │
│  PASS ──────────────────────────────────────────────────▶ Continue   │
│  FAIL ──▶ Warning injected into response ──▶ Still predicts (demo)  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FEATURE PIPELINE                                                     │
│  Pivot to wide format                                                │
│  Rolling statistics (mean, std, min, max) @ windows [5,15,30,60]   │
│  FFT frequency-domain features                                       │
│  Rate-of-change / degradation indicators                             │
│  sklearn Pipeline with fitted transformers                           │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ INFERENCE                                                             │
│  XGBoost predict_proba() → calibrated probability                    │
│  If probability ≥ 0.5 → failure predicted                            │
│  SHAP TreeExplainer computes per-feature contributions               │
│  Top-10 contributors returned in /explain response                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Data Quality Firewall Deep Dive

The quality gate is the heart of this project. It runs **5 sequential validation stages** on every batch of sensor data before the ML model ever sees it.

| Gate | What It Catches | Implementation | Failure Mode |
|------|----------------|----------------|--------------|
| **SchemaGate** | Missing columns, extra columns, wrong dtypes | Pydantic-style column validation + dtype checking | Reports missing/extra columns, dtype mismatches; row/column counts in metadata |
| **RangeGate** | Out-of-bounds values (temp > 500°C) | Vectorized pivot + per-sensor range lookup from `sensor_registry.yaml` | Tracks violation fraction per sensor; reports worst offender |
| **TimestampGate** | Gaps > 10min, duplicates, future dates, non-monotonic ordering | Grouped monotonicity check, duplicate detection, gap analysis | Flags affected equipment, gap locations, duplicate counts |
| **CrossChannelGate** | Redundant sensor disagreement, stuck/flatlined sensors | Pairs from registry; computes expected noise std from range width; statistical threshold at 3σ; flatline detection via unique value count | Reports mismatch rate, flatline conflicts, mean/max diff per pair |
| **DriftGate** | Gradual calibration shift (0.01%/day) | CUSUM (Cumulative Sum) SPC: tracks C+ / C- accumulations against decision interval H=50, reference value k=5 | Alerts on drift direction, cumulative values; auto-resets after detection |

### Key Design Choices
- **Non-blocking stages**: `DriftGate` is non-blocking by default (information-only)
- **Stop-on-first-failure**: Configurable; when enabled, fails fast without running downstream gates
- **Per-sensor, per-equipment CUSUM detectors**: Stateful — drift state persists across batches
- **Calibration API**: `set_reference_statistics()` allows recalibration after maintenance
- **Sensor registry driven**: Zero hardcoded sensor names or thresholds

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker** & **Docker Compose** (for full platform)
- **Git**
- **Make** (optional, but convenient)

### One-Command Setup (Full Platform)

```bash
# 1. Clone the repository
git clone https://github.com/privo211/predictive-maintenance-ml.git
cd predictive-maintenance-ml

# 2. Generate synthetic sensor data (50 equipment units, 90 days)
make generate-data

# 3. Train the XGBoost model (optional — API works in demo mode without it)
make train

# 4. Start all 6 services
make docker-up

# 5. Access the platform
#    API Docs:     http://localhost:8000/docs
#    MLflow UI:    http://localhost:5000
#    Grafana:      http://localhost:3000  (admin/admin)
```

**In demo mode** (no trained model), the API uses heuristic predictions based on equipment health signals. Train a model with `make train` for ML-powered predictions.

### Step-by-Step Breakdown

```bash
# Generate data with custom parameters
python scripts/generate_data.py \
    --output-dir data \
    --equipment-count 100 \
    --days 180 \
    --failure-rate 0.03 \
    --seed 42

# Train with custom config
python scripts/train_model.py \
    --config config/model_config.yaml \
    --data-path data/sensor_data.csv \
    --output-dir models

# Serve the API
python scripts/serve_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

### Available Make Commands

```bash
make setup           # Bootstrap project (install deps, pre-commit hooks)
make generate-data   # Generate synthetic sensor data
make train           # Train XGBoost failure prediction model
make serve           # Start FastAPI inference server
make test            # Run full test suite with coverage
make lint            # Run all linters (ruff + mypy)
make docker-up       # Start all services via Docker Compose
make docker-down     # Stop and remove all Docker services
make clean           # Remove build artifacts and caches
```

---

## 💻 Local Development (Without Docker)

```bash
# 1. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install package in development mode
pip install -e ".[dev]"

# 3. Copy environment config
cp .env.example .env
# Edit .env if needed — defaults work for local development

# 4. Generate sample data
python scripts/generate_data.py

# 5. Train model
python scripts/train_model.py

# 6. Run tests
make test

# 7. Start API server
make serve
# API available at http://localhost:8000/docs
```

---

## 📖 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|-------------|----------|
| `GET` | `/health` | Kubernetes liveness probe | — | `{"status": "healthy", "timestamp": "..."}` |
| `GET` | `/ready` | Readiness probe (model loaded?) | — | `{"status": "ready", "model_loaded": true, ...}` |
| `GET` | `/metrics` | Runtime counters | — | `{"uptime_seconds": 123, "prediction_count": 42, ...}` |
| `POST` | `/api/v1/predict` | Single-equipment failure prediction | `PredictionRequest` | `PredictionResponse` |
| `POST` | `/api/v1/batch-predict` | Batch prediction (up to 50 equipment) | `BatchPredictionRequest` | `BatchPredictionResponse` |
| `POST` | `/api/v1/explain` | Prediction + SHAP explanation | `ExplainRequest` | `ExplanationResponse` |
| `GET` | `/api/v1/models` | List registered model versions | — | `ModelListResponse` |
| `POST` | `/api/v1/models/{version}/promote` | Promote model to Production | `ModelPromoteRequest` | `PromoteResponse` |

### Example: Predict

```bash
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {"timestamp": "2026-01-01T00:00:00Z", "equipment_id": "PUMP-001", "sensor_name": "vibration_x", "value": 2.8},
      {"timestamp": "2026-01-01T00:00:01Z", "equipment_id": "PUMP-001", "sensor_name": "vibration_x", "value": 2.9},
      {"timestamp": "2026-01-01T00:00:02Z", "equipment_id": "PUMP-001", "sensor_name": "temperature", "value": 75.0}
    ]
  }' | python3 -m json.tool
```

**Response:**
```json
{
  "equipment_id": "PUMP-001",
  "failure_probability": 0.12,
  "predicted_class": 0,
  "prediction_threshold": 0.5,
  "model_version": null,
  "demo_mode": true,
  "quality_gate": {
    "passed": true,
    "total_violations": 0,
    "failing_stages": [],
    "summary": "PASSED: 0 violations across 5 stages."
  },
  "data_quality_warnings": [],
  "timestamp": "2026-01-01T00:00:05.123456",
  "request_id": "req_abc123"
}
```

### Example: Explain (with SHAP)

```bash
curl -s -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {"timestamp": "2026-01-01T00:00:00Z", "equipment_id": "PUMP-001", "sensor_name": "vibration_x", "value": 2.8},
      {"timestamp": "2026-01-01T00:00:01Z", "equipment_id": "PUMP-001", "sensor_name": "vibration_x", "value": 2.9}
    ]
  }' | python3 -m json.tool
```

**Response (excerpt):**
```json
{
  "equipment_id": "PUMP-001",
  "failure_probability": 0.12,
  "predicted_class": 0,
  "base_value": -1.85,
  "top_contributors": [
    {"feature": "vibration_x_rolling_mean_10", "shap_value": 0.45, "direction": "increases_failure_risk", "feature_value": 2.85},
    {"feature": "temperature_rolling_std_10", "shap_value": -0.32, "direction": "decreases_failure_risk", "feature_value": 0.12}
  ],
  "model_version": "failure_predictor-v1.0.0",
  "demo_mode": false,
  "timestamp": "2026-01-01T00:00:05.123456"
}
```

---

## 📁 Project Structure

```
predictive-maintenance-ml/
│
├── config/                          # Configuration (env-driven + YAML)
│   ├── settings.py                  #   Pydantic Settings (PMP_ env prefix)
│   ├── model_config.yaml            #   XGBoost hyperparameters + eval gates
│   ├── sensor_registry.yaml         #   3 equipment types, 30+ sensors, redundant pairs
│   ├── alerting_rules.yaml          #   8 alert rules with escalation chains
│   └── logging.yaml                 #   Structured logging (JSON/console)
│
├── src/                             # Source code
│   ├── data/                        #   Data Layer (2,818 lines)
│   │   ├── synthetic_generator.py   #     Physics-informed generator (3 equipment types, 6 faults)
│   │   ├── quality_gate.py          #     5-stage DataQualityFirewall
│   │   ├── quality_checks.py        #     Low-level validation functions
│   │   ├── drift_detector.py        #     CUSUM statistical process control
│   │   └── degradation_models.py    #     Equipment degradation physics
│   │
│   ├── features/                    #   Feature Engineering (983 lines)
│   │   ├── feature_pipeline.py      #     sklearn Pipeline (rolling windows, FFT, stats)
│   │   ├── feature_store.py         #     SQLite-backed feature cache
│   │   └── degradation_features.py  #     Trend/health indicator extraction
│   │
│   ├── models/                      #   Model Layer (1,247 lines)
│   │   ├── train.py                 #     XGBoost training with MLflow tracking
│   │   ├── evaluate.py              #     Metrics + promotion gates
│   │   └── registry.py              #     MLflow model registry wrapper
│   │
│   ├── explainability/              #   SHAP Explainability (231 lines)
│   │   └── shap_explainer.py        #     Exact TreeExplainer for XGBoost
│   │
│   ├── api/                         #   FastAPI Application (2,068 lines)
│   │   ├── main.py                  #     App factory with async lifespan
│   │   ├── routers/                  #     predict, explain, health, models
│   │   ├── schemas/                  #     Pydantic v2 request/response models
│   │   ├── dependencies.py          #     FastAPI dependency injection
│   │   ├── errors.py                #     Typed exception handlers
│   │   └── middleware.py            #     Request ID tracing
│   │
│   ├── monitoring/                  #   Monitoring (1,127 lines)
│   │   ├── alerting.py              #     Alert lifecycle + dedup + escalation
│   │   ├── evidently_reporter.py    #     Evidently AI drift reports
│   │   └── prediction_logger.py     #     Async prediction/feedback logging
│   │
│   └── utils/                       #   Utilities
│       ├── database.py              #     Async SQLAlchemy engine + session
│       └── logger.py                #     Structured logging setup
│
├── tests/                           # Test Suite (1,543 lines)
│   ├── conftest.py                  #   Fixtures (session-scoped generators, models)
│   ├── unit/                        #   Quick unit tests (quality gate, model, features)
│   ├── integration/                 #   API + pipeline integration tests
│   └── validation/                  #   Data quality + model performance validation
│
├── scripts/                         # CLI Entry Points
│   ├── generate_data.py             #   Synthetic data generation
│   ├── train_model.py               #   End-to-end training pipeline
│   └── serve_api.py                 #   Uvicorn API server
│
├── docker/                          # Containerization
│   ├── api.Dockerfile               #   FastAPI production image
│   ├── training.Dockerfile          #   Training job image
│   └── postgres/init.sql            #   TimescaleDB initialization
│
├── dashboards/grafana/              # Grafana provisioning
├── docker-compose.yml               # 6 services, one command
├── pyproject.toml                   # Project metadata + tool config
├── Makefile                         # 16 automation targets
└── README.md                        # This file
```

---

## 📊 Model Performance

| Metric | Target | Enforcement |
|--------|--------|-------------|
| **Recall** (failure class) | ≥ 0.90 | Gate — blocks promotion if below |
| **False Positive Rate** | ≤ 0.10 | Gate — blocks promotion if above |
| **ROC-AUC** | ≥ 0.92 | Gate — blocks promotion if below |
| **Precision** | ≥ 0.70 | Gate — blocks promotion if below |
| **F1 Score** | ≥ 0.80 | Gate — blocks promotion if below |
| **PR-AUC** | ≥ 0.85 | Gate — blocks promotion if below |
| **Inference Latency (p95)** | < 300ms | Monitored — alert if exceeded |
| **Data Quality Pass Rate** | > 98% | Monitored — alert if exceeded |

**Promotion gates**: New model versions are automatically blocked from production deployment if any metric falls below its threshold. This prevents regressions from reaching operators.

---

## 🧪 Testing

The project uses `pytest` with three test categories:

```bash
# Run full suite with coverage
make test

# Run specific categories
make test-unit          # Quality gate, feature pipeline, models
make test-integration   # API endpoints, pipeline integration
make test-validation    # Data quality, model performance

# View HTML coverage report
open htmlcov/index.html
```

### Test Structure

| File | Category | Tests |
|------|----------|-------|
| `tests/unit/test_quality_gate.py` | Unit | SchemaGate, RangeGate, TimestampGate, CrossChannelGate, DriftGate, QualityReport |
| `tests/unit/test_feature_pipeline.py` | Unit | Feature extraction, pipeline transform |
| `tests/unit/test_models.py` | Unit | Training, evaluation metrics, registry |
| `tests/integration/test_api.py` | Integration | All 8 API endpoints, demo mode, error cases |
| `tests/integration/test_pipeline.py` | Integration | End-to-end data→features→prediction flow |
| `tests/validation/test_data_quality.py` | Validation | Data completeness, range compliance |
| `tests/validation/test_model_performance.py` | Validation | Model meets promotion gate thresholds |

### Fixtures

The `conftest.py` provides session-scoped fixtures for:
- `SyntheticDataGenerator` with fixed seed (reproducible)
- Pre-fitted `FeaturePipeline` 
- Pre-trained `XGBClassifier` (20 estimators, fast)
- `TestClient` with model injected into app state
- `DataQualityGate` loaded from real sensor registry

### Code Quality

- **Type checking**: mypy strict mode (96/100 strictness)
- **Linting**: ruff with 7 rule sets (pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade, simplify)
- **Formatting**: ruff formatter (double quotes, 100 char lines)
- **Coverage**: pytest-cov with HTML report generation

---

## 🧠 Design Decisions

### Why XGBoost and not deep learning?

1. **SHAP TreeExplainer is exact** (not approximate) for tree-based models — operator-facing explanations must be correct in safety-critical industrial settings
2. **Interpretability > marginal accuracy** — plant operators need to trust predictions, not just see them
3. **Fast training** on moderate-sized datasets without GPU requirements — this needs to run on-premises
4. The model architecture is the least important part — the *system around the model* (quality gates, monitoring, retraining triggers) is what determines real-world success

### Why temporal split and not random?

In predictive maintenance, random train/test splits **leak future data** into training. A model trained on next week's data to predict this week's failure is cheating. Temporal splits respect causality — the model only sees past data to predict future failures. This is the #1 evaluation mistake in industrial ML.

### Why 5-stage quality gate and not just null checks?

- Industrial sensors fail in subtle ways: calibration drift, stuck values, intermittent dropout — null checks only catch missing data
- Redundant sensors (standard in critical power generation installations) provide powerful cross-validation that single-sensor checks can't
- CUSUM catches gradual drift (0.01%/day) that simple threshold checks would miss for weeks
- A quality gate failure doesn't crash the system — it downgrades the prediction confidence with a data quality warning

### Why Docker Compose and not serverless?

- Industrial environments run on-premises with air-gapped networks — no cloud access
- Docker Compose provides **one-command reviewability**: a recruiter or engineer can evaluate the entire system in 2 minutes
- The same Compose file works in development, CI, and production

### Why synthetic data?

- Real industrial sensor data is proprietary and safety-regulated — no public datasets exist for equipment failure prediction
- The synthetic generator is **physics-informed**: it models actual degradation trajectories, not random noise
- This means the ML pipeline, quality gates, and monitoring all experience realistic failure patterns

---

## 📝 What I Learned

### What Worked

- **Tree-based models (XGBoost)** were fast enough for exact SHAP explanations — inference in <50ms for single samples
- **CUSUM caught gradual sensor drift** that range checks missed entirely — 0.01% daily drift took weeks to exceed range thresholds
- **Docker Compose** made the project reviewable in 2 minutes — this is the single best decision for portfolio projects
- **In-memory fallbacks** for database-dependent components enabled local development without infrastructure
- **Pydantic v2 validators** at the API boundary caught data issues before they reached the ML pipeline

### What Didn't Work

- **LSTM autoencoders** for anomaly detection overfit on small synthetic datasets — classic unsupervised deep learning problem
- **SHAP waterfall plots in React** required too much frontend work for a solo project — server-rendered JSON with top-10 contributors is sufficient
- **Trying to cover all 120+ features** with SHAP explanations was noisy — top-10 contributors by |SHAP| is the right level for operator consumption
- **Real-time MQTT/OPC-UA ingestion** added significant complexity without proportional value — CSV batch processing is more practical for portfolio demonstration

---

## 📄 License

MIT — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built as a demonstration of production ML engineering principles for industrial predictive maintenance.</em>
  <br>
  <em>Designed to be reviewable, runnable, and impressive — not just another Kaggle notebook with a pretty README.</em>
  <br><br>
  <strong>Priyanshu Vora</strong> ·
  <a href="https://github.com/privo211">GitHub</a> ·
  <a href="https://www.linkedin.com/in/priyanshuvora">LinkedIn</a> ·
  <a href="https://priyanshu-vora.vercel.app/">Portfolio</a>
</p>
