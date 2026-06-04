# Industrial Predictive Maintenance Platform

**Production-grade ML system for equipment failure prediction with automated data quality gates.**

Industrial ML models don't fail because of bad algorithms. They fail because of bad data. Sensors drift. Networks drop packets. Maintenance teams swap hardware without updating the data pipeline. Your 99% accurate model from training becomes a random number generator in production — and nobody notices until a $400K transformer fails.

This project demonstrates **production ML engineering**, not Kaggle ML. It answers: *"How do you know your predictions are still trustworthy at 3 AM?"*

---

## What Makes This Different From Every Other PdM Project

| Most PdM Projects | This Project |
|---|---|
| `model.fit()` → "95% accuracy!" → Jupyter notebook | Data quality firewall → drift monitoring → calibrated predictions → SHAP explanations → Docker deployment |
| Random train/test split | Temporal split — no future data leakage |
| No data validation | 5-stage quality gate: schema, range, timestamp, cross-channel, CUSUM drift |
| Single sensor | Multi-sensor fusion with redundant channel validation |
| "It works on my laptop" | `docker compose up` — 6 services, one command |
| Accuracy without context | Recall, precision, FPR, ROC-AUC with promotion gates |

---

## Architecture

```
                    DATA INGESTION
   [CSV / Redis Streams / Synthetic Generator]
                         |
              DATA QUALITY FIREWALL (5 gates)
   [Schema → Range → Timestamp → Cross-Channel → CUSUM Drift]
                         |
              FEATURE ENGINEERING
   [Rolling windows + FFT + Degradation → sklearn Pipeline]
                         |
              MODEL INFERENCE (ThreadPoolExecutor)
   [XGBoost Classifier → SHAP TreeExplainer → Calibrated Probability]
                         |
              API & MONITORING
   [FastAPI → TimescaleDB → Evidently AI → Grafana → AlertManager]
```

### Data Quality Firewall

| Gate | What It Catches | Method |
|------|----------------|--------|
| **Schema** | Missing/extra columns, wrong dtypes | Pydantic-style validation |
| **Range** | Impossible values (temp > 500C on a bearing) | Per-sensor registry from YAML |
| **Timestamp** | Gaps, duplicates, future timestamps, non-monotonic | Monotonicity + gap detection |
| **Cross-Channel** | Sensor A says failure, redundant Sensor B says normal | Redundancy matrix from sensor registry |
| **Drift** | Gradual calibration shift (0.01%/day) | CUSUM statistical process control |

---

## Stack

| Component | Technology |
|---|---|
| API | FastAPI + Uvicorn (async) |
| ML | XGBoost + scikit-learn |
| Experiment Tracking | MLflow |
| Database | PostgreSQL + TimescaleDB |
| Monitoring | Evidently AI (drift detection) |
| Explainability | SHAP TreeExplainer (exact) |
| Dashboard | Grafana |
| Container | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |
| Config | Pydantic Settings (env-driven) |

---

## Quick Start

```bash
# Clone and start everything
git clone https://github.com/[username]/industrial-predictive-maintenance
cd industrial-predictive-maintenance

# Generate sample data
make generate-data

# Train the model (optional — API works in demo mode without it)
make train

# Start the full platform
make docker-up

# Access points:
#   API:      http://localhost:8000/docs
#   MLflow:   http://localhost:5000
#   Grafana:  http://localhost:3000  (admin/admin)
```

**In demo mode** (no trained model), the API uses heuristic predictions based on equipment health signals. Train a model with `make train` for ML-powered predictions.

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Kubernetes liveness probe |
| `GET` | `/ready` | Readiness probe (model loaded + uptime) |
| `GET` | `/metrics` | Runtime counters |
| `POST` | `/api/v1/predict` | Single-equipment failure prediction |
| `POST` | `/api/v1/batch-predict` | Batch prediction (up to 50 equipment) |
| `POST` | `/api/v1/explain` | Prediction + SHAP explanation |
| `GET` | `/api/v1/models` | List model versions |
| `POST` | `/api/v1/models/{version}/promote` | Promote model to Production |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "PUMP-001",
    "readings": [
      {"timestamp": "2026-01-01T00:00:00Z", "equipment_id": "PUMP-001", "sensor_name": "vibration_x", "value": 2.8},
      {"timestamp": "2026-01-01T00:00:01Z", "equipment_id": "PUMP-001", "sensor_name": "vibration_x", "value": 2.9}
    ]
  }'
```

### Example Response

```json
{
  "equipment_id": "PUMP-001",
  "failure_probability": 0.12,
  "predicted_class": 0,
  "failure_horizon_hours": null,
  "model_version": "failure_predictor-v1.0.0",
  "inference_latency_ms": 45.3,
  "quality_gate": {
    "passed": true,
    "stages": {
      "SchemaGate": {"passed": true, "violations": 0},
      "RangeGate": {"passed": true, "violations": 0},
      "TimestampGate": {"passed": true, "violations": 0},
      "CrossChannelGate": {"passed": true, "violations": 0},
      "DriftGate": {"passed": true, "violations": 0}
    }
  },
  "data_quality_warning": false,
  "warning_message": null
}
```

---

## Project Structure

```
predictive-maintenance-platform/
├── config/
│   ├── settings.py              # Pydantic Settings (env-driven)
│   ├── model_config.yaml        # XGBoost hyperparameters + eval gates
│   ├── sensor_registry.yaml     # Sensor specs, ranges, redundant pairs
│   ├── alerting_rules.yaml      # Alert thresholds + escalation
│   └── logging.yaml            # Structured logging config
├── src/
│   ├── data/                    # Synthetic generator + quality gate
│   ├── features/                # sklearn Pipeline + feature store
│   ├── models/                  # Training + evaluation + registry
│   ├── api/                     # FastAPI application
│   ├── monitoring/              # Evidently drift + alerts + prediction log
│   ├── explainability/          # SHAP TreeExplainer
│   ├── training/                # Training orchestration
│   └── utils/                   # Database + logging utilities
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── validation/              # Model + data validation tests
├── docker/                      # Dockerfiles + DB init
├── dashboards/grafana/          # Grafana dashboard JSON
├── scripts/                     # Training + data generation scripts
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

## Model Performance

| Metric | Target | Status |
|--------|--------|--------|
| Recall (failure class) | >= 0.90 | Gate-enforced |
| False Positive Rate | <= 0.10 | Gate-enforced |
| ROC-AUC | >= 0.92 | Gate-enforced |
| Inference Latency (p95) | < 300ms | Monitored |
| Data Quality Pass Rate | > 98% | Monitored |

**Promotion gates**: New model versions are automatically blocked from production deployment if any metric falls below its threshold.

---

## Design Decisions

### Why XGBoost and not deep learning?

1. **SHAP TreeExplainer is exact** (not approximate) for tree-based models — operator-facing explanations must be correct
2. **Interpretability > marginal accuracy** in safety-critical industrial settings
3. **Fast training** on moderate-sized datasets without GPU requirements
4. The model architecture is the least important part — the *system around the model* (quality gates, monitoring, retraining triggers) is what matters

### Why temporal split and not random?

In predictive maintenance, random train/test splits **leak future data** into training. A model that sees next week's data to predict this week's failure is cheating. Temporal splits respect causality — and prevent the #1 evaluation mistake in industrial ML.

### Why 5-stage quality gate and not just null checks?

- Industrial sensors fail in subtle ways: calibration drift, stuck values, intermittent dropout
- Redundant sensors (standard in critical installations) provide cross-validation
- CUSUM catches gradual drift that simple threshold checks miss
- A quality gate failure doesn't crash the system — it downgrades the prediction confidence

---

## What I Learned

### What Worked

- Tree-based models (XGBoost) were fast enough for exact SHAP explanations
- CUSUM caught gradual sensor drift that range checks missed entirely
- Docker Compose made the project reviewable in 2 minutes
- In-memory fallbacks for database-dependent components enabled local development

### What Didn't Work

- LSTM autoencoders for anomaly detection overfit on small synthetic datasets
- SHAP waterfall plots in React required too much frontend work for a weekend — server-rendered JSON is sufficient
- Trying to cover all 120+ features with SHAP explanations was noisy — top-10 contributors is the right level
- Initial plan to include real-time MQTT/OPC-UA ingestion added complexity without proportional value

---

## License

MIT

---

*Built as a demonstration of production ML engineering principles for industrial predictive maintenance. Designed to be reviewable, runnable, and impressive — not just another Kaggle notebook with a pretty README.*
