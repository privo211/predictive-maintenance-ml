"""Pydantic response models for the predictive maintenance API.

All response models use Pydantic v2 and return structured, self-describing
JSON suitable for downstream monitoring dashboards and alerting pipelines.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field


class QualityGateInfo(BaseModel):
    """Summary of quality gate validation results for a single prediction."""

    passed: Annotated[
        bool,
        Field(description="Whether all quality gate stages passed"),
    ]
    total_violations: Annotated[
        int,
        Field(description="Total number of quality violations across all stages"),
    ]
    failing_stages: Annotated[
        list[str],
        Field(
            description="Names of stages that failed (empty if all passed)",
        ),
    ]
    summary: Annotated[
        str,
        Field(description="Human-readable quality gate summary"),
    ]


class DataQualityWarning(BaseModel):
    """Non-blocking data quality advisory attached to prediction responses."""

    severity: Annotated[
        str,
        Field(
            description="Warning severity level",
            examples=["low", "medium", "high"],
        ),
    ]
    message: Annotated[
        str,
        Field(description="Human-readable warning message"),
    ]
    affected_sensors: Annotated[
        list[str],
        Field(
            description="Sensor names affected by the quality issue",
        ),
    ]


class PredictionResponse(BaseModel):
    """Response for a single-equipment failure prediction."""

    equipment_id: Annotated[
        str,
        Field(description="Equipment unit identifier"),
    ]
    failure_probability: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Predicted probability of failure (0.0 to 1.0)",
        ),
    ]
    predicted_class: Annotated[
        int,
        Field(
            ge=0,
            le=1,
            description="Predicted class: 0=healthy, 1=failure imminent",
        ),
    ]
    prediction_threshold: Annotated[
        float,
        Field(description="Threshold used for classification"),
    ]
    model_version: Annotated[
        str | None,
        Field(
            description="Model version used (None in demo mode)",
        ),
    ]
    demo_mode: Annotated[
        bool,
        Field(
            description="Whether prediction was made in demo/fallback mode",
        ),
    ]
    quality_gate: Annotated[
        QualityGateInfo | None,
        Field(
            description="Quality gate validation results (None if skipped)",
        ),
    ]
    data_quality_warnings: Annotated[
        list[DataQualityWarning],
        Field(description="Non-blocking data quality warnings"),
    ]
    timestamp: Annotated[
        str,
        Field(
            description="ISO-8601 timestamp of the prediction",
        ),
    ]
    request_id: Annotated[
        str | None,
        Field(description="X-Request-ID for tracing"),
    ]


class FeatureContribution(BaseModel):
    """Single feature's contribution to a prediction, ordered by SHAP magnitude."""

    feature: Annotated[
        str,
        Field(description="Feature name"),
    ]
    shap_value: Annotated[
        float,
        Field(description="SHAP value for this feature"),
    ]
    direction: Annotated[
        str,
        Field(
            description="Direction of the contribution",
            examples=["increases_failure_risk", "decreases_failure_risk"],
        ),
    ]
    feature_value: Annotated[
        float,
        Field(description="Actual feature value used in prediction"),
    ]


class ExplanationResponse(BaseModel):
    """Response for a prediction with SHAP-based explanation."""

    equipment_id: Annotated[
        str,
        Field(description="Equipment unit identifier"),
    ]
    failure_probability: Annotated[
        float,
        Field(ge=0.0, le=1.0),
    ]
    predicted_class: Annotated[
        int,
        Field(ge=0, le=1),
    ]
    base_value: Annotated[
        float,
        Field(description="SHAP expected (base) model output in log-odds space"),
    ]
    shap_values: Annotated[
        list[float],
        Field(description="SHAP values for all features"),
    ]
    top_contributors: Annotated[
        list[FeatureContribution],
        Field(description="Top feature contributors ordered by |SHAP| magnitude"),
    ]
    model_version: Annotated[
        str | None,
        Field(description="Model version used (None in demo mode)"),
    ]
    demo_mode: Annotated[
        bool,
        Field(description="Whether prediction was made in demo/fallback mode"),
    ]
    timestamp: Annotated[
        str,
        Field(description="ISO-8601 timestamp of the explanation"),
    ]
    request_id: Annotated[
        str | None,
        Field(description="X-Request-ID for tracing"),
    ]


class BatchPredictionItem(BaseModel):
    """Single equipment result within a batch prediction response."""

    equipment_id: str
    failure_probability: float
    predicted_class: int
    data_quality_warnings: list[DataQualityWarning]


class BatchPredictionResponse(BaseModel):
    """Response for batch failure predictions across multiple equipment."""

    results: Annotated[
        list[BatchPredictionItem],
        Field(description="Per-equipment prediction results"),
    ]
    total_equipment: Annotated[
        int,
        Field(description="Number of equipment units in the batch"),
    ]
    failures_detected: Annotated[
        int,
        Field(description="Number of equipment predicted as failure imminent"),
    ]
    model_version: Annotated[
        str | None,
        Field(description="Model version used (None in demo mode)"),
    ]
    demo_mode: Annotated[
        bool,
        Field(description="Whether predictions were made in demo/fallback mode"),
    ]
    timestamp: Annotated[
        str,
        Field(description="ISO-8601 timestamp of the batch prediction"),
    ]
    request_id: Annotated[
        str | None,
        Field(description="X-Request-ID for tracing"),
    ]


class ModelInfo(BaseModel):
    """Information about a registered model version."""

    version: Annotated[
        str,
        Field(description="Model version number"),
    ]
    stage: Annotated[
        str,
        Field(description="Current stage (e.g., Production, Staging, Archived)"),
    ]
    run_id: Annotated[
        str | None,
        Field(description="MLflow run ID"),
    ]
    status: Annotated[
        str | None,
        Field(description="Version status (READY, PENDING_REGISTRATION, etc.)"),
    ]
    creation_timestamp: Annotated[
        int | None,
        Field(description="Unix millisecond timestamp of version creation"),
    ]
    metrics: Annotated[
        dict[str, float] | None,
        Field(description="Evaluation metrics for this version"),
    ]


class ModelListResponse(BaseModel):
    """Response listing all versions of a registered model."""

    model_name: Annotated[
        str,
        Field(description="Registered model name"),
    ]
    versions: Annotated[
        list[ModelInfo],
        Field(description="All registered versions"),
    ]
    current_production: Annotated[
        str | None,
        Field(description="Version currently in Production (None if none)"),
    ]


class PromoteResponse(BaseModel):
    """Response after a model promotion request."""

    model_name: str
    version: str
    previous_stage: str
    new_stage: Annotated[
        str,
        Field(default="Production"),
    ]
    message: str


class HealthResponse(BaseModel):
    """Liveness probe response.

    Always returns 200 OK as long as the process is running. This is a
    lightweight endpoint suitable for Kubernetes liveness probes.
    """

    status: Annotated[
        str,
        Field(default="healthy"),
    ]
    timestamp: Annotated[
        str,
        Field(description="ISO-8601 timestamp of the health check"),
    ]


class ReadinessResponse(BaseModel):
    """Readiness probe response.

    Reports whether the model is loaded and the server is ready to accept
    prediction requests. Returns 503 if the model is unavailable (but the
    server can still serve in demo mode).
    """

    status: Annotated[
        str,
        Field(description="'ready' or 'degraded'"),
    ]
    model_loaded: Annotated[
        bool,
        Field(description="Whether the ML model is loaded in memory"),
    ]
    demo_mode: Annotated[
        bool,
        Field(description="Whether the server is operating in demo/fallback mode"),
    ]
    uptime_seconds: Annotated[
        float,
        Field(description="Server uptime in seconds"),
    ]
    model_version: Annotated[
        str | None,
        Field(description="Loaded model version (None if not loaded)"),
    ]
    timestamp: Annotated[
        str,
        Field(description="ISO-8601 timestamp of the readiness check"),
    ]


class MetricsResponse(BaseModel):
    """API server metrics response.

    Exposes runtime metrics including uptime, request counts, and error
    counters for integration with Prometheus or custom monitoring.
    """

    uptime_seconds: float
    model_loaded: bool
    demo_mode: bool
    model_version: str | None
    feature_count: Annotated[
        int | None,
        Field(description="Number of features in the pipeline (None if not fitted)"),
    ]
    prediction_count: Annotated[
        int,
        Field(description="Total prediction requests served since startup"),
    ]
    error_count: Annotated[
        int,
        Field(description="Total error responses since startup"),
    ]
    timestamp: str
