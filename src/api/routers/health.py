"""Health check and metrics endpoints for Kubernetes probes and monitoring.

Provides liveness (is the process alive?), readiness (can we serve
predictions?), and a lightweight metrics endpoint for monitoring.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from api.schemas.responses import HealthResponse, MetricsResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Kubernetes liveness probe — always returns 200 if the process is alive."""
    from datetime import datetime, timezone

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(request: Request) -> ReadinessResponse:
    """Kubernetes readiness probe — reports model load status and uptime."""
    from datetime import datetime, timezone

    start_time = request.app.state.start_time
    uptime_seconds = time.perf_counter() - start_time

    model_loaded = request.app.state.model is not None
    demo_mode = request.app.state.demo_mode

    model_version = None
    if model_loaded:
        model_version = getattr(request.app.state, "model_version", None)

    status = "degraded" if demo_mode else "ready"

    return ReadinessResponse(
        status=status,
        model_loaded=model_loaded,
        demo_mode=demo_mode,
        uptime_seconds=round(uptime_seconds, 3),
        model_version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    """Lightweight metrics endpoint exposing runtime counters and state."""
    from datetime import datetime, timezone

    start_time = request.app.state.start_time
    uptime_seconds = time.perf_counter() - start_time

    model_loaded = request.app.state.model is not None
    demo_mode = request.app.state.demo_mode
    model_version = getattr(request.app.state, "model_version", None)
    prediction_count = getattr(request.app.state, "prediction_count", 0)
    error_count = getattr(request.app.state, "error_count", 0)

    feature_count = None
    pipeline = request.app.state.feature_pipeline
    if pipeline is not None:
        try:
            feature_count = len(pipeline.get_feature_names())
        except Exception:
            feature_count = None

    return MetricsResponse(
        uptime_seconds=round(uptime_seconds, 3),
        model_loaded=model_loaded,
        demo_mode=demo_mode,
        model_version=model_version,
        feature_count=feature_count,
        prediction_count=prediction_count,
        error_count=error_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
