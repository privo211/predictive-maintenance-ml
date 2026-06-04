"""Prediction endpoints — single and batch failure probability inference.

POST /api/v1/predict       — single equipment prediction
POST /api/v1/batch-predict — up to 50 equipment in parallel

Quality gate validation runs first. If the model is not loaded, the
endpoint falls back to demo mode using a threshold on the health column.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request

from api.dependencies import (
    get_feature_pipeline,
    get_model,
    get_quality_gate,
    get_thread_pool,
)
from api.errors import DataQualityException, ModelNotLoadedException
from api.schemas.requests import BatchPredictionRequest, PredictionRequest
from api.schemas.responses import (
    BatchPredictionItem,
    BatchPredictionResponse,
    DataQualityWarning,
    PredictionResponse,
    QualityGateInfo,
)

router = APIRouter(prefix="/api/v1", tags=["predictions"])
logger = logging.getLogger(__name__)


def _readings_to_dataframe(readings: list) -> pd.DataFrame:
    """Convert a list of SensorReading pydantic models to a DataFrame."""
    records = [
        {
            "timestamp": r.timestamp,
            "equipment_id": r.equipment_id,
            "sensor_name": r.sensor_name,
            "value": r.value,
        }
        for r in readings
    ]
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _run_quality_gate(quality_gate, df: pd.DataFrame) -> tuple[QualityGateInfo, list[DataQualityWarning]]:
    """Run all quality gate stages and convert the report to API models."""
    if quality_gate is None:
        return QualityGateInfo(passed=True, total_violations=0, failing_stages=[], summary="Quality gate not configured"), []

    report = quality_gate.validate(df)

    failing_stages = [
        name for name, result in report.stage_results.items() if not result.passed
    ]

    gate_info = QualityGateInfo(
        passed=report.passed,
        total_violations=report.total_violations,
        failing_stages=failing_stages,
        summary=report.summary,
    )

    warnings: list[DataQualityWarning] = []
    if not report.passed:
        severity = "high" if report.total_violations > 10 else "medium"
        for stage_name, result in report.stage_results.items():
            if not result.passed:
                for warning_msg in result.warnings[:3]:
                    warnings.append(
                        DataQualityWarning(
                            severity=severity,
                            message=f"[{stage_name}] {warning_msg}",
                            affected_sensors=[],
                        )
                    )

    return gate_info, warnings


def _demo_mode_predict(df: pd.DataFrame, equipment_id: str, threshold: float) -> tuple[float, int]:
    """Demo mode: predict failure based on health column heuristics.

    If the 'health' column exists, use it directly. Otherwise, compute
    a synthetic health score from vibration and temperature sensors.

    Args:
        df: DataFrame with sensor readings.
        equipment_id: Equipment identifier.
        threshold: Health threshold below which failure is predicted.

    Returns:
        Tuple of (failure_probability, predicted_class).
    """
    if "health" in df.columns:
        health_values = df["health"].dropna()
        if len(health_values) > 0:
            latest_health = float(health_values.iloc[-1])
            failure_probability = 1.0 - latest_health
            predicted_class = 1 if latest_health < threshold else 0
            return failure_probability, predicted_class

    vibration_sensors = [c for c in df.columns if c.startswith("vibration_") and c in df.columns]
    temperature_sensors = [c for c in df.columns if c.startswith("temperature_") and c in df.columns]

    vib_score = 0.0
    temp_score = 0.0
    n_vib = 0
    n_temp = 0

    for col in vibration_sensors:
        vals = df[col].dropna()
        if len(vals) > 0:
            vib_score += float(vals.mean())
            n_vib += 1

    for col in temperature_sensors:
        vals = df[col].dropna()
        if len(vals) > 0:
            temp_score += float(vals.mean())
            n_temp += 1

    if n_vib == 0 and n_temp == 0:
        failure_probability = 0.01
        return failure_probability, 0

    if n_vib > 0:
        vib_score /= n_vib
    if n_temp > 0:
        temp_score /= n_temp

    estimated_health = max(0.0, min(1.0, 1.0 - (vib_score / 15.0) * 0.5 - (temp_score / 120.0) * 0.3))
    estimated_health = max(0.0, min(1.0, estimated_health))

    failure_probability = 1.0 - estimated_health
    predicted_class = 1 if estimated_health < threshold else 0

    return failure_probability, predicted_class


def _do_inference_sync(model, feature_pipeline, df: pd.DataFrame) -> tuple[float, int]:
    """Run feature extraction + model inference synchronously.

    This function is meant to be run inside a ThreadPoolExecutor to avoid
    blocking the event loop.

    Raises:
        ModelNotLoadedException: If model or feature_pipeline is None.
    """
    if model is None:
        raise ModelNotLoadedException("Model is not loaded for inference")
    if feature_pipeline is None:
        raise ModelNotLoadedException("Feature pipeline is not loaded for inference")

    features = feature_pipeline.transform(df)
    if features.shape[0] == 0:
        return 0.01, 0

    proba = model.predict_proba(features)
    failure_probability = float(proba[0, 1])
    predicted_class = int(failure_probability >= 0.5)

    return failure_probability, predicted_class


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    body: PredictionRequest,
    model=Depends(get_model),
    feature_pipeline=Depends(get_feature_pipeline),
    quality_gate=Depends(get_quality_gate),
    thread_pool=Depends(get_thread_pool),
) -> PredictionResponse:
    """Predict failure probability for a single equipment unit.

    Applies the quality gate first. If quality checks fail, a
    DataQualityException is raised (422). The prediction itself runs
    in a ThreadPoolExecutor to keep the event loop free.

    Falls back to demo mode (heuristic-based prediction) if the model
    or feature pipeline is not loaded.
    """
    equipment_id = body.readings[0].equipment_id
    request_id = getattr(request.state, "request_id", None)

    demo_mode = request.app.state.demo_mode
    model_version = getattr(request.app.state, "model_version", None)

    df = _readings_to_dataframe(body.readings)

    gate_info, quality_warnings = _run_quality_gate(quality_gate, df)

    if not gate_info.passed and quality_gate is not None:
        raise DataQualityException(
            detail=f"Quality gate failed: {gate_info.summary}",
            violations=gate_info.total_violations,
        )

    if demo_mode:
        logger.debug(
            "Demo mode prediction for equipment=%s (model not loaded)",
            equipment_id,
        )
        failure_probability, predicted_class = await _demo_mode_predict_async(
            thread_pool, df, equipment_id, 0.2
        )
    else:
        try:
            failure_probability, predicted_class = await _run_inference_in_thread(
                thread_pool, model, feature_pipeline, df
            )
        except ModelNotLoadedException:
            logger.warning(
                "Model not loaded during prediction for equipment=%s, falling back to demo mode",
                equipment_id,
            )
            _increment_error_count(request)
            demo_mode = True
            failure_probability, predicted_class = await _demo_mode_predict_async(
                thread_pool, df, equipment_id, 0.2
            )
        except Exception:
            logger.exception(
                "Unexpected error during inference for equipment=%s",
                equipment_id,
            )
            _increment_error_count(request)
            demo_mode = True
            failure_probability, predicted_class = await _demo_mode_predict_async(
                thread_pool, df, equipment_id, 0.2
            )

    pred_count = getattr(request.app.state, "prediction_count", 0) + 1
    with suppress(Exception):
        request.app.state.prediction_count = pred_count

    return PredictionResponse(
        equipment_id=equipment_id,
        failure_probability=round(failure_probability, 6),
        predicted_class=predicted_class,
        prediction_threshold=0.5,
        model_version=model_version if not demo_mode else None,
        demo_mode=demo_mode,
        quality_gate=gate_info,
        data_quality_warnings=quality_warnings,
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
    )


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
    request: Request,
    body: BatchPredictionRequest,
    model=Depends(get_model),
    feature_pipeline=Depends(get_feature_pipeline),
    quality_gate=Depends(get_quality_gate),
    thread_pool=Depends(get_thread_pool),
) -> BatchPredictionResponse:
    """Predict failure probability for up to 50 equipment units in parallel.

    Each equipment unit's inference runs in the ThreadPoolExecutor.
    Individual equipment that fail quality gate checks are still
    predicted but receive data quality warnings.
    """
    demo_mode = request.app.state.demo_mode
    model_version = getattr(request.app.state, "model_version", None)
    request_id = getattr(request.state, "request_id", None)

    df = _readings_to_dataframe(body.readings)
    equipment_groups = df.groupby("equipment_id")

    results: list[BatchPredictionItem] = []
    failures_detected = 0

    for eq_id, eq_df in equipment_groups:
        gate_info, quality_warnings = _run_quality_gate(quality_gate, eq_df)

        if demo_mode:
            failure_prob, pred_class = await _demo_mode_predict_async(
                thread_pool, eq_df, eq_id, 0.2
            )
        else:
            try:
                failure_prob, pred_class = await _run_inference_in_thread(
                    thread_pool, model, feature_pipeline, eq_df
                )
            except ModelNotLoadedException:
                _increment_error_count(request)
                demo_mode = True
                failure_prob, pred_class = await _demo_mode_predict_async(
                    thread_pool, eq_df, eq_id, 0.2
                )
            except Exception:
                _increment_error_count(request)
                logger.exception(
                    "Unexpected error during batch inference for equipment=%s",
                    eq_id,
                )
                demo_mode = True
                failure_prob, pred_class = await _demo_mode_predict_async(
                    thread_pool, eq_df, eq_id, 0.2
                )

        if pred_class == 1:
            failures_detected += 1

        results.append(
            BatchPredictionItem(
                equipment_id=str(eq_id),
                failure_probability=round(failure_prob, 6),
                predicted_class=pred_class,
                data_quality_warnings=quality_warnings,
            )
        )

    pred_count = getattr(request.app.state, "prediction_count", 0) + 1
    with suppress(Exception):
        request.app.state.prediction_count = pred_count

    return BatchPredictionResponse(
        results=results,
        total_equipment=len(results),
        failures_detected=failures_detected,
        model_version=model_version if not demo_mode else None,
        demo_mode=demo_mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
    )


def _increment_error_count(request: Request) -> None:
    """Safely increment the error counter on the application state."""
    with suppress(Exception):
        current = getattr(request.app.state, "error_count", 0)
        request.app.state.error_count = current + 1


async def _demo_mode_predict_async(
    thread_pool: ThreadPoolExecutor,
    df: pd.DataFrame,
    equipment_id: str,
    threshold: float,
) -> tuple[float, int]:
    """Run demo mode prediction in thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_pool,
        _demo_mode_predict,
        df.copy(),
        equipment_id,
        threshold,
    )


async def _run_inference_in_thread(
    thread_pool: ThreadPoolExecutor,
    model,
    feature_pipeline,
    df: pd.DataFrame,
) -> tuple[float, int]:
    """Run feature extraction + model inference in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        thread_pool,
        _do_inference_sync,
        model,
        feature_pipeline,
        df.copy(),
    )
