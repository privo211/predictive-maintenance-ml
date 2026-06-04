"""SHAP explanation endpoint — prediction with feature-level contributions.

POST /api/v1/explain — predict failure + return top SHAP contributors.

Requires both the model and SHAP explainer to be loaded. Falls back to
demo mode with synthetic feature contributions when models are missing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request

from api.dependencies import get_explainer, get_feature_pipeline, get_model, get_thread_pool
from api.errors import ModelNotLoadedException
from api.schemas.requests import ExplainRequest
from api.schemas.responses import ExplanationResponse, FeatureContribution

router = APIRouter(prefix="/api/v1", tags=["explainability"])
logger = logging.getLogger(__name__)


def _explain_sync(model, explainer, feature_pipeline, df: pd.DataFrame) -> dict:
    """Run feature extraction + SHAP explanation synchronously.

    This is meant to run inside a ThreadPoolExecutor.
    """
    features = feature_pipeline.transform(df)
    feature_names = feature_pipeline.get_feature_names()

    if features.shape[0] == 0:
        return {
            "failure_probability": 0.01,
            "predicted_class": 0,
            "base_value": 0.0,
            "shap_values": [],
            "top_contributors": [],
        }

    proba = model.predict_proba(features)
    failure_probability = float(proba[0, 1])
    predicted_class = int(failure_probability >= 0.5)

    explanation = explainer.explain_single(features, feature_names)

    shap_values_list = []
    try:
        shap_values_list = explainer.explain(features, feature_names)["shap_values"]
        shap_values_list = shap_values_list[0].tolist() if shap_values_list.ndim > 1 else shap_values_list.tolist()
    except Exception:
        shap_values_list = [0.0] * len(feature_names)

    top_contributors = [
        FeatureContribution(
            feature=c["feature"],
            shap_value=round(c["shap_value"], 6),
            direction=c["direction"],
            feature_value=round(c["feature_value"], 6),
        )
        for c in explanation.get("top_contributors", [])
    ]

    return {
        "failure_probability": failure_probability,
        "predicted_class": predicted_class,
        "base_value": explainer.expected_value,
        "shap_values": shap_values_list,
        "top_contributors": top_contributors,
    }


def _demo_explain_sync(df: pd.DataFrame, feature_pipeline) -> dict:
    """Demo-mode explanation using synthetic feature contributions.

    When the model/explainer is unavailable, we extract features from the
    pipeline and assign synthetic SHAP-like values based on feature
    statistics (variance, mean deviation).
    """
    try:
        features = feature_pipeline.transform(df)
        feature_names = feature_pipeline.get_feature_names()
    except Exception:
        return {
            "failure_probability": 0.05,
            "predicted_class": 0,
            "base_value": 0.0,
            "shap_values": [],
            "top_contributors": [],
        }

    if features.shape[0] == 0:
        return {
            "failure_probability": 0.05,
            "predicted_class": 0,
            "base_value": 0.0,
            "shap_values": [],
            "top_contributors": [],
        }

    feature_row = features[0]
    synthetic_shap = np.abs(feature_row - feature_row.mean()) * np.sign(
        feature_row - feature_row.mean()
    )

    synthetic_shap = np.nan_to_num(synthetic_shap, nan=0.0)

    synthetic_proba = float(1.0 / (1.0 + np.exp(-synthetic_shap.sum() * 0.01)))
    synthetic_proba = max(0.0, min(1.0, synthetic_proba))
    predicted_class = 1 if synthetic_proba >= 0.5 else 0

    indexed = sorted(
        [(abs(synthetic_shap[i]), synthetic_shap[i], feature_names[i], float(feature_row[i]))
         for i in range(len(feature_names))],
        key=lambda x: x[0],
        reverse=True,
    )

    top_contributors = [
        FeatureContribution(
            feature=feat_name,
            shap_value=round(shap_val, 6),
            direction="increases_failure_risk" if shap_val > 0 else "decreases_failure_risk",
            feature_value=round(feat_val, 6),
        )
        for _, shap_val, feat_name, feat_val in indexed[:10]
    ]

    return {
        "failure_probability": round(synthetic_proba, 6),
        "predicted_class": predicted_class,
        "base_value": 0.0,
        "shap_values": synthetic_shap.tolist(),
        "top_contributors": top_contributors,
    }


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: Request,
    body: ExplainRequest,
    model=Depends(get_model),
    explainer=Depends(get_explainer),
    feature_pipeline=Depends(get_feature_pipeline),
    thread_pool=Depends(get_thread_pool),
) -> ExplanationResponse:
    """Predict failure probability and return SHAP feature contributions.

    Requires the model and SHAP explainer to be loaded. When either is
    unavailable, falls back to demo mode with synthetic contribution
    values derived from feature statistics.
    """
    import asyncio

    equipment_id = body.readings[0].equipment_id
    request_id = getattr(request.state, "request_id", None)

    records = [
        {
            "timestamp": r.timestamp,
            "equipment_id": r.equipment_id,
            "sensor_name": r.sensor_name,
            "value": r.value,
        }
        for r in body.readings
    ]
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    demo_mode = False
    model_version = getattr(request.app.state, "model_version", None)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            thread_pool,
            _explain_sync,
            model,
            explainer,
            feature_pipeline,
            df.copy(),
        )
    except ModelNotLoadedException:
        logger.warning(
            "Model/explainer not loaded for equipment=%s, using demo explanation",
            equipment_id,
        )
        demo_mode = True
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            thread_pool,
            _demo_explain_sync,
            df.copy(),
            feature_pipeline,
        )

    pred_count = getattr(request.app.state, "prediction_count", 0) + 1
    try:
        request.app.state.prediction_count = pred_count
    except Exception:
        pass

    return ExplanationResponse(
        equipment_id=equipment_id,
        failure_probability=result["failure_probability"],
        predicted_class=result["predicted_class"],
        base_value=result["base_value"],
        shap_values=result["shap_values"],
        top_contributors=result["top_contributors"],
        model_version=model_version if not demo_mode else None,
        demo_mode=demo_mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id,
    )
