"""Model management endpoints — list, inspect, and promote model versions.

GET  /api/v1/models                 — list all model versions
POST /api/v1/models/{version}/promote — promote a version to Production
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.errors import ModelNotLoadedException
from api.schemas.responses import (
    ModelInfo,
    ModelListResponse,
    PromoteResponse,
)

router = APIRouter(prefix="/api/v1", tags=["models"])
logger = logging.getLogger(__name__)


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """List all registered model versions with their stages and metrics."""
    registry = request.app.state.model_registry

    if registry is None:
        raise ModelNotLoadedException("Model registry is not configured")

    try:
        model_name = getattr(request.app.state, "model_name", "failure_predictor")
        versions = registry.list_versions(model_name)

        current_production = None
        model_infos: list[ModelInfo] = []

        for v in versions:
            if v.get("stage") == "Production":
                current_production = v.get("version")

            metrics = None
            try:
                metrics = registry.get_model_metrics(model_name, v["version"])
            except Exception:
                metrics = None

            model_infos.append(
                ModelInfo(
                    version=v["version"],
                    stage=v.get("stage", "Unknown"),
                    run_id=v.get("run_id"),
                    status=v.get("status"),
                    creation_timestamp=v.get("creation_timestamp"),
                    metrics=metrics,
                )
            )

        return ModelListResponse(
            model_name=model_name,
            versions=model_infos,
            current_production=current_production,
        )

    except Exception as exc:
        logger.exception("Failed to list models")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model list: {str(exc)}",
        )


@router.post("/models/{version}/promote", response_model=PromoteResponse)
async def promote_model(
    version: str,
    request: Request,
) -> PromoteResponse:
    """Promote a model version to Production stage.

    Archives the currently active Production version (if any) and
    promotes the specified version.
    """
    registry = request.app.state.model_registry

    if registry is None:
        raise ModelNotLoadedException("Model registry is not configured")

    model_name = getattr(request.app.state, "model_name", "failure_predictor")
    target_version = version

    try:
        previous_stage = "Unknown"
        versions = registry.list_versions(model_name)
        for v in versions:
            if v.get("version") == target_version:
                previous_stage = v.get("stage", "Unknown")
                break

        registry.promote_to_production(model_name, target_version)

        logger.info(
            "Promoted model '%s' version %s from %s to Production",
            model_name,
            target_version,
            previous_stage,
        )

        return PromoteResponse(
            model_name=model_name,
            version=target_version,
            previous_stage=previous_stage,
            new_stage="Production",
            message=f"Model '{model_name}' version {target_version} promoted to Production",
        )

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to promote model version %s", target_version)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to promote model: {str(exc)}",
        )
