"""FastAPI dependency injection for shared application state.

Provides callable dependency functions that yield application-level
singletons (model, explainer, feature pipeline, quality gate) stored
in the app.state namespace.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request

from api.errors import ModelNotLoadedException

logger = logging.getLogger(__name__)


def get_model(request: Request) -> Any:
    """Retrieve the loaded XGBoost model from application state.

    If the model is not loaded, raises ModelNotLoadedException. Callers
    in the predict router should catch this to fall back to demo mode
    rather than propagating the error to the client.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The XGBoost classifier or None if not loaded.

    Raises:
        ModelNotLoadedException: If the model is not available.
    """
    model = request.app.state.model
    if model is None:
        logger.warning("Model requested but not loaded in application state.")
        raise ModelNotLoadedException()
    return model


def get_explainer(request: Request) -> Any:
    """Retrieve the SHAP explainer from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The SHAPExplainer instance or None if not loaded.

    Raises:
        ModelNotLoadedException: If the explainer is not available.
    """
    explainer = request.app.state.shap_explainer
    if explainer is None:
        logger.warning("SHAP explainer requested but not loaded.")
        raise ModelNotLoadedException("SHAP explainer is not loaded")
    return explainer


def get_feature_pipeline(request: Request) -> Any:
    """Retrieve the fitted FeaturePipeline from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The fitted FeaturePipeline instance.

    Raises:
        ModelNotLoadedException: If the feature pipeline is not fitted.
    """
    pipeline = request.app.state.feature_pipeline
    if pipeline is None:
        logger.warning("Feature pipeline requested but not loaded.")
        raise ModelNotLoadedException("Feature pipeline is not loaded")
    return pipeline


def get_quality_gate(request: Request) -> Any:
    """Retrieve the DataQualityGate from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The DataQualityGate instance or None if not configured.
    """
    return request.app.state.quality_gate


def get_thread_pool(request: Request) -> Any:
    """Retrieve the ThreadPoolExecutor for non-blocking ML inference.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The ThreadPoolExecutor instance.
    """
    return request.app.state.thread_pool
