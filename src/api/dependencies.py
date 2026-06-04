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

    Returns None if the model is not loaded, letting endpoint handlers
    decide whether to fall back to demo mode or raise an error.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The XGBoost classifier or None if not loaded.
    """
    return request.app.state.model


def get_explainer(request: Request) -> Any:
    """Retrieve the SHAP explainer from application state.

    Returns None if the explainer is not available, letting endpoint
    handlers decide whether to fall back to demo mode.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The SHAPExplainer instance or None if not loaded.
    """
    explainer = request.app.state.shap_explainer
    if explainer is None:
        logger.warning("SHAP explainer requested but not loaded.")
    return explainer


def get_feature_pipeline(request: Request) -> Any:
    """Retrieve the fitted FeaturePipeline from application state.

    Returns None if the pipeline is not available, letting endpoint
    handlers decide whether to fall back to demo mode.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The fitted FeaturePipeline instance or None.
    """
    return request.app.state.feature_pipeline


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
