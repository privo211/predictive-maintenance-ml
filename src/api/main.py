"""FastAPI application factory for the Industrial Predictive Maintenance Platform.

Creates and configures a production-grade FastAPI application with:
- Async lifespan for model/quality-gate/feature-pipeline loading at startup.
- ThreadPoolExecutor for non-blocking ML inference.
- CORS middleware, request ID tracing, and structured logging.
- Graceful degradation to "demo mode" when the ML model is not available.
"""

from __future__ import annotations

import logging
import logging.config
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.errors import (
    DataQualityException,
    ModelNotLoadedException,
    data_quality_exception_handler,
    general_exception_handler,
    model_not_loaded_exception_handler,
)
from api.middleware import RequestIDMiddleware
from api.routers import explain, health, models, predict
from config.settings import settings

logger = logging.getLogger(__name__)

_MAX_WORKERS = 4
_PROJECT_ROOT = settings.project_root


def _resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return (_PROJECT_ROOT / relative_path).resolve()


def _setup_logging() -> None:
    """Load structured logging configuration from config/logging.yaml."""
    log_config_path = _PROJECT_ROOT / "config" / "logging.yaml"
    if log_config_path.exists():
        try:
            with open(log_config_path) as fh:
                log_dict = yaml.safe_load(fh)
            logging.config.dictConfig(log_dict)
        except (ValueError, ModuleNotFoundError) as exc:
            logger.warning("Failed to configure logging from %s: %s. Falling back to console logging.", log_config_path, exc)
            _fallback_logging()
    else:
        _fallback_logging()


def _fallback_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="[%(asctime)s] %(levelname)-8s %(name)-30s %(message)s",
    )


def _load_quality_gate() -> object | None:
    """Attempt to load DataQualityGate with sensor registry."""
    sensor_registry = _PROJECT_ROOT / "config" / "sensor_registry.yaml"
    if not sensor_registry.exists():
        logger.warning(
            "Sensor registry not found at %s; quality gate disabled.",
            sensor_registry,
        )
        return None

    try:
        from data.quality_gate import DataQualityGate

        gate = DataQualityGate(
            sensor_registry_path=str(sensor_registry),
            config={
                "stop_on_first_failure": False,
                "max_gap_minutes": 60.0,
            },
        )
        logger.info("DataQualityGate initialised successfully.")
        return gate
    except Exception:
        logger.exception("Failed to initialise DataQualityGate.")
        return None


def _load_feature_pipeline(model_path: Path) -> object | None:
    """Attempt to load a fitted FeaturePipeline from disk.

    Falls back to initialising a fresh (unfitted) pipeline if the saved
    pipeline is not found.
    """
    import joblib

    if model_path.exists():
        try:
            pipeline = joblib.load(str(model_path))
            logger.info("FeaturePipeline loaded from %s", model_path)
            return pipeline
        except Exception:
            logger.exception("Failed to load FeaturePipeline from %s", model_path)

    logger.warning("FeaturePipeline not found at %s; initialising fresh pipeline.", model_path)
    try:
        from features.feature_pipeline import FeaturePipeline

        return FeaturePipeline(
            window_sizes=settings.feature_rolling_window_sizes,
        )
    except Exception:
        logger.exception("Failed to create FeaturePipeline.")
        return None


def _load_model(
    model_artifact_path: Path,
    model_registry_uri: str,
    model_name: str,
) -> tuple[object | None, object | None, object | None, bool]:
    """Load the production model and SHAP explainer.

    Attempts loading in this order:
    1. MLflow model registry (if tracking URI is reachable).
    2. Local model artifact file (joblib/xgb).

    Returns:
        Tuple of (model, explainer, registry, demo_mode).
    """
    model = None
    explainer = None
    registry = None
    demo_mode = True

    if model_registry_uri:
        try:
            from models.registry import ModelRegistry

            registry = ModelRegistry(tracking_uri=model_registry_uri)
            result = registry.get_production_model(model_name)
            if result is not None:
                model, explainer = result
                if model is not None:
                    demo_mode = False
                    logger.info(
                        "Model '%s' loaded from MLflow registry (Production stage).",
                        model_name,
                    )
        except Exception:
            logger.exception(
                "Failed to load model from MLflow registry at %s.",
                model_registry_uri,
            )

    if model is None and model_artifact_path.exists():
        try:
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(str(model_artifact_path))
            demo_mode = False
            logger.info("Model loaded from local artifact: %s", model_artifact_path)

            if explainer is None:
                try:
                    from explainability.shap_explainer import SHAPExplainer

                    explainer = SHAPExplainer(model)
                    logger.info("SHAPExplainer initialised from local model.")
                except Exception:
                    logger.exception("Failed to initialise SHAPExplainer.")
        except Exception:
            logger.exception("Failed to load model from %s", model_artifact_path)

    if model is None and model_registry_uri and registry is None:
        try:
            from models.registry import ModelRegistry

            registry = ModelRegistry(tracking_uri=model_registry_uri)
        except Exception:
            logger.exception("Failed to initialise ModelRegistry.")

    if demo_mode:
        logger.warning(
            "No model loaded — operating in DEMO MODE. "
            "Predictions will use heuristics on the health column."
        )

    return model, explainer, registry, demo_mode


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load resources at startup, clean up at shutdown.

    Startup:
    - Initialise structured logging.
    - Load or create the feature pipeline.
    - Load the XGBoost model and SHAP explainer (or enter demo mode).
    - Initialise the data quality gate.
    - Create a ThreadPoolExecutor for non-blocking inference.

    Shutdown:
    - Gracefully shut down the thread pool.
    """
    _setup_logging()

    logger.info("Starting Predictive Maintenance API server...")
    app.state.start_time = time.perf_counter()
    app.state.prediction_count = 0
    app.state.error_count = 0

    model_artifact = _resolve_path(str(settings.model_artifact_path))
    feature_pipeline_path = _resolve_path(str(settings.feature_pipeline_path))
    model_name = settings.model_registry_name

    app.state.feature_pipeline = _load_feature_pipeline(feature_pipeline_path)
    app.state.model, app.state.shap_explainer, app.state.model_registry, app.state.demo_mode = (
        _load_model(
            model_artifact_path=model_artifact,
            model_registry_uri=settings.mlflow_tracking_uri,
            model_name=model_name,
        )
    )
    app.state.model_name = model_name
    app.state.model_version = None

    if app.state.model is not None:
        app.state.model_version = "Production"

    app.state.quality_gate = _load_quality_gate()
    app.state.thread_pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

    logger.info(
        "API server ready. demo_mode=%s, model=%s, feature_pipeline=%s, quality_gate=%s",
        app.state.demo_mode,
        "loaded" if app.state.model else "not loaded",
        "loaded" if app.state.feature_pipeline else "not loaded",
        "loaded" if app.state.quality_gate else "not loaded",
    )

    yield

    logger.info("Shutting down API server...")
    app.state.thread_pool.shutdown(wait=True)
    logger.info("Thread pool shut down. Goodbye.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        A fully configured FastAPI application ready for uvicorn.
    """
    app = FastAPI(
        title="Predictive Maintenance Platform API",
        description=(
            "Industrial predictive maintenance API for real-time failure "
            "probability inference, SHAP-based explainability, and model "
            "lifecycle management."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestIDMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(DataQualityException, data_quality_exception_handler)
    app.add_exception_handler(ModelNotLoadedException, model_not_loaded_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(explain.router)
    app.include_router(models.router)

    return app


app = create_app()
