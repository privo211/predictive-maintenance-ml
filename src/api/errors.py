"""Custom exception classes and FastAPI exception handlers.

Defines domain-specific exceptions for data quality failures and model
availability issues, plus their corresponding FastAPI exception handlers.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class PredictiveMaintenanceError(Exception):
    """Base exception for all predictive maintenance API errors."""

    def __init__(self, detail: str, status_code: int = 500) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class DataQualityException(PredictiveMaintenanceError):
    """Raised when sensor data fails quality gate validation.

    The quality gate report is attached so callers can inspect per-stage
    results and decide whether to reject or accept the batch.

    Attributes:
        quality_report: The full QualityReport from the data quality gate,
            or None if the report is unavailable.
    """

    def __init__(
        self,
        detail: str,
        quality_report: object | None = None,
        violations: int = 0,
    ) -> None:
        super().__init__(detail, status_code=422)
        self.quality_report = quality_report
        self.violations = violations


class ModelNotLoadedException(PredictiveMaintenanceError):
    """Raised when a required model (or feature pipeline) is not loaded.

    The API will still serve predictions in demo mode when this exception
    is caught by the router-level handlers, but this exception marks the
    explicit case where model-dependent operations (like SHAP explanations)
    cannot proceed.
    """

    def __init__(self, detail: str = "ML model is not loaded") -> None:
        super().__init__(detail, status_code=503)


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------


async def data_quality_exception_handler(
    request: Request,
    exc: DataQualityException,
) -> JSONResponse:
    """Convert DataQualityException into a structured 422 response.

    The response body includes the quality report stage summaries and
    total violation counts so downstream clients can make informed
    decisions about retries or alerting.
    """
    logger.warning(
        "Data quality failure for %s %s: %s (violations=%d)",
        request.method,
        request.url.path,
        exc.detail,
        exc.violations,
    )

    stage_results = None
    if exc.quality_report is not None:
        try:
            stage_results = {
                name: {
                    "passed": result.passed,
                    "violations": result.violations,
                    "warnings": result.warnings,
                }
                for name, result in exc.quality_report.stage_results.items()
            }
        except Exception:
            stage_results = None

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "data_quality_failure",
            "detail": exc.detail,
            "violations": exc.violations,
            "stage_results": stage_results,
        },
    )


async def model_not_loaded_exception_handler(
    request: Request,
    exc: ModelNotLoadedException,
) -> JSONResponse:
    """Convert ModelNotLoadedException into a structured 503 response."""
    logger.error(
        "Model not loaded for %s %s: %s",
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "model_not_loaded",
            "detail": exc.detail,
        },
    )


async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Never leaks internal stack traces to the client. Logs the full
    traceback server-side for post-mortem analysis.
    """
    logger.exception(
        "Unhandled exception for %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. The incident has been logged.",
        },
    )
