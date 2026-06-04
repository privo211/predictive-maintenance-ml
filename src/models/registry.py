"""
MLflow model registry wrapper for production model management.

Handles loading, promoting, and querying registered models in the
MLflow model registry, including SHAP explainer artifacts.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow model registry wrapper for production model management.

    Provides a clean interface over the MLflow tracking client for:
    - Retrieving the current production model and its SHAP explainer.
    - Promoting model versions from Staging to Production.
    - Listing all versions with their stage and metric information.

    Usage:
        >>> registry = ModelRegistry(tracking_uri="http://localhost:5000")
        >>> model, explainer = registry.get_production_model("failure_predictor")
        >>> registry.promote_to_production("failure_predictor", "3")
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        """Initialise MLflow client with optional tracking URI.

        Args:
            tracking_uri: MLflow tracking server URI. If None, uses
                the default (local file store or MLFLOW_TRACKING_URI env var).
        """
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._client = mlflow.tracking.MlflowClient()
        self._tracking_uri = tracking_uri or mlflow.get_tracking_uri()

        logger.info("ModelRegistry initialised — tracking URI: %s", self._tracking_uri)

    def get_production_model(self, model_name: str) -> tuple[Any, Any] | None:
        """Load the Production-stage model and its SHAP explainer.

        Args:
            model_name: Registered model name in MLflow.

        Returns:
            Tuple of (model, shap_explainer) if a Production version
            exists, or None if no model is found in Production.
        """
        import mlflow

        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.xgboost.load_model(model_uri)
            logger.info("Loaded production model: %s", model_uri)
        except Exception:
            logger.warning("No Production model found for '%s'.", model_name)
            return None

        # Attempt to load SHAP explainer from MLflow artifacts
        explainer = None
        try:
            versions = self._client.search_model_versions(
                f"name='{model_name}'", max_results=1,
                order_by=["version_number DESC"],
            )
            for v in versions:
                if v.current_stage == "Production":
                    run = self._client.get_run(v.run_id)
                    artifact_uri = run.info.artifact_uri
                    import joblib
                    import os

                    explainer_path = os.path.join(
                        artifact_uri.replace("file://", ""), "shap_explainer.joblib"
                    )
                    if os.path.exists(explainer_path):
                        explainer = joblib.load(explainer_path)
                        logger.info("Loaded SHAP explainer from %s", explainer_path)
                    break
        except Exception:
            logger.debug("Could not load SHAP explainer (optional).")

        return model, explainer

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Archive current Production model and promote specified version.

        Args:
            model_name: Registered model name.
            version: Version number (as string) to promote.

        Raises:
            ValueError: If the specified version does not exist.
        """
        # Archive current Production
        try:
            prod_versions = self._client.search_model_versions(
                f"name='{model_name}'"
            )
            for v in prod_versions:
                if v.current_stage == "Production":
                    self._client.transition_model_version_stage(
                        name=model_name,
                        version=v.version,
                        stage="Archived",
                    )
                    logger.info(
                        "Archived previous Production version %s of '%s'.",
                        v.version,
                        model_name,
                    )
        except Exception:
            logger.debug("No existing Production version to archive.")

        # Promote target version
        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        logger.info(
            "Promoted '%s' version %s to Production.", model_name, version
        )

    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        """List all versions with stage and run info.

        Args:
            model_name: Registered model name.

        Returns:
            List of dicts with keys: ``version``, ``stage``, ``run_id``,
            ``status``, ``creation_timestamp``.
        """
        versions = self._client.search_model_versions(f"name='{model_name}'")
        result: list[dict[str, Any]] = []
        for v in sorted(versions, key=lambda x: int(x.version)):
            result.append({
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
                "creation_timestamp": v.creation_timestamp,
            })
        return result

    def get_model_metrics(
        self, model_name: str, version: str
    ) -> dict[str, float]:
        """Fetch logged metrics for a specific model version.

        Args:
            model_name: Registered model name.
            version: Version number to query.

        Returns:
            Dict of metric_name -> value from the model's MLflow run.
            Empty dict if the run or metrics cannot be retrieved.
        """
        versions = self._client.search_model_versions(
            f"name='{model_name}'"
        )
        for v in versions:
            if v.version == version:
                try:
                    run = self._client.get_run(v.run_id)
                    return {k: float(v) for k, v in run.data.metrics.items()}
                except Exception:
                    logger.warning(
                        "Could not fetch metrics for '%s' version %s.",
                        model_name,
                        version,
                    )
                    return {}
        logger.warning(
            "Version %s not found for model '%s'.", version, model_name
        )
        return {}
