"""
SHAP TreeExplainer wrapper for exact, fast XGBoost explainability.

The TreeExplainer provides exact (not approximate) SHAP values for
tree-based models like XGBoost, making explanations both fast and
guaranteed-correct. This is the gold-standard approach for industrial
predictive maintenance where explanations must be trustworthy.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP TreeExplainer wrapper for XGBoost models.

    Initialises a SHAP TreeExplainer which computes exact SHAP values
    for tree ensemble models. Supports both global feature importance
    and per-prediction explanations suitable for API responses.

    Usage:
        >>> explainer = SHAPExplainer(trained_xgb_model)
        >>> result = explainer.explain(X_test, feature_names)
        >>> single = explainer.explain_single(X_test[0:1], feature_names)
        >>> importances = explainer.global_importance(X_test, feature_names)
        >>> explainer.save_explainer("models/shap_explainer.joblib")
    """

    def __init__(self, model) -> None:
        """Initialise SHAP TreeExplainer for an XGBoost model.

        Args:
            model: A trained xgboost.XGBClassifier (or compatible) instance.
        """
        self._model = model
        self._explainer = shap.TreeExplainer(model)
        self._expected_value = self._explainer.expected_value

        if isinstance(self._expected_value, (np.ndarray, list)):
            # For binary classification, TreeExplainer may return a list of
            # two expected values. We want the positive-class value.
            if isinstance(self._expected_value, np.ndarray) and self._expected_value.ndim > 0:
                self._expected_value = float(self._expected_value[1])
            elif isinstance(self._expected_value, list):
                self._expected_value = float(self._expected_value[1])
            else:
                self._expected_value = float(self._expected_value)
        else:
            self._expected_value = float(self._expected_value)

        logger.info(
            "SHAPExplainer initialised — expected_value=%.4f", self._expected_value
        )

    @property
    def expected_value(self) -> float:
        """Base (expected) model output in log-odds space."""
        return self._expected_value

    def explain(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute SHAP values for a set of predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            feature_names: Ordered list of feature names. If None, uses
                generic names ``feature_0``, ``feature_1``, etc.

        Returns:
            Dict with:
            - ``shap_values``: ndarray of shape (n_samples, n_features).
            - ``base_value``: float, expected model output.
            - ``feature_importance``: dict of {name: mean_abs_shap}.
            - ``top_contributors``: list of top 5 feature names by |SHAP|.
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        shap_values = self._explainer.shap_values(X)

        # For binary classification, shap_values may come as list of two arrays.
        # We want the positive class (index 1).
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[1])

        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, mean_abs.tolist()))

        sorted_idx = np.argsort(mean_abs)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:5]]

        return {
            "shap_values": shap_values,
            "base_value": self._expected_value,
            "feature_importance": feature_importance,
            "top_contributors": top_features,
        }

    def explain_single(
        self,
        X: np.ndarray,
        feature_names: list[str],
        class_idx: int = 1,
    ) -> dict[str, Any]:
        """Explain a single prediction in an API-friendly format.

        Args:
            X: Single sample of shape (1, n_features).
            feature_names: Ordered list of feature names.
            class_idx: Class index for explanation (0 = healthy, 1 = failure).
                Defaults to 1 (failure class).

        Returns:
            Dict suitable for API response:
            - ``prediction``: int, predicted class (0 or 1).
            - ``probability``: float, predicted probability.
            - ``top_contributors``: list of dicts with keys:
              ``feature``, ``shap_value``, ``direction``,
              ``feature_value``, ``expected_range``.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        proba = self._model.predict_proba(X)[0]
        prediction = int(self._model.predict(X)[0])
        probability = float(proba[class_idx])

        shap_values = self._explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[class_idx])

        shap_row = shap_values[0]

        # Build contributor list sorted by absolute SHAP value
        indexed = [
            (
                abs(shap_row[i]),
                shap_row[i],
                feature_names[i],
                float(X[0, i]),
            )
            for i in range(len(shap_row))
        ]
        indexed.sort(key=lambda x: x[0], reverse=True)

        top_contributors: list[dict[str, Any]] = []
        for _, shap_val, feat_name, feat_value in indexed[:10]:
            top_contributors.append({
                "feature": feat_name,
                "shap_value": float(shap_val),
                "direction": "increases_failure_risk" if shap_val > 0 else "decreases_failure_risk",
                "feature_value": feat_value,
                "expected_range": f"mean ± std",
            })

        return {
            "prediction": prediction,
            "probability": probability,
            "top_contributors": top_contributors,
        }

    def global_importance(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Compute global feature importance from mean absolute SHAP values.

        Args:
            X: Representative sample of feature matrix for computing
               global SHAP values. Should be a subset (e.g., 500-1000
               samples) rather than the full dataset for performance.
            feature_names: Ordered list of feature names.

        Returns:
            Dict with:
            - ``feature_importance``: dict of {name: mean_abs_shap},
              sorted descending.
            - ``top_features``: list of top 10 feature names.
            - ``shap_values``: ndarray of shape (n_samples, n_features).
        """
        shap_values = self._explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[1])

        mean_abs = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]

        feature_importance = {
            feature_names[i]: float(mean_abs[i])
            for i in sorted_idx
        }
        top_features = [feature_names[i] for i in sorted_idx[:10]]

        logger.info(
            "Global importance computed — top feature: %s (%.4f)",
            top_features[0] if top_features else "N/A",
            mean_abs[sorted_idx[0]] if len(sorted_idx) > 0 else 0.0,
        )

        return {
            "feature_importance": feature_importance,
            "top_features": top_features,
            "shap_values": shap_values,
        }

    def save_explainer(self, path: str) -> None:
        """Persist the SHAP explainer to disk via joblib.

        Args:
            path: File path (e.g., ``"models/shap_explainer.joblib"``).
        """
        from pathlib import Path

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(file_path))
        logger.info("SHAPExplainer saved to %s", file_path)
