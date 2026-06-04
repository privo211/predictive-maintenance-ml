"""
Model training orchestration with MLflow tracking and SHAP explainability.

This module provides the ModelTrainer class that handles end-to-end
XGBoost classifier training with temporal data splits, class imbalance
handling via scale_pos_weight, and full MLflow experiment tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import shap
import xgboost as xgb
import yaml

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates end-to-end model training with MLflow tracking.

    Loads hyperparameters from a YAML configuration file, trains an
    XGBoost classifier with temporal train/validation/test splits,
    and logs all artifacts to MLflow.

    The temporal split is **critical** for industrial ML: it ensures
    that no future data leaks into training, simulating real-world
    deployment where models predict unseen future equipment states.

    Class imbalance is handled via ``scale_pos_weight`` computed
    dynamically from the training labels.

    Usage:
        >>> trainer = ModelTrainer(config_path="config/model_config.yaml")
        >>> model, explainer, metrics = trainer.train(
        ...     X, y, feature_names, experiment_name="predictive_maintenance"
        ... )
    """

    def __init__(self, config_path: str) -> None:
        """Load model configuration from a YAML file.

        Args:
            config_path: Path to model_config.yaml.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file is malformed.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            msg = f"Model config not found: {config_path}"
            raise FileNotFoundError(msg)

        with config_file.open("r") as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

        self._hp = self._config.get("hyperparameters", {})
        self._eval_thresholds = self._config.get("evaluation", {})
        self._splits = self._config.get("data_splits", {})
        self._model_name = self._config.get("model", {}).get("name", "failure_predictor")

        logger.info(
            "ModelTrainer initialised for '%s' from %s",
            self._model_name,
            config_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        experiment_name: str = "predictive_maintenance",
    ) -> tuple[xgb.XGBClassifier, shap.TreeExplainer, dict[str, float]]:
        """Train XGBoost classifier with full MLflow tracking.

        Pipeline:
        1. Temporal train/validation/test split (70/15/15).
        2. Compute ``scale_pos_weight`` from class imbalance.
        3. Train with early stopping on validation set.
        4. Log params, metrics, feature importance, confusion matrix,
           and SHAP summary plot to MLflow.
        5. Register model in MLflow registry at "Staging".
        6. Create SHAP TreeExplainer for post-hoc explainability.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,), 0 = healthy, 1 = failure.
            feature_names: Ordered list of feature names matching X columns.
            experiment_name: MLflow experiment name.

        Returns:
            Tuple of (trained_model, shap_explainer, metrics_dict).

            - **trained_model**: Fitted XGBoost classifier.
            - **shap_explainer**: SHAP TreeExplainer (exact for tree models).
            - **metrics_dict**: Validation metrics dict with keys like
              ``recall``, ``precision``, ``f1``, ``roc_auc``, ``pr_auc``.
        """
        import mlflow
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            accuracy_score,
            ConfusionMatrixDisplay,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # -- Temporal split --------------------------------------------------
        X_train, y_train, X_val, y_val, X_test, y_test = self._temporal_split(X, y)

        logger.info(
            "Temporal split: train=%d, val=%d, test=%d samples",
            len(y_train),
            len(y_val),
            len(y_test),
        )

        # -- Compute class weight --------------------------------------------
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        logger.info("scale_pos_weight = %.3f", scale_pos_weight)

        # -- Build hyperparameters -------------------------------------------
        params: dict[str, Any] = {
            "objective": self._config.get("model", {}).get("objective", "binary:logistic"),
            "eval_metric": self._config.get("model", {}).get("eval_metric", ["logloss", "aucpr"]),
            "max_depth": int(self._hp.get("max_depth", 6)),
            "learning_rate": float(self._hp.get("learning_rate", 0.05)),
            "n_estimators": int(self._hp.get("n_estimators", 300)),
            "subsample": float(self._hp.get("subsample", 0.8)),
            "colsample_bytree": float(self._hp.get("colsample_bytree", 0.8)),
            "scale_pos_weight": scale_pos_weight,
            "min_child_weight": int(self._hp.get("min_child_weight", 5)),
            "gamma": float(self._hp.get("gamma", 0.1)),
            "reg_alpha": float(self._hp.get("reg_alpha", 0.01)),
            "reg_lambda": float(self._hp.get("reg_lambda", 1.0)),
            "tree_method": str(self._hp.get("tree_method", "hist")),
            "random_state": int(self._hp.get("random_state", 42)),
            "early_stopping_rounds": int(self._hp.get("early_stopping_rounds", 30)),
        }

        early_stopping = params.pop("early_stopping_rounds")

        # -- MLflow tracking -------------------------------------------------
        mlflow.set_experiment(experiment_name)
        mlflow.xgboost.autolog()

        with mlflow.start_run(run_name=f"{self._model_name}_train") as run:
            # Log all hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("early_stopping_rounds", early_stopping)
            mlflow.log_param("train_samples", len(y_train))
            mlflow.log_param("val_samples", len(y_val))
            mlflow.log_param("test_samples", len(y_test))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("train_positive_rate", float(np.mean(y_train)))
            mlflow.log_param("scale_pos_weight", scale_pos_weight)

            # Build and train model
            model = xgb.XGBClassifier(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=early_stopping,
            )

            # -- Validation metrics ------------------------------------------
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]

            metrics: dict[str, float] = {
                "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_roc_auc": roc_auc_score(y_val, y_val_proba),
                # PR-AUC via average_precision_score
                "__val_pr_auc": _safe_pr_auc(y_val, y_val_proba),
            }
            # Rename for clean MLflow logging
            metrics["val_pr_auc"] = metrics.pop("__val_pr_auc")

            mlflow.log_metrics(metrics)

            # -- Feature importance plot -------------------------------------
            _log_feature_importance_plot(model, feature_names)

            # -- Confusion matrix --------------------------------------------
            _log_confusion_matrix_plot(y_val, y_val_pred)

            # -- SHAP summary plot -------------------------------------------
            shap_explainer = _log_shap_summary(model, X_val, feature_names)

            # -- Register model ----------------------------------------------
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=self._model_name,
            )

            # Transition to Staging
            try:
                client = mlflow.tracking.MlflowClient()
                model_versions = client.search_model_versions(
                    f"name='{self._model_name}'"
                )
                if model_versions:
                    latest = max(model_versions, key=lambda v: int(v.version))
                    client.transition_model_version_stage(
                        name=self._model_name,
                        version=latest.version,
                        stage="Staging",
                    )
            except Exception:
                logger.warning(
                    "Could not transition model to Staging stage "
                    "(MLflow tracking server may not be running)."
                )

            run_id = run.info.run_id
            logger.info("MLflow run complete: %s", run_id)

        return model, shap_explainer, metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _temporal_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data respecting time ordering.

        **Precondition**: ``X`` and ``y`` MUST already be sorted by
        timestamp (oldest first) before calling this method. The split
        uses sequential indices — first N% for training, next M% for
        validation, last L% for testing — so shuffle or random ordering
        will silently produce a random split, not a temporal one.

        A heuristic check warns if the positive rate differs
        dramatically between train and validation sets, which may
        indicate unsorted data.

        Args:
            X: Feature matrix, ordered by time (oldest first).
            y: Labels, ordered by time (oldest first).

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        train_frac = float(self._splits.get("train_fraction", 0.70))
        val_frac = float(self._splits.get("validation_fraction", 0.15))

        n = len(y)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # Heuristic: warn if positive rates differ dramatically
        # across adjacent splits (can indicate unsorted data).
        train_pos_rate = float(np.mean(y_train))
        val_pos_rate = float(np.mean(y_val))
        if val_pos_rate > 0 and train_pos_rate > 0:
            ratio = val_pos_rate / train_pos_rate
            if ratio > 3.0 or ratio < 0.33:
                logger.warning(
                    "Positive rate differs substantially between train "
                    "(%.3f) and validation (%.3f) sets. Data may not be "
                    "time-ordered. A temporal split requires pre-sorted data.",
                    train_pos_rate,
                    val_pos_rate,
                )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """Compute class weight from the ratio of negatives to positives.

        ``scale_pos_weight = count(y == 0) / count(y == 1)``

        This is the primary mitigation for imbalanced industrial data
        where failures typically represent 5-15% of samples.

        Args:
            y: Binary labels array.

        Returns:
            Scale factor. Returns 1.0 if either class has zero samples.
        """
        n_neg = int(np.sum(y == 0))
        n_pos = int(np.sum(y == 1))

        if n_pos == 0:
            logger.warning("No positive samples in training data; scale_pos_weight set to 1.0")
            return 1.0
        if n_neg == 0:
            logger.warning("No negative samples in training data; scale_pos_weight set to 1.0")
            return 1.0

        return n_neg / n_pos


# ------------------------------------------------------------------
# Plotting helpers (internal)
# ------------------------------------------------------------------


def _log_feature_importance_plot(
    model: xgb.XGBClassifier,
    feature_names: list[str],
) -> None:
    """Log XGBoost feature importance bar chart to MLflow."""
    import matplotlib.pyplot as plt
    import mlflow

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[-20:]  # top 20

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()

    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)


def _log_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Log confusion matrix plot to MLflow."""
    import matplotlib.pyplot as plt
    import mlflow
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Healthy", "Failure"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()

    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)


def _log_shap_summary(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
) -> shap.TreeExplainer:
    """Create SHAP explainer and log summary plot to MLflow."""
    import matplotlib.pyplot as plt
    import mlflow

    explainer = shap.TreeExplainer(model)

    # Use a sample of up to 500 points for the summary plot
    sample_size = min(500, len(X))
    sample_idx = np.random.default_rng(42).choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
    )
    fig.tight_layout()

    mlflow.log_figure(fig, "plots/shap_summary.png")
    plt.close(fig)

    return explainer


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC safely, returning 0.0 if degenerate."""
    from sklearn.metrics import average_precision_score

    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return 0.0
