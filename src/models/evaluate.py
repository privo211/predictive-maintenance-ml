"""
Rigorous evaluation harness for failure prediction models.

Provides comprehensive metrics computation, visualization, and
promotion-gate checking for industrial predictive maintenance models.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Rigorous evaluation harness for failure prediction models.

    Computes standard classification metrics, diagnostic plots, and
    promotion-gate checks against configurable thresholds. Industrial
    failure prediction is evaluated primarily on recall (must catch
    failures) and false-positive rate (must not trigger false alarms).

    Usage:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate(model, X_test, y_test, feature_names)
        >>> passed, failures = evaluator.check_promotion_gates(metrics)
        >>> report = evaluator.generate_report(metrics, y_test, y_pred, y_proba)
    """

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Compute all evaluation metrics for a trained model.

        Args:
            model: Trained classifier with ``predict`` and ``predict_proba``.
            X_test: Test feature matrix of shape (n_samples, n_features).
            y_test: True labels of shape (n_samples,).
            feature_names: Ordered list of feature names.

        Returns:
            Structured dict:

            .. code-block:: python

                {
                    "accuracy": float,
                    "recall": float,
                    "precision": float,
                    "f1": float,
                    "roc_auc": float,
                    "pr_auc": float,
                    "confusion_matrix": list[list[int]],
                    "false_positive_rate": float,
                    "false_negative_rate": float,
                    "classification_report": str,
                    "n_samples": int,
                    "n_features": int,
                }
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Healthy", "Failure"], zero_division=0,
            ),
            "n_samples": int(len(y_test)),
            "n_features": int(X_test.shape[1]),
        }

        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        else:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

        logger.info(
            "Evaluation complete — recall=%.4f, precision=%.4f, roc_auc=%.4f, fpr=%.4f",
            metrics["recall"],
            metrics["precision"],
            metrics["roc_auc"],
            metrics["false_positive_rate"],
        )

        return metrics

    def check_promotion_gates(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check if model passes promotion thresholds.

        Six gates covering recall (must catch failures), false-positive
        rate (must not trigger false alarms), and overall discrimination:

        - ``recall >= 0.90``
        - ``fpr <= 0.10``
        - ``roc_auc >= 0.92``
        - ``precision >= 0.70``
        - ``f1 >= 0.80``
        - ``pr_auc >= 0.85``

        Thresholds can be overridden via ``self._thresholds`` (set after
        construction or from a config file).

        Args:
            metrics: Metrics dict from ``evaluate()``.

        Returns:
            Tuple of (passed: bool, failed_gates: list of gate descriptions).
        """
        thresholds = getattr(self, "_thresholds", None) or {
            "recall": 0.90,
            "fpr": 0.10,
            "roc_auc": 0.92,
            "precision": 0.70,
            "f1": 0.80,
            "pr_auc": 0.85,
        }

        gates = {
            "recall": ("recall >= {:.2f}".format(thresholds["recall"]), metrics.get("recall", 0.0), thresholds["recall"], "gte"),
            "fpr": ("fpr <= {:.2f}".format(thresholds["fpr"]), metrics.get("false_positive_rate", 1.0), thresholds["fpr"], "lte"),
            "roc_auc": ("roc_auc >= {:.2f}".format(thresholds["roc_auc"]), metrics.get("roc_auc", 0.0), thresholds["roc_auc"], "gte"),
            "precision": ("precision >= {:.2f}".format(thresholds["precision"]), metrics.get("precision", 0.0), thresholds["precision"], "gte"),
            "f1": ("f1 >= {:.2f}".format(thresholds["f1"]), metrics.get("f1", 0.0), thresholds["f1"], "gte"),
            "pr_auc": ("pr_auc >= {:.2f}".format(thresholds["pr_auc"]), metrics.get("pr_auc", 0.0), thresholds["pr_auc"], "gte"),
        }

        failures: list[str] = []
        for gate_name, (description, value, threshold, operator) in gates.items():
            if operator == "gte" and value < threshold:
                failures.append(f"{description} (actual: {value:.4f})")
            elif operator == "lte" and value > threshold:
                failures.append(f"{description} (actual: {value:.4f})")

        passed = len(failures) == 0

        if passed:
            logger.info("All promotion gates PASSED.")
        else:
            logger.warning("Promotion gates FAILED: %s", failures)

        return passed, failures

    def generate_report(
        self,
        metrics: dict[str, Any],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> str:
        """Generate a human-readable evaluation report.

        Args:
            metrics: Metrics dict from ``evaluate()``.
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            Multi-line string report suitable for logging or display.
        """
        cm = metrics["confusion_matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        lines = [
            "=" * 60,
            "  MODEL EVALUATION REPORT",
            "=" * 60,
            f"  Samples:           {metrics['n_samples']}",
            f"  Features:          {metrics['n_features']}",
            f"  Positive rate:     {np.mean(y_true):.4f}",
            "",
            "  --- Classification Metrics ---",
            f"  Accuracy:          {metrics['accuracy']:.4f}",
            f"  Recall:            {metrics['recall']:.4f}",
            f"  Precision:         {metrics['precision']:.4f}",
            f"  F1 Score:          {metrics['f1']:.4f}",
            f"  ROC AUC:           {metrics['roc_auc']:.4f}",
            f"  PR AUC:            {metrics['pr_auc']:.4f}",
            "",
            "  --- Error Rates ---",
            f"  False Positive Rate: {metrics['false_positive_rate']:.4f}",
            f"  False Negative Rate: {metrics['false_negative_rate']:.4f}",
            "",
            "  --- Confusion Matrix ---",
            f"                    Predicted",
            f"                  Healthy   Failure",
            f"  Actual Healthy   {tn:6d}   {fp:6d}",
            f"  Actual Failure   {fn:6d}   {tp:6d}",
            "",
            "  --- Promotion Gates ---",
        ]

        passed, failures = self.check_promotion_gates(metrics)
        if passed:
            lines.append("  Status: PASSED")
        else:
            lines.append("  Status: FAILED")
            for failure in failures:
                lines.append(f"    - {failure}")

        lines.append("")
        lines.append("  --- Classification Report ---")
        lines.append(metrics.get("classification_report", "N/A"))

        return "\n".join(lines)


# ------------------------------------------------------------------
# Standalone plotting functions
# ------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Healthy", "Failure"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot ROC curve with AUC annotation.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot precision-recall curve with average precision annotation.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="green", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PR curve saved to %s", save_path)

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot calibration (reliability) curve.

    A well-calibrated model has predicted probabilities that match
    observed frequencies. XGBoost tends to be under-confident without
    calibration, so this plot is useful for assessing reliability.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        n_bins: Number of bins for calibration curve.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", color="steelblue", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Calibration curve saved to %s", save_path)

    return fig


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics at a custom decision threshold.

    This is useful for tuning the operating point — for predictive
    maintenance, you may want higher recall at the cost of precision
    by lowering the threshold.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold in [0.0, 1.0].

    Returns:
        Dict with ``recall``, ``precision``, ``f1``, ``accuracy``,
        ``false_positive_rate``, ``false_negative_rate``.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0.0),
        "false_negative_rate": float(fn / (fn + tp) if (fn + tp) > 0 else 0.0),
    }
