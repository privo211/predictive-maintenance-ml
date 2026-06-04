"""Drift and data-quality monitoring powered by Evidently AI.

Reports are generated on-demand (e.g. via a scheduled background task) and
compared against a reference dataset that represents the training distribution.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy Evidently imports — the package is heavy and may not be available in
# every deployment context (e.g. CI runners that skip drift tests).
# ---------------------------------------------------------------------------

_drift_preset: Any = None
_quality_preset: Any = None


def _import_evidently() -> tuple[Any, Any]:
    global _drift_preset, _quality_preset

    if _drift_preset is None:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        _drift_preset = DataDriftPreset
        _quality_preset = DataQualityPreset

    return _drift_preset, _quality_preset


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, dict[str, Any]] = {
    "dataset_drift": {"threshold": True, "severity": "HIGH"},
    "column_drift_fraction": {"threshold": 0.50, "severity": "MEDIUM"},
    "missing_rate": {"threshold": 0.05, "severity": "HIGH"},
    "duplicate_rate": {"threshold": 0.001, "severity": "LOW"},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class EvidentlyMonitor:
    """Generates data-drift, data-quality, and prediction-drift reports.

    Instantiate with optional reference data or call :meth:`set_reference`
    before generating the first drift report.

    Parameters
    ----------
    reference_data:
        Training / reference DataFrame used as the drift baseline.
    """

    def __init__(self, reference_data: pd.DataFrame | None = None) -> None:
        self.reference_data: pd.DataFrame | None = reference_data

    # ------------------------------------------------------------------
    # Reference management
    # ------------------------------------------------------------------

    def set_reference(self, df: pd.DataFrame) -> None:
        self.reference_data = df
        _logger.info("Reference data set with %d rows, %d columns", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # Drift & quality reports
    # ------------------------------------------------------------------

    def run_drift_report(self, current_data: pd.DataFrame) -> dict[str, Any]:
        if self.reference_data is None:
            raise ValueError("Reference data has not been set. Call set_reference() first.")

        DataDriftPreset, DataQualityPreset = _import_evidently()

        drift = DataDriftPreset()
        drift.execute(reference_data=self.reference_data, current_data=current_data)
        drift_dict = drift.as_dict()

        quality = DataQualityPreset()
        quality.execute(reference_data=self.reference_data, current_data=current_data)
        quality_dict = quality.as_dict()

        report: dict[str, Any] = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reference_rows": len(self.reference_data),
            "reference_columns": list(self.reference_data.columns),
            "current_rows": len(current_data),
            "current_columns": list(current_data.columns),
            "dataset_drift": drift_dict.get("metrics", [{}])[0].get("result", {}).get("dataset_drift", False),
            "drift_by_columns": self._parse_column_drift(drift_dict),
            "data_quality": self._parse_data_quality(quality_dict),
        }

        _logger.info(
            "Drift report %s: dataset_drift=%s, columns_drifted=%d",
            report["report_id"],
            report["dataset_drift"],
            len([c for c in report["drift_by_columns"].values() if c.get("drifted")]),
        )

        return report

    def run_prediction_drift(
        self,
        current_preds: np.ndarray,
        reference_preds: np.ndarray,
    ) -> dict[str, Any]:
        if len(reference_preds) == 0:
            raise ValueError("Reference predictions array is empty.")

        pred_df = pd.DataFrame({"prediction": current_preds})
        ref_df = pd.DataFrame({"prediction": reference_preds})

        DataDriftPreset, _ = _import_evidently()

        drift = DataDriftPreset()
        drift.execute(reference_data=ref_df, current_data=pred_df)
        drift_dict = drift.as_dict()

        report: dict[str, Any] = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reference_count": len(reference_preds),
            "current_count": len(current_preds),
            "drifted": drift_dict.get("metrics", [{}])[0].get("result", {}).get("dataset_drift", False),
            "drift_score": drift_dict.get("metrics", [{}])[0].get("result", {}).get("share_of_drifted_columns", 0),
            "current_mean": float(np.mean(current_preds)),
            "current_std": float(np.std(current_preds)),
            "reference_mean": float(np.mean(reference_preds)),
            "reference_std": float(np.std(reference_preds)),
        }

        _logger.info("Prediction drift report %s: drifted=%s", report["report_id"], report["drifted"])
        return report

    # ------------------------------------------------------------------
    # Threshold evaluation
    # ------------------------------------------------------------------

    def check_thresholds(
        self,
        drift_report: dict[str, Any],
        thresholds: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        t = thresholds or _DEFAULT_THRESHOLDS
        alerts: list[dict[str, Any]] = []

        dataset_drifted = bool(drift_report.get("dataset_drift", False))
        if dataset_drifted:
            alerts.append({
                "metric": "dataset_drift",
                "value": dataset_drifted,
                "threshold": t["dataset_drift"]["threshold"],
                "severity": t["dataset_drift"]["severity"],
                "message": "Overall dataset drift detected.",
            })

        drift_cols = {
            col: info
            for col, info in drift_report.get("drift_by_columns", {}).items()
            if info.get("drifted")
        }
        total_cols = len(drift_report.get("drift_by_columns", {}))
        if total_cols > 0 and len(drift_cols) / total_cols > t["column_drift_fraction"]["threshold"]:
            alerts.append({
                "metric": "column_drift_fraction",
                "value": round(len(drift_cols) / total_cols, 3),
                "threshold": t["column_drift_fraction"]["threshold"],
                "severity": t["column_drift_fraction"]["severity"],
                "message": f"{len(drift_cols)}/{total_cols} columns have drifted.",
            })

        quality = drift_report.get("data_quality", {})
        missing_rate = float(quality.get("missing_rate", 0))
        if missing_rate > t["missing_rate"]["threshold"]:
            alerts.append({
                "metric": "missing_rate",
                "value": round(missing_rate, 4),
                "threshold": t["missing_rate"]["threshold"],
                "severity": t["missing_rate"]["severity"],
                "message": f"Missing data rate is {missing_rate:.2%}.",
            })

        duplicate_rate = float(quality.get("duplicate_rate", 0))
        if duplicate_rate > t["duplicate_rate"]["threshold"]:
            alerts.append({
                "metric": "duplicate_rate",
                "value": round(duplicate_rate, 4),
                "threshold": t["duplicate_rate"]["threshold"],
                "severity": t["duplicate_rate"]["severity"],
                "message": f"Duplicate rate is {duplicate_rate:.2%}.",
            })

        _logger.debug("Threshold check: %d alerts triggered", len(alerts))
        return alerts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(self, report: dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        _logger.info("Drift report saved to %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_column_drift(drift_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
        parsed: dict[str, dict[str, Any]] = {}

        metrics = drift_dict.get("metrics", [])
        for metric in metrics:
            for col_result in metric.get("result", {}).get("drift_by_columns", {}).values():
                col_name = col_result.get("column_name", "unknown")
                parsed[col_name] = {
                    "column_type": col_result.get("column_type"),
                    "drifted": col_result.get("drift_detected", False),
                    "p_value": col_result.get("p_value"),
                    "stat_test": col_result.get("stat_test_name"),
                }

        return parsed

    @staticmethod
    def _parse_data_quality(quality_dict: dict[str, Any]) -> dict[str, Any]:
        quality: dict[str, Any] = {}
        metrics = quality_dict.get("metrics", [])

        for metric in metrics:
            result = metric.get("result", {})

            if "current" in result and "reference" in result:
                current = result["current"]
                ref = result["reference"]

                quality["missing_rate"] = current.get("share_of_missing_values", 0)
                quality["duplicate_rate"] = current.get("share_of_duplicates", 0)
                quality["reference_missing_rate"] = ref.get("share_of_missing_values", 0)
                quality["reference_duplicate_rate"] = ref.get("share_of_duplicates", 0)

        return quality
