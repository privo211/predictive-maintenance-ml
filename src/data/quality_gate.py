"""5-stage data quality firewall for sensor data validation.

The DataQualityGate validates long-form sensor DataFrames before they
reach ML models, running five sequential validation stages:

    Stage 1 - SchemaGate:     DataFrame structure & column validation.
    Stage 2 - RangeGate:      Per-sensor value range checks (from registry).
    Stage 3 - TimestampGate:  Temporal integrity (monotonicity, gaps, future).
    Stage 4 - CrossChannelGate: Redundant sensor agreement verification.
    Stage 5 - DriftGate:      CUSUM-based gradual calibration drift detection.

All sensor specifications and thresholds are sourced from
config/sensor_registry.yaml. No sensor names or thresholds are hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.drift_detector import CUSUMDetector, compute_reference_statistics
from data.quality_checks import (
    check_missing_values,
    check_timestamp_monotonicity,
    check_value_ranges,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result of a single validation gate stage.

    Attributes:
        stage_name: Human-readable stage identifier (e.g. "SchemaGate").
        passed: Whether this stage passed all checks.
        violations: Total number of violations found.
        details: Stage-specific diagnostic information.
        warnings: Human-readable warning messages.
    """

    stage_name: str
    passed: bool
    violations: int
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Structured report from a full DataQualityGate validation run.

    Attributes:
        passed: Overall pass/fail (True only if ALL stages passed).
        stage_results: Per-stage GateResult keyed by stage name.
        total_violations: Sum of violations across all stages.
        rejected_batch: Whether the batch should be rejected (any stage fails).
        summary: Human-readable one-line summary.
        timestamp: When the validation was performed.
    """

    passed: bool
    stage_results: dict[str, GateResult]
    total_violations: int
    rejected_batch: bool
    summary: str
    timestamp: pd.Timestamp


# ---------------------------------------------------------------------------
# DataQualityGate
# ---------------------------------------------------------------------------


class DataQualityGate:
    """5-stage data quality firewall for industrial sensor data.

    Validates long-form DataFrames (timestamp, equipment_id, sensor_name,
    value, health, fault_label) against sensor specifications defined in
    config/sensor_registry.yaml.

    Stages run sequentially by default. If ``stop_on_first_failure`` is set,
    validation stops at the first failing stage.

    Args:
        sensor_registry_path: Path to the sensor_registry.yaml file.
        config: Optional overrides for gate behaviour. Supported keys:
            - stop_on_first_failure (bool): Stop at first failing stage.
            - max_gap_minutes (float): Maximum allowed timestamp gap.
            - cross_channel_threshold (float): Max diff for sensor pairs
              expressed as multiple of expected noise std.
            - cusum_h (float): CUSUM decision interval.
            - cusum_k (float): CUSUM reference value (allowable deviation).
    """

    REQUIRED_COLUMNS = ["timestamp", "equipment_id", "sensor_name", "value"]
    OPTIONAL_COLUMNS = ["health", "fault_label", "equipment_type"]
    NON_BLOCKING_STAGES = {"DriftGate"}  # stages that don't affect overall pass/fail
    EXPECTED_DTYPES = {
        "timestamp": "datetime64[ns]",
        "equipment_id": "object",
        "sensor_name": "object",
        "value": "float64",
    }

    def __init__(
        self,
        sensor_registry_path: str,
        config: dict | None = None,
    ) -> None:
        self._config = config or {}
        self._stop_on_first = self._config.get("stop_on_first_failure", False)
        self._max_gap_minutes = self._config.get("max_gap_minutes", 10.0)
        self._cross_channel_factor = self._config.get(
            "cross_channel_threshold", 3.0
        )
        self._cusum_h = self._config.get("cusum_h", 50.0)
        self._cusum_k = self._config.get("cusum_k", 5.0)

        # Load sensor registry
        self._registry = self._load_registry(sensor_registry_path)

        # Build lookup tables from the registry
        self._sensor_range_map, self._redundant_pairs = self._build_sensor_maps()

        # Per-sensor, per-equipment CUSUM detectors (lazy-initialized)
        self._cusum_detectors: dict[tuple[str, str], CUSUMDetector] = {}
        # Map sensor_name -> reference mean (for initializing new detectors)
        self._sensor_ref_means: dict[str, float] = {}

        # Pre-compute reference means from registry normal ranges
        for sensor_name, (lo, hi) in self._sensor_range_map.items():
            self._sensor_ref_means[sensor_name] = (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def _load_registry(self, path: str) -> dict:
        registry_path = Path(path)
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Sensor registry not found at {registry_path.resolve()}"
            )
        with open(registry_path, "r") as fh:
            return yaml.safe_load(fh)

    def _build_sensor_maps(self) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]]]:
        sensor_range_map: dict[str, tuple[float, float]] = {}
        pair_registry: set[tuple[str, str]] = set()

        equipment_types = self._registry.get("equipment_types", {})
        for _eq_type, eq_def in equipment_types.items():
            for sensor in eq_def.get("sensors", []):
                name = sensor["name"]
                normal_min = float(sensor.get("normal_min", -np.inf))
                normal_max = float(sensor.get("normal_max", np.inf))
                sensor_range_map[name] = (normal_min, normal_max)

                redundant = sensor.get("redundant_pair")
                if redundant and redundant is not None:
                    pair = tuple(sorted([name, redundant]))
                    pair_registry.add(pair)

        redundant_pairs = sorted(pair_registry)
        logger.info(
            "Loaded %d sensors and %d redundant pairs from registry.",
            len(sensor_range_map),
            len(redundant_pairs),
        )
        return sensor_range_map, redundant_pairs

    # ------------------------------------------------------------------
    # Main validation entry point
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> QualityReport:
        """Run all 5 validation stages and return a structured report.

        Stages that would depend on data from a failed prior stage
        (e.g., RangeGate after SchemaGate failure) are skipped when
        the prior stage fails.

        Args:
            df: Long-form sensor DataFrame.

        Returns:
            QualityReport with per-stage results and overall pass/fail.
        """
        overall_passed = True
        total_violations = 0
        stage_results: dict[str, GateResult] = {}

        stage_labels = [
            "SchemaGate",
            "RangeGate",
            "TimestampGate",
            "CrossChannelGate",
            "DriftGate",
        ]
        stages = [
            self._run_schema_gate,
            self._run_range_gate,
            self._run_timestamp_gate,
            self._run_cross_channel_gate,
            self._run_drift_gate,
        ]

        for label, stage_fn in zip(stage_labels, stages):
            logger.info("Running %s ...", label)
            try:
                result = stage_fn(df)
            except Exception as exc:
                logger.exception("%s raised an unexpected exception.", label)
                result = GateResult(
                    stage_name=label,
                    passed=False,
                    violations=1,
                    warnings=[f"Exception during {label}: {exc}"],
                )

            stage_results[label] = result
            total_violations += result.violations

            if not result.passed and label not in self.NON_BLOCKING_STAGES:
                overall_passed = False
                if self._stop_on_first:
                    break

        rejected = not overall_passed
        summary = (
            f"PASSED: 0 violations across {len(stage_results)} stages."
            if overall_passed
            else f"FAILED: {total_violations} total violations. "
            f"Failing stages: {[k for k, v in stage_results.items() if not v.passed]}"
        )

        return QualityReport(
            passed=overall_passed,
            stage_results=stage_results,
            total_violations=total_violations,
            rejected_batch=rejected,
            summary=summary,
            timestamp=pd.Timestamp.now(),
        )

    # ------------------------------------------------------------------
    # Stage 1 — SchemaGate
    # ------------------------------------------------------------------

    def _run_schema_gate(self, df: pd.DataFrame) -> GateResult:
        violations = 0
        details: dict[str, Any] = {}
        warnings: list[str] = []

        # Required columns presence
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            violations += len(missing_cols)
            warnings.append(f"Missing required columns: {missing_cols}")
            details["missing_required"] = missing_cols

        # Extra columns beyond expected
        all_expected = set(self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS)
        extra_cols = [c for c in df.columns if c not in all_expected]
        if extra_cols:
            violations += len(extra_cols)
            warnings.append(f"Unexpected extra columns: {extra_cols}")
            details["extra_columns"] = extra_cols

        # Dtype validation (only for columns present)
        dtype_mismatches = {}
        STRING_DTYPES = {"object", "string", "str"}
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if expected_dtype == "object" and actual in STRING_DTYPES:
                continue
            if actual != expected_dtype:
                # datetime64 can have timezone variants
                if expected_dtype == "datetime64[ns]" and actual.startswith(
                    "datetime64"
                ):
                    continue
                dtype_mismatches[col] = {"expected": expected_dtype, "actual": actual}

        if dtype_mismatches:
            violations += len(dtype_mismatches)
            warnings.append(f"Column dtype mismatches: {dtype_mismatches}")
            details["dtype_mismatches"] = dtype_mismatches

        # Row count
        details["row_count"] = len(df)
        details["column_count"] = len(df.columns)

        passed = violations == 0
        return GateResult(
            stage_name="SchemaGate",
            passed=passed,
            violations=violations,
            details=details,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Stage 2 — RangeGate
    # ------------------------------------------------------------------

    def _run_range_gate(self, df: pd.DataFrame) -> GateResult:
        if "value" not in df.columns or "sensor_name" not in df.columns:
            return GateResult(
                stage_name="RangeGate",
                passed=True,
                violations=0,
                details={"skipped": "Required columns (value, sensor_name) missing."},
            )

        pivot_ranges: dict[str, tuple[float | None, float | None]] = {}
        sensor_range_violations: dict[str, dict[str, Any]] = {}
        total_violations = 0

        sensors_in_df = df["sensor_name"].unique().tolist()
        for sn in sensors_in_df:
            if sn in self._sensor_range_map:
                lo, hi = self._sensor_range_map[sn]
                pivot_ranges[sn] = (lo, hi if hi != np.inf else None)

        # Pivot to wide form for vectorized range checks
        pivot = df.pivot_table(
            index=["equipment_id", "timestamp"],
            columns="sensor_name",
            values="value",
            aggfunc="first",
        ).reset_index()

        range_result = check_value_ranges(pivot, pivot_ranges)
        if not range_result["passed"]:
            for col, vinfo in range_result["violations"].items():
                sensor_range_violations[col] = {
                    "violation_count": vinfo["violation_count"],
                    "violation_fraction": round(vinfo["violation_fraction"], 4),
                    "observed_min": vinfo["observed_min"],
                    "observed_max": vinfo["observed_max"],
                    "expected_range": vinfo["expected_range"],
                }
                total_violations += vinfo["violation_count"]

        warnings: list[str] = []
        if sensor_range_violations:
            worst = max(
                sensor_range_violations.items(),
                key=lambda kv: kv[1]["violation_count"],
            )
            warnings.append(
                f"Range violation: {worst[0]} had {worst[1]['violation_count']} "
                f"out-of-range readings."
            )

        return GateResult(
            stage_name="RangeGate",
            passed=len(sensor_range_violations) == 0,
            violations=total_violations,
            details={
                "sensors_checked": len(sensors_in_df),
                "sensor_range_violations": sensor_range_violations,
            },
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Stage 3 — TimestampGate
    # ------------------------------------------------------------------

    def _run_timestamp_gate(self, df: pd.DataFrame) -> GateResult:
        violations = 0
        details: dict[str, Any] = {}
        warnings: list[str] = []

        # Monotonicity per equipment_id group
        mono_result = check_timestamp_monotonicity(df, group_col="equipment_id")
        if not mono_result["passed"]:
            violations += len(mono_result["non_monotonic_groups"])
            details["non_monotonic_groups"] = mono_result["non_monotonic_groups"]
            warnings.append(
                f"Non-monotonic timestamps in {len(mono_result['non_monotonic_groups'])} groups."
            )

        # Duplicate timestamps within equipment groups
        dup_result = self._check_duplicate_timestamps(df)
        if dup_result["duplicate_count"] > 0:
            violations += dup_result["duplicate_count"]
            details["duplicate_timestamps"] = dup_result
            warnings.append(
                f"Found {dup_result['duplicate_count']} duplicate timestamp rows "
                f"across {len(dup_result['affected_equipment'])} equipment units."
            )

        # Maximum gap detection
        gap_result = self._check_max_gap(df)
        if gap_result["gaps_exceeded"] > 0:
            violations += gap_result["gaps_exceeded"]
            details["max_gap_violations"] = gap_result
            warnings.append(
                f"Found {gap_result['gaps_exceeded']} timestamp gaps exceeding "
                f"{self._max_gap_minutes} minutes."
            )

        # Future timestamps
        now = pd.Timestamp.now()
        future_mask = df["timestamp"] > now
        if future_mask.any():
            future_count = int(future_mask.sum())
            violations += future_count
            details["future_timestamps"] = future_count
            warnings.append(f"Found {future_count} future timestamps.")

        return GateResult(
            stage_name="TimestampGate",
            passed=violations == 0,
            violations=violations,
            details=details,
            warnings=warnings,
        )

    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> dict[str, Any]:
        dup_mask = df.duplicated(
            subset=["equipment_id", "timestamp", "sensor_name"], keep=False
        )
        dup_count = int(dup_mask.sum())
        affected = (
            df.loc[dup_mask, "equipment_id"].unique().tolist() if dup_count > 0 else []
        )
        return {
            "duplicate_count": dup_count,
            "affected_equipment": affected,
        }

    def _check_max_gap(self, df: pd.DataFrame) -> dict[str, Any]:
        max_gap_td = pd.Timedelta(minutes=self._max_gap_minutes)
        gaps_exceeded = 0
        max_gap_found = pd.Timedelta(0)
        gap_details: list[dict] = []

        for eq_id, grp in df.groupby("equipment_id"):
            grp_sorted = grp.sort_values("timestamp")
            diffs = grp_sorted["timestamp"].diff().dropna()
            exceeding = diffs[diffs > max_gap_td]
            if len(exceeding) > 0:
                gaps_exceeded += len(exceeding)
                if exceeding.max() > max_gap_found:
                    max_gap_found = exceeding.max()
                for idx_val in exceeding.index[:3]:
                    row = grp_sorted.loc[idx_val]
                    gap_details.append(
                        {
                            "equipment_id": eq_id,
                            "timestamp": str(row["timestamp"]),
                            "gap_seconds": exceeding.loc[idx_val].total_seconds(),
                        }
                    )

        return {
            "gaps_exceeded": gaps_exceeded,
            "max_gap_found_minutes": round(max_gap_found.total_seconds() / 60, 2),
            "threshold_minutes": self._max_gap_minutes,
            "sample_gap_details": gap_details[:5],
        }

    # ------------------------------------------------------------------
    # Stage 4 — CrossChannelGate
    # ------------------------------------------------------------------

    def _run_cross_channel_gate(self, df: pd.DataFrame) -> GateResult:
        violations = 0
        pair_details: dict[str, dict] = {}
        warnings: list[str] = []

        # Pivot so each sensor_name is a column
        try:
            pivot = df.pivot_table(
                index=["equipment_id", "timestamp"],
                columns="sensor_name",
                values="value",
                aggfunc="first",
            )
        except Exception:
            return GateResult(
                stage_name="CrossChannelGate",
                passed=True,
                violations=0,
                details={"skipped": "Pivot failed, cannot cross-check sensors."},
            )

        for primary, secondary in self._redundant_pairs:
            if primary not in pivot.columns or secondary not in pivot.columns:
                continue

            pair_key = f"{primary}__vs__{secondary}"
            primary_series = pivot[primary].dropna()
            secondary_series = pivot[secondary].dropna()

            # Align on shared index
            common_idx = primary_series.index.intersection(secondary_series.index)
            if len(common_idx) < 2:
                continue

            p_aligned = primary_series.loc[common_idx]
            s_aligned = secondary_series.loc[common_idx]

            # Expected noise std: derived from registry range width / 6 (rough)
            lo, hi = self._sensor_range_map.get(primary, (0, 1))
            expected_noise_std = max((hi - lo) / 20.0, 0.01)
            threshold = self._cross_channel_factor * expected_noise_std

            diffs = (p_aligned - s_aligned).abs()
            exceeding = diffs[diffs > threshold]
            mismatch_rate = len(exceeding) / len(diffs) if len(diffs) > 0 else 0.0

            # Flatline detection: one sensor stuck while the other varies
            p_flat = _series_is_flatline(p_aligned)
            s_flat = _series_is_flatline(s_aligned)
            flatline_conflict = (p_flat and not s_flat) or (s_flat and not p_flat)

            pair_info = {
                "samples_compared": len(diffs),
                "threshold": round(threshold, 4),
                "exceeding_count": len(exceeding),
                "mismatch_rate": round(mismatch_rate, 4),
                "mean_diff": round(float(diffs.mean()), 4),
                "max_diff": round(float(diffs.max()), 4),
                "primary_flatline": p_flat,
                "secondary_flatline": s_flat,
                "flatline_conflict": flatline_conflict,
            }

            if len(exceeding) > 0 or flatline_conflict:
                violations += len(exceeding)
                if flatline_conflict:
                    violations += 1
                    pair_info["flatline_conflict"] = flatline_conflict

            pair_details[pair_key] = pair_info

            if pair_info["mismatch_rate"] > 0.05:
                warnings.append(
                    f"Cross-channel mismatch: {pair_key} has "
                    f"{pair_info['mismatch_rate']:.2%} mismatch rate "
                    f"({pair_info['exceeding_count']}/{pair_info['samples_compared']})."
                )

        return GateResult(
            stage_name="CrossChannelGate",
            passed=violations == 0,
            violations=violations,
            details={
                "pairs_checked": len(pair_details),
                "pair_details": pair_details,
            },
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Stage 5 — DriftGate
    # ------------------------------------------------------------------

    def _run_drift_gate(self, df: pd.DataFrame) -> GateResult:
        violations = 0
        drift_alerts: list[dict] = []
        warnings: list[str] = []

        if "sensor_name" not in df.columns or "value" not in df.columns:
            return GateResult(
                stage_name="DriftGate",
                passed=True,
                violations=0,
                details={"skipped": "Missing sensor_name or value columns."},
            )

        for (eq_id, sn), grp in df.groupby(["equipment_id", "sensor_name"]):
            key = (str(eq_id), sn)

            # Get or create CUSUM detector
            if key not in self._cusum_detectors:
                ref_mean = self._sensor_ref_means.get(sn)
                if ref_mean is None:
                    ref_mean = float(grp["value"].mean())
                self._cusum_detectors[key] = CUSUMDetector(
                    target_mean=ref_mean,
                    k=self._cusum_k,
                    h=self._cusum_h,
                )

            detector = self._cusum_detectors[key]
            grp_sorted = grp.sort_values("timestamp")
            for _, row in grp_sorted.iterrows():
                result = detector.update(float(row["value"]))
                if result.any_drift:
                    direction = "up" if result.drift_up else "down"
                    drift_alerts.append(
                        {
                            "equipment_id": str(eq_id),
                            "sensor_name": sn,
                            "direction": direction,
                            "timestamp": str(row["timestamp"]),
                            "c_plus": result.c_plus,
                            "c_minus": result.c_minus,
                        }
                    )
                    violations += 1
                    # Reset after detection so subsequent batches get a clean start
                    detector.reset()

        if drift_alerts:
            warnings.append(
                f"Drift detected in {violations} readings across "
                f"{len(set(d['sensor_name'] for d in drift_alerts))} sensors."
            )

        return GateResult(
            stage_name="DriftGate",
            passed=violations == 0,
            violations=violations,
            details={
                "drift_alerts": drift_alerts[:20],
                "total_detectors_active": len(self._cusum_detectors),
            },
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Management helpers
    # ------------------------------------------------------------------

    def reset_cusum(self, sensor_name: str, equipment_id: str) -> None:
        """Reset CUSUM drift tracking for a specific sensor+equipment pair.

        Call after sensor maintenance, recalibration, or replacement.

        Args:
            sensor_name: Sensor identifier (e.g. "vibration_x_a").
            equipment_id: Equipment unit identifier (e.g. "CP-0001").
        """
        key = (equipment_id, sensor_name)
        if key in self._cusum_detectors:
            self._cusum_detectors[key].reset()
            logger.info(
                "CUSUM reset for equipment=%s, sensor=%s",
                equipment_id,
                sensor_name,
            )
        else:
            logger.warning(
                "No CUSUM state found for equipment=%s, sensor=%s. Nothing reset.",
                equipment_id,
                sensor_name,
            )

    def set_reference_statistics(
        self,
        df: pd.DataFrame,
        sensor_name: str,
    ) -> None:
        """Calibrate CUSUM reference parameters from healthy baseline data.

        Computes the expected mean and noise from a reference period
        (e.g., first 24 hours after maintenance) and updates all existing
        detectors for the given sensor.

        Args:
            df: Reference DataFrame with healthy-operation data.
            sensor_name: Sensor to calibrate.
        """
        stats = compute_reference_statistics(df, sensor_name)
        mean_val = stats["mean"]
        k_val = stats["k"]

        self._sensor_ref_means[sensor_name] = mean_val

        for (eq_id, sn), detector in self._cusum_detectors.items():
            if sn == sensor_name:
                detector.target_mean = mean_val
                detector.k = k_val
                detector.h = self._cusum_h
                detector.reset()

        logger.info(
            "Reference statistics updated for sensor '%s': mean=%.3f, k=%.3f.",
            sensor_name,
            mean_val,
            k_val,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series_is_flatline(series: pd.Series, min_unique_values: int = 3) -> bool:
    return series.nunique() < min_unique_values
