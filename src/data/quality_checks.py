"""Data quality validation functions for synthetic and real datasets.

This module provides basic validation checks used by the 5-stage
DataQualityGate (schema, range, timestamp, cross-channel, drift).

All functions accept pandas DataFrames and return boolean or diagnostic
dictionaries suitable for integration into data pipeline validation steps.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_missing_values(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Check for missing (NaN) values in specified columns.

    Args:
        df: DataFrame to validate.
        columns: List of column names to check. If None, checks all columns.

    Returns:
        Dictionary with:
        - ``"passed"``: bool, True if no missing values found.
        - ``"missing_count"``: dict[str, int], column -> count of NaNs.
        - ``"total_rows"``: int, number of rows checked.
        - ``"missing_fraction"``: float, fraction of total cells missing.

    Example:
        >>> result = check_missing_values(df, ["vibration_x", "health"])
        >>> if not result["passed"]:
        ...     logger.warning("Missing values: %s", result["missing_count"])
    """
    cols = columns if columns is not None else list(df.columns)
    missing_series = df[cols].isna().sum()
    missing = missing_series[missing_series > 0].to_dict()
    total_cells = len(df) * len(cols)
    total_missing = int(missing_series.sum())

    return {
        "passed": total_missing == 0,
        "missing_count": missing,
        "total_rows": len(df),
        "missing_fraction": total_missing / total_cells if total_cells > 0 else 0.0,
    }


def check_value_ranges(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float | None, float | None]],
) -> dict[str, Any]:
    """Check that column values fall within expected physical ranges.

    Each column can have an optional lower bound, upper bound, or both.
    A bound of None means no limit in that direction.

    Args:
        df: DataFrame to validate.
        ranges: Dict mapping column name to (min, max) tuple. Use None
            for unbounded sides. For example:
            ``{"health": (0.0, 1.0), "temperature": (None, 600.0)}``.

    Returns:
        Dictionary with:
        - ``"passed"``: bool, True if all values within ranges.
        - ``"violations"``: dict[str, dict], column -> violation details
          (count, fraction, min/max values seen).
        - ``"columns_checked"``: list[str], columns that were validated.

    Example:
        >>> ranges = {"health": (0.0, 1.0), "vibration_x": (0.0, None)}
        >>> result = check_value_ranges(df, ranges)
    """
    violations: dict[str, dict[str, Any]] = {}
    all_passed = True

    for col, (vmin, vmax) in ranges.items():
        if col not in df.columns:
            logger.warning("Column '%s' not found in DataFrame, skipping range check", col)
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        col_violations = 0
        col_min = float(series.min())
        col_max = float(series.max())

        if vmin is not None and col_min < vmin:
            col_violations += int((series < vmin).sum())
        if vmax is not None and col_max > vmax:
            col_violations += int((series > vmax).sum())

        if col_violations > 0:
            all_passed = False
            violations[col] = {
                "violation_count": col_violations,
                "violation_fraction": col_violations / len(series),
                "observed_min": col_min,
                "observed_max": col_max,
                "expected_range": (vmin, vmax),
            }

    return {
        "passed": all_passed,
        "violations": violations,
        "columns_checked": list(ranges.keys()),
    }


def check_timestamp_monotonicity(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    group_col: str | None = "equipment_id",
) -> dict[str, Any]:
    """Verify that timestamps are monotonically increasing.

    Optionally checks monotonicity per group (e.g., per equipment unit).
    This is critical for time-series ML where temporal ordering must be
    preserved for train/test splits and sequential models.

    Args:
        df: DataFrame to validate.
        timestamp_col: Name of the timestamp column.
        group_col: Optional column to group by before checking. If None,
            checks the entire DataFrame as one sequence.

    Returns:
        Dictionary with:
        - ``"passed"``: bool, True if all timestamps are monotonic.
        - ``"non_monotonic_groups"``: list[str], group IDs with violations.
        - ``"total_groups"``: int, number of groups checked.
        - ``"violation_details"``: dict, per-group violation indices.

    Example:
        >>> result = check_timestamp_monotonicity(df, group_col="equipment_id")
        >>> assert result["passed"], f"Non-monotonic: {result['non_monotonic_groups']}"
    """
    if timestamp_col not in df.columns:
        return {
            "passed": False,
            "non_monotonic_groups": ["__all__"],
            "total_groups": 0,
            "violation_details": {"__all__": f"Column '{timestamp_col}' not found"},
        }

    non_monotonic: list[str] = []
    violation_details: dict[str, Any] = {}

    if group_col and group_col in df.columns:
        groups = df.groupby(group_col)
        for grp_id, grp_df in groups:
            timestamps = grp_df[timestamp_col].dropna()
            diff = np.diff(timestamps.values.astype(np.int64))
            if np.any(diff < 0):
                grp_key = str(grp_id)
                non_monotonic.append(grp_key)
                violation_idx = np.where(diff < 0)[0].tolist()
                violation_details[grp_key] = {
                    "violations_at_indices": violation_idx[:10],
                    "total_violations": len(violation_idx),
                }
        total_groups = len(groups)
    else:
        timestamps = df[timestamp_col].dropna()
        diff = np.diff(timestamps.values.astype(np.int64))
        if np.any(diff < 0):
            non_monotonic.append("__all__")
            violation_idx = np.where(diff < 0)[0].tolist()
            violation_details["__all__"] = {
                "violations_at_indices": violation_idx[:10],
                "total_violations": len(violation_idx),
            }
        total_groups = 1

    return {
        "passed": len(non_monotonic) == 0,
        "non_monotonic_groups": non_monotonic,
        "total_groups": total_groups,
        "violation_details": violation_details,
    }


def detect_flatline_sensor(
    series: pd.Series,
    min_unique_values: int = 3,
) -> bool:
    """Detect if a sensor is stuck or flatlined.

    A flatline sensor emits the same value repeatedly, indicating
    a frozen instrument or data pipeline stall. Detection is based on
    the number of unique values in the series falling below a threshold.

    Args:
        series: Sensor reading series to check.
        min_unique_values: Minimum distinct readings required to
            consider the sensor alive. Default 3.

    Returns:
        True if the sensor appears flatlined (too few unique values).

    Example:
        >>> detect_flatline_sensor(pd.Series([1.0, 1.0, 1.0, 1.0]))
        True
        >>> detect_flatline_sensor(pd.Series([1.0, 1.1, 1.0, 1.2]))
        False
    """
    clean = series.dropna()
    if len(clean) == 0:
        return True
    return clean.nunique() < min_unique_values


def detect_unrealistic_jump(
    series: pd.Series,
    max_pct_change: float = 50.0,
) -> list[int]:
    """Find indices where consecutive readings exhibit unrealistic jumps.

    An unrealistic jump is defined as a percent change between two
    consecutive readings exceeding the configured maximum.

    Args:
        series: Sensor reading series (ordered by time).
        max_pct_change: Maximum allowed percent change between
            consecutive readings. Default 50.0 (i.e., 50%).

    Returns:
        List of row indices (positions in the original series) where
        an unrealistic jump was detected. Returns empty list if none.

    Example:
        >>> s = pd.Series([10.0, 10.5, 100.0, 10.2])
        >>> detect_unrealistic_jump(s, max_pct_change=50.0)
        [2]  # 10.5 -> 100.0 is a ~852% change
    """
    clean = series.dropna().reset_index(drop=True)
    if len(clean) < 2:
        return []

    prev = clean.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = ((clean - prev).abs() / prev.abs()) * 100.0

    spike_positions = np.where(pct_change > max_pct_change)[0]
    return spike_positions.tolist()


def check_sensor_redundancy(
    primary: pd.Series,
    secondary: pd.Series,
    max_diff_pct: float = 10.0,
) -> dict[str, Any]:
    """Compare two redundant sensor readings for agreement.

    In industrial installations, critical measurements are often taken
    by two physically separate sensors. Disagreement between them may
    indicate a failing sensor, wiring fault, or signal conditioning issue.

    Args:
        primary: Primary sensor readings.
        secondary: Secondary (redundant) sensor readings.
        max_diff_pct: Maximum allowable percent difference between
            readings (relative to the primary). Default 10.0.

    Returns:
        Dictionary with keys:
        - ``"passed"``: bool, True if sensors agree within threshold.
        - ``"mean_diff"``: float, mean absolute difference.
        - ``"max_diff"``: float, maximum absolute difference.
        - ``"pct_outside_threshold"``: float, fraction of readings exceeding max_diff_pct.
        - ``"correlation"``: float, Pearson correlation coefficient.
        - ``"samples_compared"``: int, number of overlapping readings.
        - ``"primary_flatline"``: bool, whether primary appears stuck.
        - ``"secondary_flatline"``: bool, whether secondary appears stuck.

    Example:
        >>> result = check_sensor_redundancy(vib_a, vib_b, max_diff_pct=10.0)
        >>> if not result["passed"]:
        ...     logger.warning("Sensor mismatch: %s", result)
    """
    p = primary.dropna()
    s = secondary.dropna()

    if p.index.name is not None or s.index.name is not None:
        common_idx = p.index.intersection(s.index)
    elif len(p) == len(s):
        common_idx = p.index
    else:
        common_idx = p.index.intersection(s.index)

    if len(common_idx) < 2:
        return {
            "passed": True,
            "mean_diff": 0.0,
            "max_diff": 0.0,
            "pct_outside_threshold": 0.0,
            "correlation": 1.0,
            "samples_compared": len(common_idx),
            "primary_flatline": detect_flatline_sensor(primary),
            "secondary_flatline": detect_flatline_sensor(secondary),
        }

    p_aligned = p.loc[common_idx].astype(float)
    s_aligned = s.loc[common_idx].astype(float)

    abs_diff = (p_aligned - s_aligned).abs()

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = (abs_diff / p_aligned.abs()) * 100.0
    outside = (pct_diff > max_diff_pct).sum()
    fraction_outside = outside / len(p_aligned) if len(p_aligned) > 0 else 0.0

    corr = float(p_aligned.corr(s_aligned)) if len(p_aligned) > 1 else 1.0
    if np.isnan(corr):
        corr = 0.0

    return {
        "passed": fraction_outside < 0.05,
        "mean_diff": round(float(abs_diff.mean()), 6),
        "max_diff": round(float(abs_diff.max()), 6),
        "pct_outside_threshold": round(fraction_outside, 4),
        "correlation": round(corr, 4),
        "samples_compared": len(common_idx),
        "primary_flatline": detect_flatline_sensor(primary),
        "secondary_flatline": detect_flatline_sensor(secondary),
    }


def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI measures how much a distribution has shifted from an expected
    (reference) distribution to an actual (current) distribution.
    Commonly used in model monitoring to detect feature drift.

    Interpretation:
        - PSI < 0.1: No significant shift.
        - 0.1 <= PSI < 0.25: Moderate shift, investigate.
        - PSI >= 0.25: Significant shift, action required.

    Args:
        expected: Reference distribution values (baseline/historical).
        actual: Current distribution values to compare.
        bins: Number of bins for discretization. Default 10.

    Returns:
        PSI value (float). Returns 0.0 if either series is empty.

    Example:
        >>> ref = pd.Series(np.random.normal(0, 1, 1000))
        >>> cur = pd.Series(np.random.normal(0.5, 1.2, 1000))
        >>> psi = compute_psi(ref, cur)
    """
    e = expected.dropna()
    a = actual.dropna()

    if len(e) == 0 or len(a) == 0:
        return 0.0

    combined = pd.concat([e, a])
    if combined.nunique() <= 1:
        return 0.0

    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    e_hist, _ = np.histogram(e, bins=bin_edges)
    a_hist, _ = np.histogram(a, bins=bin_edges)

    e_dist = e_hist / len(e)
    a_dist = a_hist / len(a)

    epsilon = 1e-10
    e_dist = np.clip(e_dist, epsilon, None)
    a_dist = np.clip(a_dist, epsilon, None)

    psi_values = (a_dist - e_dist) * np.log(a_dist / e_dist)
    return float(np.sum(psi_values))
