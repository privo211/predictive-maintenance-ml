"""
Degradation feature computations for predictive maintenance.

Functions for extracting trend-based and health-indicator features
from sensor time series. These capture the degradation trajectory
rather than instantaneous values.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy import stats

logger = logging.getLogger(__name__)


def compute_rate_of_change(
    series: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
) -> float:
    """Compute linear regression slope of sensor values over time.

    Uses ordinary least squares to fit value = slope * time + intercept.
    The slope represents the average rate of change over the window.
    Positive slope indicates increasing trend, negative indicates decreasing.

    Args:
        series: 1-D array of sensor readings.
        time: 1-D array of timestamps (must be same length as series,
            numeric type, e.g. seconds or hours since epoch).

    Returns:
        Linear regression slope (units depend on value/time units).
        Returns 0.0 if fewer than 3 valid data points.

    Raises:
        ValueError: If series and time have different lengths.
    """
    if len(series) != len(time):
        msg = (
            f"series length ({len(series)}) must match "
            f"time length ({len(time)})"
        )
        raise ValueError(msg)

    valid = ~np.isnan(series) & ~np.isnan(time)
    x = time[valid]
    y = series[valid]

    if len(x) < 3:
        return 0.0

    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def compute_health_index(
    values: npt.NDArray[np.float64],
    baseline_mean: float,
    baseline_std: float,
    decreasing_is_bad: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute a health index proxy (0-1) from sensor values.

    Maps sensor values to a normalized health score where 1.0 is healthy
    and 0.0 is fully degraded. The mapping uses the baseline (healthy)
    distribution: values within baseline_mean +/- baseline_std are considered
    healthy, while deviations beyond 3 standard deviations are fully degraded.

    Args:
        values: 1-D array of current sensor readings.
        baseline_mean: Mean sensor value during healthy operation.
        baseline_std: Standard deviation during healthy operation.
        decreasing_is_bad: If True, values below baseline indicate degradation
            (e.g. pressure dropping). If False, values above baseline indicate
            degradation (e.g. vibration increasing).

    Returns:
        Array of health indices in [0, 1] with same shape as values.
    """
    if baseline_std == 0.0:
        baseline_std = 1e-6

    z_scores = (values - baseline_mean) / baseline_std

    if decreasing_is_bad:
        # Lower values = worse health
        z_scores = -z_scores

    # Clamp z-scores to [-3, 3] range, then map to [0, 1]
    z_clipped = np.clip(z_scores, -3.0, 3.0)
    health = (z_clipped + 3.0) / 6.0

    return health


def compute_trend_direction(
    series: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
) -> float:
    """Compute Spearman rank correlation between value and time.

    Captures monotonic trend direction irrespective of linearity.
    - +1.0: perfectly increasing trend
    - -1.0: perfectly decreasing trend
    - 0.0: no monotonic trend

    Args:
        series: 1-D array of sensor readings.
        time: 1-D array of timestamps (numeric, same length as series).

    Returns:
        Spearman correlation coefficient in [-1, 1].
        Returns 0.0 if fewer than 4 valid data points or if all values
        are constant (no variance).
    """
    if len(series) != len(time):
        msg = (
            f"series length ({len(series)}) must match "
            f"time length ({len(time)})"
        )
        raise ValueError(msg)

    valid = ~np.isnan(series) & ~np.isnan(time)
    x = time[valid]
    y = series[valid]

    if len(x) < 4:
        return 0.0

    if np.std(y) < 1e-10:
        return 0.0

    corr, _ = stats.spearmanr(x, y)
    if np.isnan(corr):
        return 0.0

    return float(corr)


def compute_cumulative_deviation(
    series: npt.NDArray[np.float64],
    baseline_mean: float,
) -> float:
    """Compute cumulative deviation from baseline over the window.

    Sum of (value - baseline_mean) across all samples. A large positive
    deviation indicates sustained drift above baseline; a large negative
    deviation indicates sustained drift below.

    Args:
        series: 1-D array of sensor readings.
        baseline_mean: Expected baseline (healthy) value for the sensor.

    Returns:
        Cumulative deviation sum. Returns 0.0 if no valid data.
    """
    valid = series[~np.isnan(series)]
    if len(valid) == 0:
        return 0.0

    deviations = valid - baseline_mean
    return float(np.sum(deviations))


def compute_volatility(series: npt.NDArray[np.float64]) -> float:
    """Compute coefficient of variation over the window.

    CV = std / mean. Higher values indicate more relative variability,
    which often precedes equipment failure.

    Args:
        series: 1-D array of sensor readings.

    Returns:
        Coefficient of variation (dimensionless). Returns 0.0 if fewer
        than 3 valid data points or if mean is near zero.
    """
    valid = series[~np.isnan(series)]
    if len(valid) < 3:
        return 0.0

    mean_val = np.mean(valid)
    if abs(mean_val) < 1e-10:
        return 0.0

    return float(np.std(valid, ddof=1) / mean_val)
