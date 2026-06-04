"""CUSUM-based sensor drift detection for data quality gates.

Provides a standalone CUSUM (Cumulative Sum) detector that tracks
gradual shifts in sensor readings. Used by the DriftGate stage of
DataQualityGate to detect calibration drift before it corrupts
ML model predictions.

The CUSUM algorithm maintains two cumulative sums (C+ and C-) that
accumulate positive and negative deviations from a target mean. When
either sum exceeds a decision interval H, a drift alarm is raised.

Algorithm reference:
    Page, E.S. (1954). "Continuous Inspection Schemes." Biometrika.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CUSUMResult:
    """Result of a single CUSUM update step.

    Attributes:
        c_plus: Upper cumulative sum (positive deviations).
        c_minus: Lower cumulative sum (negative deviations).
        drift_up: Whether an upward drift has been detected (c_plus >= h).
        drift_down: Whether a downward drift has been detected (c_minus >= h).
        samples_since_reset: Number of samples processed since last reset.
    """

    c_plus: float
    c_minus: float
    drift_up: bool
    drift_down: bool
    samples_since_reset: int

    @property
    def any_drift(self) -> bool:
        """True if either upward or downward drift is detected."""
        return self.drift_up or self.drift_down


class CUSUMDetector:
    """Two-sided tabular CUSUM detector for sensor drift monitoring.

    Tracks both positive and negative deviations from a target mean.
    Designed for streaming use: call ``update()`` with each new sensor
    reading to continuously monitor for gradual shifts.

    The sensitivity is controlled by two parameters:
    - ``k`` (drift_magnitude): The minimum shift magnitude to detect,
      typically set to 0.5 * sigma (half the expected noise std).
    - ``h`` (decision_interval): The alarm threshold. Higher values
      reduce false alarms but increase detection delay. Typical: 4.0-5.0.

    Attributes:
        target_mean: Expected sensor value under normal operation.
        k: Reference value (allowable slack), typically 0.5*sigma.
        h: Decision interval (alarm threshold).
    """

    def __init__(self, target_mean: float, k: float = 0.5, h: float = 5.0) -> None:
        """Initialize CUSUM detector.

        Args:
            target_mean: Expected sensor value under healthy conditions.
            k: Allowable deviation magnitude (reference value).
               Default 0.5 detects ~0.5-sigma shifts.
            h: Decision interval for alarm. Default 5.0.
        """
        if h <= 0:
            raise ValueError(f"Decision interval 'h' must be positive, got {h}")
        if k <= 0:
            raise ValueError(f"Reference value 'k' must be positive, got {k}")

        self.target_mean = target_mean
        self.k = k
        self.h = h

        self._c_plus: float = 0.0
        self._c_minus: float = 0.0
        self._samples: int = 0

    def update(self, value: float) -> CUSUMResult:
        """Feed one sensor reading into the CUSUM detector.

        Args:
            value: Current sensor reading (scalar).

        Returns:
            CUSUMResult with current CUSUM statistics and drift flags.
        """
        # Standardized deviation from target
        deviation = value - self.target_mean

        # Two-sided CUSUM update
        self._c_plus = max(0.0, self._c_plus + deviation - self.k)
        self._c_minus = max(0.0, self._c_minus - deviation - self.k)
        self._samples += 1

        return CUSUMResult(
            c_plus=round(self._c_plus, 6),
            c_minus=round(self._c_minus, 6),
            drift_up=self._c_plus >= self.h,
            drift_down=self._c_minus >= self.h,
            samples_since_reset=self._samples,
        )

    def reset(self) -> None:
        """Reset both cumulative sums and sample counter to zero.

        Call this after sensor maintenance, recalibration, or when a
        known operational change occurs that shifts the baseline.
        """
        self._c_plus = 0.0
        self._c_minus = 0.0
        self._samples = 0
        logger.debug(
            "CUSUM reset: target_mean=%.3f, k=%.3f, h=%.3f",
            self.target_mean,
            self.k,
            self.h,
        )

    def is_drift_detected(self) -> bool:
        """Check if either CUSUM statistic has crossed the alarm threshold.

        Returns:
            True if drift is currently detected (C+ >= H or C- >= H).
        """
        return self._c_plus >= self.h or self._c_minus >= self.h

    @property
    def state(self) -> dict:
        """Return current detector state as a serializable dict."""
        return {
            "target_mean": self.target_mean,
            "k": self.k,
            "h": self.h,
            "c_plus": round(self._c_plus, 6),
            "c_minus": round(self._c_minus, 6),
            "samples": self._samples,
            "drift_detected": self.is_drift_detected(),
        }


def compute_reference_statistics(
    df: pd.DataFrame,
    sensor_name: str,
    mean_col: str = "value",
    sensor_col: str = "sensor_name",
) -> dict:
    """Compute reference statistics from healthy (baseline) data.

    Extracts mean, standard deviation, and derived CUSUM parameters
    (k, h) from a DataFrame of healthy-operation sensor readings.
    These serve as the baseline for drift detection.

    Args:
        df: DataFrame with healthy-operation sensor data.
        sensor_name: Name of the sensor to compute stats for.
        mean_col: Column containing sensor values (default: "value").
        sensor_col: Column containing sensor names (default: "sensor_name").

    Returns:
        Dictionary with keys:
        - ``"mean"``: float, arithmetic mean of healthy values.
        - ``"std"``: float, standard deviation of healthy values.
        - ``"count"``: int, number of samples used.
        - ``"k"``: float, recommended CUSUM reference value (0.5 * std).
        - ``"h"``: float, recommended CUSUM decision interval (5.0 default).
        - ``"sensor_name"``: str, the sensor name.
    """
    if sensor_col not in df.columns or mean_col not in df.columns:
        raise KeyError(
            f"Required columns '{sensor_col}' and/or '{mean_col}' not in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    sensor_data = df.loc[df[sensor_col] == sensor_name, mean_col].dropna()

    if len(sensor_data) == 0:
        raise ValueError(
            f"No data found for sensor '{sensor_name}'. "
            f"Available sensors: {df[sensor_col].unique().tolist()}"
        )

    mean_val = float(sensor_data.mean())
    std_val = float(sensor_data.std(ddof=1)) if len(sensor_data) > 1 else 0.0
    count = len(sensor_data)

    # Standard CUSUM parameterization: k = 0.5 * sigma (half standard deviation)
    # This detects a shift of 1 sigma within approximately 10 samples
    k = 0.5 * std_val if std_val > 0 else 0.1

    return {
        "sensor_name": sensor_name,
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "count": count,
        "k": round(k, 6),
        "h": 5.0,
    }
