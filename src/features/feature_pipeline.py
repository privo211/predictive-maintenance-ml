"""
sklearn-compatible feature engineering pipeline for predictive maintenance.

Transforms raw long-form multi-sensor time-series data into fixed-length
feature vectors for XGBoost training and inference. Handles multi-rate
sensor data, missing values, and computes time-domain, frequency-domain,
and degradation features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from features.degradation_features import (
    compute_cumulative_deviation,
    compute_rate_of_change,
    compute_trend_direction,
    compute_volatility,
)

logger = logging.getLogger(__name__)

_TIME_DOMAIN_AGGS: list[str] = ["mean", "std", "min", "max", "skew", "kurtosis"]
_FFT_FEATURE_NAMES: list[str] = [
    "dominant_freq",
    "spectral_centroid",
    "spectral_energy",
    "spectral_entropy",
]
_MIN_FFT_SAMPLES: int = 64
_MIN_FFT_SAMPLE_RATE: float = 10.0  # Hz
_RESAMPLE_RULE: str = "1min"
_FFILL_LIMIT: int = 5


def _rolling_skew(series: pd.Series) -> float:
    vals = series.dropna().values
    if len(vals) < 3:
        return np.nan
    return float(stats.skew(vals))


def _rolling_kurtosis(series: pd.Series) -> float:
    vals = series.dropna().values
    if len(vals) < 4:
        return np.nan
    return float(stats.kurtosis(vals))


def _infer_sensor_sampling_rates(
    X: pd.DataFrame,
) -> dict[str, float]:
    """Infer effective sampling rate (Hz) for each sensor from raw data.

    Uses the median time delta between consecutive samples within each
    equipment group, averaged across equipment.
    """
    rates: dict[str, float] = {}
    for sensor_name in sorted(X["sensor_name"].unique()):
        sensor_df = X[X["sensor_name"] == sensor_name]
        if len(sensor_df) < 2:
            rates[sensor_name] = 0.0
            continue

        all_deltas: list[float] = []
        for _, grp in sensor_df.groupby("equipment_id"):
            if len(grp) < 2:
                continue
            ts_sorted = grp["timestamp"].sort_values()
            deltas = ts_sorted.diff().dropna()
            if len(deltas) == 0:
                continue
            median_delta_sec = deltas.median().total_seconds()
            if median_delta_sec > 0:
                all_deltas.append(1.0 / median_delta_sec)

        if all_deltas:
            rates[sensor_name] = float(np.mean(all_deltas))
        else:
            rates[sensor_name] = 0.0

    return rates


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Transforms raw sensor readings into fixed-length feature vectors.

    Pipeline stages:
    1. Pivot long-form -> wide-form (sensor_name columns)
    2. Resample to uniform time grid (1-minute intervals)
    3. Handle missing values (forward-fill then median imputation)
    4. Extract time-domain features per sensor
    5. Extract frequency-domain features per sensor (FFT)
    6. Extract degradation features
    7. Standardize (StandardScaler)
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        feature_config_path: str | None = None,
    ) -> None:
        """Initialize the feature pipeline.

        Args:
            window_sizes: Rolling window sizes in minutes. Default: [5, 15, 30, 60].
            feature_config_path: Optional path to YAML config with feature flags.
        """
        self.window_sizes = window_sizes or [5, 15, 30, 60]
        self.feature_config_path = feature_config_path

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeaturePipeline:
        """Fit scalers and imputers on training data.

        Learns per-sensor median values for imputation and fits the
        StandardScaler on the full feature matrix computed from X.

        Args:
            X: Long-form DataFrame with columns:
                timestamp, equipment_id, sensor_name, value.
            y: Ignored (required by sklearn API).

        Returns:
            self
        """
        self._validate_input(X)

        sensor_names = sorted(X["sensor_name"].unique())
        self._sensor_names_ = sensor_names
        self._window_sizes_ = sorted(self.window_sizes)
        self._sampling_rates_ = _infer_sensor_sampling_rates(X)

        self._load_feature_config()

        resampled = self._pivot_and_resample(X)

        self._median_values_: pd.Series = resampled.median()

        feature_df = self._extract_all_features(X, resampled)

        self._feature_names_ = list(feature_df.columns)
        self._scaler_ = StandardScaler()
        feature_values = feature_df.values.astype(np.float64)
        self._scaler_.fit(feature_values)

        logger.info(
            "FeaturePipeline fitted: %d sensors, %d features, %d windows",
            len(sensor_names),
            len(self._feature_names_),
            len(self._window_sizes_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Transform raw data to feature matrix.

        Args:
            X: Long-form DataFrame with columns:
                timestamp, equipment_id, sensor_name, value.

        Returns:
            2-D numpy array of shape (n_samples, n_features).
        """
        self._validate_input(X)

        resampled = self._pivot_and_resample(X)

        missing_sensors = set(self._sensor_names_) - set(resampled.columns)
        for sensor in missing_sensors:
            resampled[sensor] = np.nan
            logger.debug("Sensor '%s' missing from transform input, filling NaN", sensor)

        feature_df = self._extract_all_features(X, resampled)

        expected_cols = set(self._feature_names_)
        actual_cols = set(feature_df.columns)
        for col in expected_cols - actual_cols:
            feature_df[col] = np.nan

        feature_df = feature_df[self._feature_names_]

        feature_values = feature_df.values.astype(np.float64)
        return self._scaler_.transform(feature_values)

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names generated by the pipeline.

        Must call ``fit`` before calling this method.

        Returns:
            List of feature name strings in column order.

        Raises:
            AttributeError: If ``fit`` has not been called yet.
        """
        if not hasattr(self, "_feature_names_"):
            msg = (
                "FeaturePipeline has not been fitted. "
                "Call fit() before get_feature_names()."
            )
            raise AttributeError(msg)
        return list(self._feature_names_)

    # ------------------------------------------------------------------
    # Internal: validation and config
    # ------------------------------------------------------------------

    def _validate_input(self, X: pd.DataFrame) -> None:
        required = {"timestamp", "equipment_id", "sensor_name", "value"}
        missing = required - set(X.columns)
        if missing:
            msg = f"Input DataFrame missing required columns: {missing}"
            raise ValueError(msg)
        if len(X) == 0:
            raise ValueError("Input DataFrame is empty")

    def _load_feature_config(self) -> None:
        self._fft_enabled_sensors: set[str] = set()
        if self.feature_config_path is None:
            return
        config_path = Path(self.feature_config_path)
        if not config_path.exists():
            logger.warning("Feature config not found: %s", config_path)
            return
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        disabled_fft = set(config.get("disable_fft_sensors", []))
        self._fft_enabled_sensors = set(self._sensor_names_) - disabled_fft

    # ------------------------------------------------------------------
    # Internal: pivot and resample
    # ------------------------------------------------------------------

    def _pivot_and_resample(self, X: pd.DataFrame) -> pd.DataFrame:
        wide = X.pivot_table(
            index=["timestamp", "equipment_id"],
            columns="sensor_name",
            values="value",
            aggfunc="mean",
        )

        if wide.empty:
            return wide

        wide = wide.reset_index("equipment_id")
        resampled = (
            wide.groupby("equipment_id")
            .resample(_RESAMPLE_RULE)
            .mean()
        )

        return resampled

    # ------------------------------------------------------------------
    # Internal: feature extraction orchestration
    # ------------------------------------------------------------------

    def _extract_all_features(
        self,
        X_raw: pd.DataFrame,
        resampled: pd.DataFrame,
    ) -> pd.DataFrame:
        filled = self._fill_missing(resampled)

        eq_ids = filled.index.get_level_values("equipment_id").unique()
        all_feature_dfs: list[pd.DataFrame] = []

        for eq_id in eq_ids:
            eq_data = filled.xs(eq_id, level="equipment_id")
            eq_features = self._extract_equipment_features(X_raw, eq_id, eq_data)
            eq_features["equipment_id"] = eq_id
            all_feature_dfs.append(eq_features)

        if not all_feature_dfs:
            return pd.DataFrame()

        result = pd.concat(all_feature_dfs)
        result = result.set_index("equipment_id", append=True)
        return result

    def _fill_missing(self, resampled: pd.DataFrame) -> pd.DataFrame:
        filled = resampled.groupby(level="equipment_id").ffill(limit=_FFILL_LIMIT)

        if hasattr(self, "_median_values_"):
            for col in filled.columns:
                if col in self._median_values_.index:
                    filled[col] = filled[col].fillna(self._median_values_[col])
                else:
                    filled[col] = filled[col].fillna(0.0)
        else:
            filled = filled.fillna(0.0)

        return filled

    # ------------------------------------------------------------------
    # Internal: per-equipment feature extraction
    # ------------------------------------------------------------------

    def _extract_equipment_features(
        self,
        X_raw: pd.DataFrame,
        eq_id: str,
        eq_data: pd.DataFrame,
    ) -> pd.DataFrame:
        sensors = sorted(
            [c for c in eq_data.columns if c in self._sensor_names_]
        )

        td_features = self._extract_time_domain(eq_data, sensors)
        fft_features = self._extract_frequency_domain(X_raw, eq_id, sensors, eq_data)
        deg_features = self._extract_degradation(eq_data, sensors)

        all_features = pd.concat([td_features, fft_features, deg_features], axis=1)
        return all_features

    # ------------------------------------------------------------------
    # Internal: time-domain features
    # ------------------------------------------------------------------

    def _extract_time_domain(
        self,
        eq_data: pd.DataFrame,
        sensors: list[str],
    ) -> pd.DataFrame:
        feature_columns: dict[str, pd.Series] = {}

        for window_min in self._window_sizes_:
            window_str = f"{window_min}min"
            rolled = eq_data.rolling(window_str, min_periods=1)

            for sensor in sensors:
                if sensor not in eq_data.columns:
                    continue
                prefix = f"{sensor}_w{window_min}"
                sensor_rolled = rolled[sensor]

                feature_columns[f"{prefix}_mean"] = sensor_rolled.mean()
                feature_columns[f"{prefix}_std"] = sensor_rolled.std()
                feature_columns[f"{prefix}_min"] = sensor_rolled.min()
                feature_columns[f"{prefix}_max"] = sensor_rolled.max()
                feature_columns[f"{prefix}_skew"] = sensor_rolled.apply(
                    _rolling_skew, raw=False
                )
                feature_columns[f"{prefix}_kurtosis"] = sensor_rolled.apply(
                    _rolling_kurtosis, raw=False
                )

        return pd.DataFrame(feature_columns, index=eq_data.index)

    # ------------------------------------------------------------------
    # Internal: frequency-domain features
    # ------------------------------------------------------------------

    def _extract_frequency_domain(
        self,
        X_raw: pd.DataFrame,
        eq_id: str,
        sensors: list[str],
        eq_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        fft_features_per_sensor: dict[str, dict[str, float]] = {}

        for sensor in sensors:
            rate = self._sampling_rates_.get(sensor, 0.0)
            if rate < _MIN_FFT_SAMPLE_RATE:
                continue

            if self._fft_enabled_sensors and sensor not in self._fft_enabled_sensors:
                continue

            sensor_mask = (
                (X_raw["equipment_id"] == eq_id)
                & (X_raw["sensor_name"] == sensor)
            )
            values = X_raw.loc[sensor_mask, "value"].dropna().values

            if len(values) < _MIN_FFT_SAMPLES:
                continue

            fft_result = self._compute_fft(values, rate)
            for fft_name in _FFT_FEATURE_NAMES:
                key = f"{sensor}_fft_{fft_name}"
                fft_features_per_sensor[key] = fft_result[fft_name]

        if not fft_features_per_sensor:
            return pd.DataFrame(index=(eq_data.index if eq_data is not None else pd.DatetimeIndex([])))

        idx = eq_data.index if eq_data is not None else pd.DatetimeIndex([])
        return pd.DataFrame(
            {k: pd.Series(v, index=idx) for k, v in fft_features_per_sensor.items()}
        )

    @staticmethod
    def _compute_fft(
        signal: npt.NDArray[np.float64],
        sample_rate_hz: float,
    ) -> dict[str, float]:
        n = len(signal)
        signal_detrended = signal - np.mean(signal)

        fft = np.fft.rfft(signal_detrended)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

        if len(freqs) < 2:
            return dict.fromkeys(_FFT_FEATURE_NAMES, np.nan)

        dominant_idx = int(np.argmax(magnitude[1:]) + 1)
        dominant_freq = float(freqs[dominant_idx])

        mag_sum = float(np.sum(magnitude))
        if mag_sum > 1e-12:
            spectral_centroid = float(np.sum(freqs * magnitude) / mag_sum)
        else:
            spectral_centroid = np.nan

        spectral_energy = float(np.sum(magnitude**2))

        psd = magnitude**2
        psd_sum = float(np.sum(psd))
        if psd_sum > 1e-12:
            psd_norm = psd / psd_sum
            spectral_entropy = float(
                -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            )
        else:
            spectral_entropy = np.nan

        return {
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "spectral_energy": spectral_energy,
            "spectral_entropy": spectral_entropy,
        }

    # ------------------------------------------------------------------
    # Internal: degradation features
    # ------------------------------------------------------------------

    def _extract_degradation(
        self,
        eq_data: pd.DataFrame,
        sensors: list[str],
    ) -> pd.DataFrame:
        if eq_data.empty:
            return pd.DataFrame()

        time_seconds = eq_data.index.astype(np.int64) / 1e9
        time_numeric = (time_seconds - time_seconds.min()).values

        feature_columns: dict[str, pd.Series] = {}

        for window_min in self._window_sizes_:
            window_str = f"{window_min}min"
            for sensor in sensors:
                if sensor not in eq_data.columns:
                    continue
                prefix = f"{sensor}_deg_w{window_min}"

                sensor_series = eq_data[sensor]
                baseline_mean = sensor_series.mean()
                baseline_std = sensor_series.std()

                feature_columns[f"{prefix}_rate_of_change"] = self._rolling_degradation(
                    sensor_series, time_numeric, compute_rate_of_change, window_str
                )
                feature_columns[f"{prefix}_trend_direction"] = self._rolling_degradation(
                    sensor_series, time_numeric, compute_trend_direction, window_str
                )
                feature_columns[f"{prefix}_cumulative_deviation"] = (
                    self._rolling_degradation_scalar(
                        sensor_series,
                        compute_cumulative_deviation,
                        window_str,
                        baseline_mean=baseline_mean,
                    )
                )
                feature_columns[f"{prefix}_volatility"] = self._rolling_degradation_scalar(
                    sensor_series,
                    compute_volatility,
                    window_str,
                )

        return pd.DataFrame(feature_columns, index=eq_data.index)

    def _rolling_degradation(
        self,
        series: pd.Series,
        time_numeric: npt.NDArray[np.float64],
        func: Any,
        window_str: str,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=series.index, dtype=np.float64)
        rolled = series.rolling(window_str, min_periods=3)
        time_series = pd.Series(time_numeric, index=series.index)

        for i in range(len(series)):
            end = series.index[i]
            start = end - pd.Timedelta(window_str)

            mask = (series.index >= start) & (series.index <= end)
            window_vals = series.loc[mask].dropna().values
            window_time = time_series.loc[mask].dropna().values

            if len(window_vals) < 3:
                continue

            result.iloc[i] = func(window_vals, window_time)

        return result

    def _rolling_degradation_scalar(
        self,
        series: pd.Series,
        func: Any,
        window_str: str,
        **kwargs: Any,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=series.index, dtype=np.float64)

        for i in range(len(series)):
            end = series.index[i]
            start = end - pd.Timedelta(window_str)

            mask = (series.index >= start) & (series.index <= end)
            window_vals = series.loc[mask].dropna().values

            if len(window_vals) < 3:
                continue

            result.iloc[i] = func(window_vals, **kwargs)

        return result
