"""
Physics-informed synthetic data generator for industrial predictive maintenance.

This module provides a complete pipeline for generating realistic multi-sensor
time-series data from simulated industrial equipment. Each equipment unit is
modelled as a DigitalTwin that degrades over time with configurable fault
injection. The output is clean pandas DataFrames ready for feature engineering.

Architecture:
    EquipmentConfig -> DigitalTwin -> SyntheticDataGenerator -> pd.DataFrame

Supported equipment types:
    - Centrifugal pump (single-stage, end-suction)
    - Steam turbine (condensing, multi-stage)
    - Electric motor (induction, squirrel-cage)

Six fault modes with distinct physics-based signatures are supported across
all equipment types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from data.degradation_models import (
    add_fault_signature,
    bearing_fault_frequencies,
    exponential_degradation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EquipmentType(Enum):
    """Industrial equipment categories supported by the generator."""

    CENTRIFUGAL_PUMP = auto()
    STEAM_TURBINE = auto()
    ELECTRIC_MOTOR = auto()


class FaultMode(Enum):
    """Fault modes with physics-informed signatures for condition monitoring.

    Each fault mode corresponds to a distinct physical degradation mechanism
    that manifests in specific sensor channels. The enum values are the
    string keys used by add_fault_signature() in degradation_models.py.
    """

    BEARING_WEAR = "bearing_wear"
    CAVITATION = "cavitation"
    MISALIGNMENT = "misalignment"
    SEAL_LEAKAGE = "seal_leakage"
    IMBALANCE = "imbalance"
    NORMAL = "normal"


# ---------------------------------------------------------------------------
# Preset sensor definitions per equipment type
# ---------------------------------------------------------------------------

_SENSOR_SUITE: dict[EquipmentType, list[str]] = {
    EquipmentType.CENTRIFUGAL_PUMP: [
        "vibration_x_a",
        "vibration_x_b",
        "vibration_y_a",
        "vibration_y_b",
        "vibration_z_a",
        "vibration_z_b",
        "temperature_a",
        "temperature_b",
        "pressure_out",
        "flow_rate",
        "rpm",
        "current_a",
    ],
    EquipmentType.STEAM_TURBINE: [
        "vibration_x_a",
        "vibration_x_b",
        "vibration_y_a",
        "vibration_y_b",
        "vibration_z_a",
        "vibration_z_b",
        "temperature_a",
        "temperature_b",
        "steam_temp",
        "steam_pressure",
        "power_output",
        "rpm",
        "current_a",
    ],
    EquipmentType.ELECTRIC_MOTOR: [
        "vibration_x_a",
        "vibration_x_b",
        "vibration_y_a",
        "vibration_y_b",
        "vibration_z_a",
        "vibration_z_b",
        "temperature_a",
        "temperature_b",
        "rpm",
        "current_a",
        "voltage",
        "power_factor",
    ],
}

# Fault mode to affected sensor channels mapping.
# Each fault primarily affects specific sensor channels with defined physics.
_FAULT_SENSOR_MAP: dict[FaultMode, list[str]] = {
    FaultMode.BEARING_WEAR: ["vibration_x_a", "vibration_y_a", "vibration_z_a"],
    FaultMode.IMBALANCE: ["vibration_x_a", "vibration_y_a"],
    FaultMode.MISALIGNMENT: ["vibration_x_a", "vibration_y_a", "vibration_z_a"],
    FaultMode.CAVITATION: ["pressure_out", "flow_rate", "vibration_y_a"],
    FaultMode.SEAL_LEAKAGE: ["pressure_out", "flow_rate"],
    FaultMode.NORMAL: [],
}

# Health thresholds below which each fault can activate.
# Higher threshold = fault can activate earlier in the degradation curve.
_FAULT_HEALTH_THRESHOLDS: dict[FaultMode, float] = {
    FaultMode.IMBALANCE: 0.85,
    FaultMode.MISALIGNMENT: 0.75,
    FaultMode.BEARING_WEAR: 0.55,
    FaultMode.CAVITATION: 0.45,
    FaultMode.SEAL_LEAKAGE: 0.35,
    FaultMode.NORMAL: 1.0,
}


# ---------------------------------------------------------------------------
# Equipment configuration
# ---------------------------------------------------------------------------


@dataclass
class EquipmentConfig:
    """Configuration for a single equipment unit in the digital twin simulation.

    Attributes:
        equipment_id: Unique identifier for the equipment unit.
        equipment_type: Type of industrial equipment.
        base_rpm: Nominal operating speed in revolutions per minute.
        base_temp: Nominal operating temperature in degrees Celsius.
        base_pressure: Nominal operating pressure in bar (gauge).
        base_vibration: Nominal vibration level in mm/s RMS.
        degradation_rate: Base health decline per hour. Empirical values:
            1e-5 (slow, ~10yr life), 5e-4 (moderate, ~2000hr life),
            1e-3 (accelerated, ~1000hr life).
        fault_modes: Fault modes this equipment can experience.
        fault_probability: Per-step probability of fault activation when
            health is below the mode-specific threshold.
        noise_std: Per-sensor Gaussian noise standard deviation. Keys
            should match sensor names from the sensor suite. Unspecified
            sensors default to 0.01.
        seed: RNG seed for reproducible degradation and fault injection.
        bearing_geometry: Optional bearing geometry dict with keys
            'n_balls', 'ball_diameter_mm', 'pitch_diameter_mm',
            'contact_angle_rad'. Used for bearing fault frequency
            computation. If not provided, defaults for a 6309 deep-groove
            ball bearing are used.
    """

    equipment_id: str
    equipment_type: EquipmentType
    base_rpm: float
    base_temp: float
    base_pressure: float
    base_vibration: float
    degradation_rate: float = 5e-5
    fault_modes: list[FaultMode] = field(default_factory=list)
    fault_probability: float = 0.01
    noise_std: dict[str, float] = field(default_factory=dict)
    seed: int | None = None
    bearing_geometry: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.fault_probability < 0.0 or self.fault_probability > 1.0:
            msg = f"fault_probability must be in [0, 1], got {self.fault_probability}"
            raise ValueError(msg)
        if self.degradation_rate < 0.0:
            msg = f"degradation_rate must be non-negative, got {self.degradation_rate}"
            raise ValueError(msg)

    @property
    def sensor_names(self) -> list[str]:
        """Return ordered list of sensor names for this equipment type."""
        return _SENSOR_SUITE.get(self.equipment_type, [])

    def get_noise_std(self, sensor: str) -> float:
        """Get noise standard deviation for a sensor, defaulting to 0.01."""
        return self.noise_std.get(sensor, 0.01)


# ---------------------------------------------------------------------------
# Default bearing geometry (6309 deep-groove ball bearing)
# ---------------------------------------------------------------------------

_DEFAULT_BEARING = {
    "n_balls": 8,
    "ball_diameter_mm": 17.462,
    "pitch_diameter_mm": 72.5,
    "contact_angle_rad": 0.0,
}


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------


class DigitalTwin:
    """Physics-informed simulation of a single industrial equipment unit.

    Each DigitalTwin models the degradation trajectory of one equipment
    unit, capturing health decline, fault activation, and multi-sensor
    readings over time. The simulation uses seeded RNG for reproducibility.

    The health state h(t) evolves as:
        h(t+dt) = h(t) - degradation_rate * dt - sigma * sqrt(dt) * N(0,1)
        clamped to [0.0, 1.0]

    where sigma is a random walk volatility proportional to the
    degradation_rate (default sigma = 0.1 * degradation_rate).
    """

    def __init__(self, config: EquipmentConfig) -> None:
        """Initialize digital twin with equipment configuration.

        Args:
            config: Equipment configuration specifying type, base operating
                parameters, degradation model, and fault settings.
        """
        self.config = config
        self.health: float = 1.0
        self.active_faults: list[FaultMode] = []
        self._rng = np.random.default_rng(config.seed)
        self._time_elapsed: float = 0.0
        self._fault_history: list[tuple[float, FaultMode]] = []

        bearing = config.bearing_geometry or _DEFAULT_BEARING
        self._bearing_freqs = bearing_fault_frequencies(
            rpm=config.base_rpm,
            n_balls=int(bearing["n_balls"]),
            ball_diameter=float(bearing["ball_diameter_mm"]),
            pitch_diameter=float(bearing["pitch_diameter_mm"]),
            contact_angle=float(bearing["contact_angle_rad"]),
        )

        self._random_walk_sigma = 0.1 * config.degradation_rate

        logger.info(
            "DigitalTwin '%s' initialized (type=%s, rpm=%.0f, degradation=%.2e/h)",
            config.equipment_id,
            config.equipment_type.name,
            config.base_rpm,
            config.degradation_rate,
        )

    def step(self, dt_hours: float) -> dict[str, float]:
        """Advance the simulation by one time step.

        Performs health degradation, fault injection, and sensor reading
        generation for a single timestep.

        Args:
            dt_hours: Time step duration in hours.

        Returns:
            Dictionary mapping sensor name to reading value, plus keys:
            ``"health"`` (current health), ``"fault"`` (active fault mode
            name or "normal"), ``"time_hours"`` (elapsed simulation time).
        """
        if dt_hours <= 0.0:
            msg = f"dt_hours must be positive, got {dt_hours}"
            raise ValueError(msg)

        # Health degradation
        degradation = (
            self.config.degradation_rate * dt_hours
            + self._random_walk_sigma * np.sqrt(dt_hours) * self._rng.normal()
        )
        self.health = max(0.0, min(1.0, self.health - degradation))

        self._time_elapsed += dt_hours

        # Fault injection
        self.active_faults = self._inject_faults()

        # Generate sensor readings
        readings = self._generate_readings(dt_hours)

        fault_label = (
            self.active_faults[0].value if self.active_faults else "normal"
        )

        return {
            **readings,
            "health": self.health,
            "fault": fault_label,
            "time_hours": self._time_elapsed,
        }

    def _inject_faults(self) -> list[FaultMode]:
        """Determine which faults are active based on health and probability.

        Fault activation is probabilistic: when health drops below a
        mode-specific threshold, there is a chance (per step) that the
        fault becomes active. Once active, a fault persists until health
        reaches zero. Multiple faults can be simultaneously active.

        Returns:
            List of currently active fault modes.
        """
        active: list[FaultMode] = []

        for fault_mode in self.config.fault_modes:
            if fault_mode is FaultMode.NORMAL:
                continue

            threshold = _FAULT_HEALTH_THRESHOLDS.get(fault_mode, 1.0)

            if fault_mode in self.active_faults:
                active.append(fault_mode)
            elif self.health <= threshold:
                if self._rng.random() < self.config.fault_probability:
                    active.append(fault_mode)
                    self._fault_history.append((self._time_elapsed, fault_mode))
                    logger.info(
                        "Fault activated: %s on %s at t=%.1fh (health=%.3f)",
                        fault_mode.name,
                        self.config.equipment_id,
                        self._time_elapsed,
                        self.health,
                    )

        if not active and self.health > 0.95:
            return []

        return active

    def _generate_readings(self, dt_hours: float) -> dict[str, float]:
        """Generate all sensor readings for the current timestep.

        Primary sensors (no _b suffix) are computed from physics base values.
        Redundant _b sensors are derived from their _a counterpart with
        independently sampled noise to simulate physically separate instruments.
        """
        etype = self.config.equipment_type
        dt_seconds = dt_hours * 3600.0
        h = self.health

        readings: dict[str, float] = {}

        # Phase 1: generate primary (non-_b) sensor readings
        for sensor in self.config.sensor_names:
            if sensor.endswith("_b"):
                continue

            base_val = self._base_sensor_value(sensor)

            if self._is_vibration_sensor(sensor):
                health_factor = self._vibration_health_factor(h)
                reading = base_val * health_factor
            elif self._is_temperature_sensor(sensor):
                health_factor = self._temperature_health_factor(h)
                reading = base_val * health_factor
            elif self._is_pressure_sensor(sensor):
                health_factor = self._pressure_health_factor(h)
                reading = base_val * health_factor
            elif sensor == "flow_rate":
                health_factor = h
                reading = base_val * health_factor
            else:
                reading = base_val * (0.9 + 0.1 * h)

            reading = self._apply_fault_to_reading(sensor, reading, dt_seconds)
            noise_std = self.config.get_noise_std(sensor)
            reading += self._rng.normal(scale=noise_std)

            readings[sensor] = reading

        # Phase 2: generate redundant _b sensors based on _a counterpart
        for sensor in self.config.sensor_names:
            if not sensor.endswith("_b"):
                continue

            primary_sensor = sensor[:-2] + "_a"  # vibration_x_b -> vibration_x_a
            base_reading = readings.get(primary_sensor, self._base_sensor_value(sensor))

            # Apply same fault effects
            reading = self._apply_fault_to_reading(sensor, base_reading, dt_seconds)

            # Independent noise sample (different instrument, similar reading)
            noise_std = self.config.get_noise_std(sensor)
            reading += self._rng.normal(scale=noise_std * 0.8)

            readings[sensor] = reading

        return readings

    def _base_sensor_value(self, sensor: str) -> float:
        """Return the nominal operating value for a sensor."""
        etype = self.config.equipment_type
        cfg = self.config

        if self._is_vibration_sensor(sensor):
            return cfg.base_vibration
        if self._is_temperature_sensor(sensor):
            return cfg.base_temp
        if self._is_pressure_sensor(sensor):
            if etype == EquipmentType.STEAM_TURBINE:
                return 12.0
            return cfg.base_pressure
        if sensor == "flow_rate":
            if etype == EquipmentType.CENTRIFUGAL_PUMP:
                return 250.0
            return 0.0
        if sensor == "current_a":
            if etype == EquipmentType.CENTRIFUGAL_PUMP:
                return 45.0
            if etype == EquipmentType.ELECTRIC_MOTOR:
                return 32.0
            if etype == EquipmentType.STEAM_TURBINE:
                return 20.0
            return 0.0
        if sensor == "rpm" or sensor == "speed_rpm":
            return cfg.base_rpm
        if sensor == "power_output":
            return 50.0
        if sensor == "steam_temp":
            return 500.0
        if sensor == "steam_pressure":
            return 12.0
        if sensor == "voltage":
            return 440.0
        if sensor == "power_factor":
            return 0.88
        return 0.0

    def _apply_fault_to_reading(
        self, sensor: str, reading: float, dt_seconds: float
    ) -> float:
        """Apply active fault signatures to a single sensor reading.

        Creates a 1-sample signal array, applies fault signatures with
        appropriate parameters, and returns the modified value. For faults
        that affect vibration channels, fault signatures use the sample
        interval to compute correct frequency-domain patterns.
        """
        if not self.active_faults:
            return reading

        signal = np.array([reading], dtype=np.float64)

        for fault in self.active_faults:
            affected_sensors = _FAULT_SENSOR_MAP.get(fault, [])
            if sensor not in affected_sensors:
                continue

            params = self._fault_params(fault, dt_seconds)
            # Scale fault intensity by (1.0 - health) so faults grow with degradation
            severity = 1.0 - self.health
            scaled_params = {
                k: v * severity if k == "amplitude" else v
                for k, v in params.items()
            }
            signal = add_fault_signature(signal, fault.value, scaled_params)

        return float(signal[0])

    def _fault_params(self, fault: FaultMode, dt_seconds: float) -> dict[str, Any]:
        """Build parameter dict for a fault mode's signature function."""
        cfg = self.config

        base_params: dict[str, Any] = {
            "rpm": cfg.base_rpm,
            "dt": dt_seconds,
        }

        if fault is FaultMode.BEARING_WEAR:
            base_params["bearing_freqs"] = self._bearing_freqs
            base_params["amplitude"] = 0.8
        elif fault is FaultMode.CAVITATION:
            base_params["frequency"] = 200.0
            base_params["amplitude"] = 1.0
        elif fault is FaultMode.MISALIGNMENT:
            base_params["amplitude"] = 1.0
        elif fault is FaultMode.SEAL_LEAKAGE:
            base_params["offset"] = 0.3
            base_params["ramp_rate"] = 0.05
        elif fault is FaultMode.IMBALANCE:
            base_params["amplitude"] = 1.5

        return base_params

    # -- Health factor functions --------------------------------------------------

    @staticmethod
    def _vibration_health_factor(health: float) -> float:
        """Vibration increases as health degrades (inverse relationship)."""
        return 1.0 + 2.0 * (1.0 - health)

    @staticmethod
    def _temperature_health_factor(health: float) -> float:
        """Temperature rises slightly as health degrades (friction)."""
        return 1.0 + 0.5 * (1.0 - health)

    @staticmethod
    def _pressure_health_factor(health: float) -> float:
        """Pressure drops as health degrades (leaks, reduced efficiency)."""
        return 1.0 - 0.3 * (1.0 - health)

    # -- Sensor type helpers ------------------------------------------------------

    @staticmethod
    def _is_vibration_sensor(name: str) -> bool:
        return name.startswith("vibration_")

    @staticmethod
    def _is_temperature_sensor(name: str) -> bool:
        return name.startswith("temperature_") or "temp" in name

    @staticmethod
    def _is_pressure_sensor(name: str) -> bool:
        return name.startswith("pressure_") or "pressure" in name


# ---------------------------------------------------------------------------
# Synthetic Data Generator (Orchestrator)
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """Orchestrator for generating fleet-scale synthetic sensor datasets.

    Manages multiple DigitalTwin instances running in parallel (logically),
    collecting their outputs into unified long-form DataFrames suitable for
    downstream feature engineering and model training.

    Usage:
        >>> gen = SyntheticDataGenerator(seed=42)
        >>> df = gen.generate_fleet(num_units=10, simulation_hours=1000,
        ...                         sample_rate_hz=1.0/3600.0)
        >>> gen.save_to_csv(df, "data/synthetic_fleet.csv")
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the generator with a global RNG seed.

        Args:
            seed: Seed passed to each DigitalTwin's RNG. Individual twin
                seeds are derived as seed + unit_index for reproducibility
                across parallel runs.
        """
        self._global_seed = seed
        self._rng = np.random.default_rng(seed)

    # -- Fleet generation --------------------------------------------------------

    def generate_fleet(
        self,
        num_units: int,
        simulation_hours: float,
        sample_rate_hz: float,
        failure_fraction: float = 0.15,
    ) -> pd.DataFrame:
        """Generate a full fleet simulation dataset.

        Creates num_units DigitalTwin instances with a mix of equipment
        types and degradation profiles. A fraction of units are assigned
        accelerated degradation to produce failure events.

        Args:
            num_units: Total number of equipment units to simulate.
            simulation_hours: Duration of simulation in hours.
            sample_rate_hz: Sampling frequency in Hz. For example, 1/3600.0
                for hourly sampling, 0.1 for 10-second sampling.
            failure_fraction: Fraction of units (0.0 to 1.0) that will
                experience accelerated degradation leading to failure.
                Remaining units degrade slowly (healthy baseline).

        Returns:
            Long-form DataFrame with columns:
            - ``timestamp``: datetime64[ns], absolute time of each sample.
            - ``equipment_id``: str, equipment unit identifier.
            - ``sensor_name``: str, name of the sensor channel.
            - ``value``: float64, the sensor reading.
            - ``health``: float64, equipment health at that timestamp.
            - ``fault_label``: int64, 0 = healthy (health >= 0.2),
              1 = failure imminent (health < 0.2).
            - ``equipment_type``: str, equipment type name.

        Raises:
            ValueError: If failure_fraction is not in [0, 1].
        """
        if not 0.0 <= failure_fraction <= 1.0:
            msg = f"failure_fraction must be in [0, 1], got {failure_fraction}"
            raise ValueError(msg)

        dt_hours = 1.0 / (sample_rate_hz * 3600.0) if sample_rate_hz > 0 else 1.0
        num_steps = int(simulation_hours / dt_hours)

        n_failing = max(1, int(num_units * failure_fraction))
        n_healthy = num_units - n_failing

        configs = self._build_fleet_configs(n_healthy, n_failing)

        logger.info(
            "Generating fleet: %d units (%d failing) over %.1f hours "
            "at %.4f Hz (%d steps per unit)",
            num_units,
            n_failing,
            simulation_hours,
            sample_rate_hz,
            num_steps,
        )

        all_records: list[dict[str, Any]] = []

        for i, config in enumerate(configs):
            twin = DigitalTwin(config)
            unit_records = self._run_simulation(twin, dt_hours, num_steps)
            all_records.extend(unit_records)
            logger.debug(
                "Unit %d/%d (%s) complete: health=%.3f",
                i + 1, num_units, config.equipment_id, twin.health,
            )

        df = pd.DataFrame(all_records)

        # Encode fault_label as int: 0 = healthy, 1 = failure imminent
        df["fault_label"] = (df["health"] < 0.2).astype(np.int64)

        logger.info(
            "Fleet generation complete: %d total records, %d units, "
            "%.1f%% failure-imminent samples",
            len(df),
            df["equipment_id"].nunique(),
            100.0 * df["fault_label"].mean(),
        )

        return df

    def generate_normal_only(
        self,
        num_units: int,
        simulation_hours: float,
        sample_rate_hz: float,
    ) -> pd.DataFrame:
        """Generate healthy equipment data with no fault injection.

        All units use slow degradation rates and have fault_modes set to
        [NORMAL] only. This produces a clean baseline dataset for training
        anomaly detection models or establishing nominal operating envelopes.

        Args:
            num_units: Number of healthy equipment units.
            simulation_hours: Duration of simulation in hours.
            sample_rate_hz: Sampling frequency in Hz.

        Returns:
            Long-form DataFrame with same schema as generate_fleet(),
            but all fault_label values will be 0.
        """
        dt_hours = 1.0 / (sample_rate_hz * 3600.0) if sample_rate_hz > 0 else 1.0
        num_steps = int(simulation_hours / dt_hours)

        preset_types = [
            EquipmentType.CENTRIFUGAL_PUMP,
            EquipmentType.STEAM_TURBINE,
            EquipmentType.ELECTRIC_MOTOR,
        ]

        configs: list[EquipmentConfig] = []
        for i in range(num_units):
            etype = preset_types[i % len(preset_types)]
            config = self._make_normal_config(etype, i)
            configs.append(config)

        logger.info(
            "Generating normal-only fleet: %d units over %.1f hours",
            num_units,
            simulation_hours,
        )

        all_records: list[dict[str, Any]] = []
        for i, config in enumerate(configs):
            twin = DigitalTwin(config)
            unit_records = self._run_simulation(twin, dt_hours, num_steps)
            all_records.extend(unit_records)
            logger.debug(
                "Normal unit %d/%d complete", i + 1, num_units,
            )

        df = pd.DataFrame(all_records)
        df["fault_label"] = (df["health"] < 0.2).astype(np.int64)
        return df

    # -- I/O ----------------------------------------------------------------------

    @staticmethod
    def save_to_csv(df: pd.DataFrame, path: str | Path) -> None:
        """Save generated data to a CSV file.

        Args:
            df: DataFrame to save (as returned by generate_fleet).
            path: File path for the output CSV.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d records to %s", len(df), path)

    # -- Preset factories ---------------------------------------------------------

    @staticmethod
    def load_preset(preset_name: str) -> EquipmentConfig:
        """Return a predefined equipment configuration.

        Supported presets (case-insensitive):
        - ``"default_pump"``: Single-stage centrifugal pump, 1780 RPM,
          moderate degradation rate, all fault modes.
        - ``"default_turbine"``: Condensing steam turbine, 3600 RPM,
          slow degradation, bearing/misalignment/imbalance faults.
        - ``"default_motor"``: Squirrel-cage induction motor, 1480 RPM,
          moderate degradation, bearing/misalignment/imbalance faults.

        Args:
            preset_name: Name of the preset configuration.

        Returns:
            Fully populated EquipmentConfig.

        Raises:
            ValueError: If preset_name is not recognized.
        """
        name = preset_name.lower().strip()

        if name == "default_pump":
            return EquipmentConfig(
                equipment_id="PUMP-001",
                equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
                base_rpm=1780.0,
                base_temp=65.0,
                base_pressure=4.0,
                base_vibration=2.8,
                degradation_rate=5e-5,
                fault_modes=[
                    FaultMode.BEARING_WEAR,
                    FaultMode.CAVITATION,
                    FaultMode.MISALIGNMENT,
                    FaultMode.SEAL_LEAKAGE,
                    FaultMode.IMBALANCE,
                ],
                fault_probability=0.01,
                noise_std={
                    "vibration_x": 0.05,
                    "vibration_y": 0.05,
                    "vibration_z": 0.05,
                    "temperature_bearing_drive": 0.2,
                    "temperature_bearing_non_drive": 0.2,
                    "pressure_discharge": 0.02,
                    "pressure_suction": 0.02,
                    "flow_rate": 0.5,
                    "current_motor": 0.1,
                },
            )

        if name == "default_turbine":
            return EquipmentConfig(
                equipment_id="TURB-001",
                equipment_type=EquipmentType.STEAM_TURBINE,
                base_rpm=3600.0,
                base_temp=500.0,
                base_pressure=60.0,
                base_vibration=3.5,
                degradation_rate=3e-5,
                fault_modes=[
                    FaultMode.BEARING_WEAR,
                    FaultMode.MISALIGNMENT,
                    FaultMode.IMBALANCE,
                ],
                fault_probability=0.008,
                noise_std={
                    "vibration_x": 0.07,
                    "vibration_y": 0.07,
                    "vibration_z": 0.07,
                    "temperature_bearing_1": 0.5,
                    "temperature_bearing_2": 0.5,
                    "pressure_inlet": 0.3,
                    "pressure_outlet": 0.3,
                    "temperature_steam_inlet": 1.0,
                    "speed_rpm": 1.0,
                    "power_output": 0.2,
                },
            )

        if name == "default_motor":
            return EquipmentConfig(
                equipment_id="MOTOR-001",
                equipment_type=EquipmentType.ELECTRIC_MOTOR,
                base_rpm=1480.0,
                base_temp=55.0,
                base_pressure=1.0,
                base_vibration=1.5,
                degradation_rate=4e-5,
                fault_modes=[
                    FaultMode.BEARING_WEAR,
                    FaultMode.MISALIGNMENT,
                    FaultMode.IMBALANCE,
                ],
                fault_probability=0.01,
                noise_std={
                    "vibration_x": 0.04,
                    "vibration_y": 0.04,
                    "vibration_z": 0.04,
                    "temperature_winding": 0.3,
                    "temperature_bearing_drive": 0.3,
                    "temperature_bearing_non_drive": 0.3,
                    "current_phase_a": 0.2,
                    "current_phase_b": 0.2,
                    "current_phase_c": 0.2,
                    "speed_rpm": 0.5,
                },
            )

        available = ["default_pump", "default_turbine", "default_motor"]
        msg = f"Unknown preset '{preset_name}'. Available: {available}"
        raise ValueError(msg)

    # -- Internal helpers ---------------------------------------------------------

    def _build_fleet_configs(
        self, n_healthy: int, n_failing: int
    ) -> list[EquipmentConfig]:
        """Build a list of equipment configs for mixed healthy/failing fleet."""
        types_cycle = [
            EquipmentType.CENTRIFUGAL_PUMP,
            EquipmentType.STEAM_TURBINE,
            EquipmentType.ELECTRIC_MOTOR,
        ]
        all_configs: list[EquipmentConfig] = []

        for i in range(n_healthy):
            etype = types_cycle[i % len(types_cycle)]
            config = self._make_normal_config(etype, i)
            all_configs.append(config)

        for i in range(n_failing):
            etype = types_cycle[i % len(types_cycle)]
            config = self._make_failing_config(etype, n_healthy + i)
            all_configs.append(config)

        self._rng.shuffle(all_configs)
        return all_configs

    def _make_normal_config(
        self, etype: EquipmentType, unit_idx: int
    ) -> EquipmentConfig:
        """Create a config for a healthy unit with slow degradation."""
        seed = (self._global_seed or 0) + unit_idx if self._global_seed is not None else None
        preset_name = {
            EquipmentType.CENTRIFUGAL_PUMP: "default_pump",
            EquipmentType.STEAM_TURBINE: "default_turbine",
            EquipmentType.ELECTRIC_MOTOR: "default_motor",
        }[etype]

        config = self.load_preset(preset_name)
        config.equipment_id = f"{etype.name[:4]}-N{unit_idx:03d}"
        config.degradation_rate *= 0.2
        config.fault_modes = [FaultMode.NORMAL]
        config.fault_probability = 0.0
        config.seed = seed
        return config

    def _make_failing_config(
        self, etype: EquipmentType, unit_idx: int
    ) -> EquipmentConfig:
        """Create a config for a unit that will fail within the simulation."""
        seed = (self._global_seed or 0) + unit_idx if self._global_seed is not None else None
        preset_name = {
            EquipmentType.CENTRIFUGAL_PUMP: "default_pump",
            EquipmentType.STEAM_TURBINE: "default_turbine",
            EquipmentType.ELECTRIC_MOTOR: "default_motor",
        }[etype]

        config = self.load_preset(preset_name)
        config.equipment_id = f"{etype.name[:4]}-F{unit_idx:03d}"
        # Accelerated degradation: 5-15x the default rate
        config.degradation_rate *= self._rng.uniform(5.0, 15.0)
        config.fault_probability = min(0.05, config.fault_probability * 3.0)
        config.seed = seed
        return config

    @staticmethod
    def _run_simulation(
        twin: DigitalTwin,
        dt_hours: float,
        num_steps: int,
    ) -> list[dict[str, Any]]:
        """Run a single DigitalTwin simulation and collect all records.

        Returns a list of dicts, where each dict represents one sample
        with equipment_id, sensor_name, value, health, timestamp, and
        equipment_type.
        """
        base_time = pd.Timestamp("2024-01-01 00:00:00")
        records: list[dict[str, Any]] = []

        for step_idx in range(num_steps):
            step_result = twin.step(dt_hours)

            timestamp = base_time + pd.Timedelta(hours=twin._time_elapsed)

            for sensor in twin.config.sensor_names:
                records.append({
                    "equipment_id": twin.config.equipment_id,
                    "sensor_name": sensor,
                    "value": step_result[sensor],
                    "health": step_result["health"],
                    "timestamp": timestamp,
                    "equipment_type": twin.config.equipment_type.name,
                })

        return records
