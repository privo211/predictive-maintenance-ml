"""Synthetic industrial data generation and data quality module.

Provides a physics-informed data generation pipeline for predictive
maintenance, along with a 5-stage data quality firewall that validates
sensor data before it reaches ML models.

Key components:
    - Degradation models and digital twin simulation.
    - Fleet-scale synthetic data orchestration.
    - Schema, range, timestamp, cross-channel, and drift validation gates.
    - CUSUM-based sensor calibration drift detection.
    - Population Stability Index (PSI) for distribution drift monitoring.
"""

from data.drift_detector import (
    CUSUMDetector,
    CUSUMResult,
    compute_reference_statistics,
)
from data.quality_checks import (
    check_missing_values,
    check_sensor_redundancy,
    check_timestamp_monotonicity,
    check_value_ranges,
    compute_psi,
    detect_flatline_sensor,
    detect_unrealistic_jump,
)
from data.quality_gate import (
    DataQualityGate,
    GateResult,
    QualityReport,
)
from data.synthetic_generator import (
    DigitalTwin,
    EquipmentConfig,
    EquipmentType,
    FaultMode,
    SyntheticDataGenerator,
)
from data.degradation_models import (
    add_fault_signature,
    bearing_fault_frequencies,
    exponential_degradation,
    gamma_degradation,
    piecewise_linear_degradation,
    wiener_degradation,
)

__all__ = [
    # Generator classes
    "SyntheticDataGenerator",
    "DigitalTwin",
    "EquipmentConfig",
    # Enums
    "EquipmentType",
    "FaultMode",
    # Degradation models
    "exponential_degradation",
    "piecewise_linear_degradation",
    "wiener_degradation",
    "gamma_degradation",
    # Fault analysis
    "bearing_fault_frequencies",
    "add_fault_signature",
    # Quality checks
    "check_missing_values",
    "check_value_ranges",
    "check_timestamp_monotonicity",
    "detect_flatline_sensor",
    "detect_unrealistic_jump",
    "check_sensor_redundancy",
    "compute_psi",
    # Quality gate
    "DataQualityGate",
    "QualityReport",
    "GateResult",
    # Drift detection
    "CUSUMDetector",
    "CUSUMResult",
    "compute_reference_statistics",
]
