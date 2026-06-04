"""Pydantic request models for the predictive maintenance API.

All request models use Pydantic v2 with Field validators for robust
input validation at the API boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """A single sensor reading with equipment and temporal context.

    Corresponds to one row in the long-form DataFrame expected by the
    feature pipeline and quality gate.
    """

    timestamp: Annotated[
        datetime,
        Field(
            description="ISO-8601 timestamp of the sensor reading",
            examples=["2024-06-15T14:30:00Z"],
        ),
    ]
    equipment_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description="Unique equipment unit identifier",
            examples=["PUMP-001", "TURB-A12"],
        ),
    ]
    sensor_name: Annotated[
        str,
        Field(
            min_length=1,
            max_length=128,
            description="Sensor channel name (e.g. vibration_x, temperature_a)",
            examples=["vibration_x", "temperature_a", "pressure_out"],
        ),
    ]
    value: Annotated[
        float,
        Field(
            description="Numerical sensor reading value",
        ),
    ]

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_not_be_future(cls, v: datetime) -> datetime:
        """Reject timestamps more than 5 minutes in the future (clock skew tolerance)."""
        now = datetime.now(v.tzinfo) if v.tzinfo else datetime.now()
        if v > now and (v - now).total_seconds() > 300:
            msg = f"Timestamp {v.isoformat()} is more than 5 minutes in the future"
            raise ValueError(msg)
        return v

    @field_validator("equipment_id")
    @classmethod
    def equipment_id_must_be_trimmed(cls, v: str) -> str:
        """Strip whitespace from equipment IDs."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("equipment_id must not be empty after trimming")
        return stripped

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, v: float) -> float:
        """Reject NaN and infinite values at the API boundary."""
        import math

        if math.isnan(v) or math.isinf(v):
            msg = f"Sensor value must be finite, got {v}"
            raise ValueError(msg)
        return v


class PredictionRequest(BaseModel):
    """Request for a single-equipment failure prediction.

    Accepts a set of sensor readings (typically a sliding window) for
    one equipment unit.
    """

    readings: Annotated[
        list[SensorReading],
        Field(
            min_length=1,
            max_length=10_000,
            description="List of sensor readings for one equipment unit",
        ),
    ]

    @field_validator("readings")
    @classmethod
    def single_equipment_only(cls, v: list[SensorReading]) -> list[SensorReading]:
        """Ensure all readings belong to the same equipment unit."""
        equipment_ids = {r.equipment_id for r in v}
        if len(equipment_ids) > 1:
            msg = (
                f"All readings must belong to the same equipment_id; "
                f"found {len(equipment_ids)} distinct IDs: {equipment_ids}"
            )
            raise ValueError(msg)
        return v


class BatchPredictionRequest(BaseModel):
    """Request for batch failure predictions across multiple equipment.

    Limited to 50 equipment units per batch to bound inference latency
    and memory usage.
    """

    MAX_EQUIPMENT_PER_BATCH: int = 50

    readings: Annotated[
        list[SensorReading],
        Field(
            min_length=1,
            max_length=100_000,
            description="List of sensor readings across multiple equipment units",
        ),
    ]

    @field_validator("readings")
    @classmethod
    def limit_equipment_count(cls, v: list[SensorReading]) -> list[SensorReading]:
        """Enforce maximum equipment count per batch."""
        equipment_ids = {r.equipment_id for r in v}
        if len(equipment_ids) > cls.MAX_EQUIPMENT_PER_BATCH:
            msg = (
                f"Batch size limited to {cls.MAX_EQUIPMENT_PER_BATCH} equipment units; "
                f"received {len(equipment_ids)}"
            )
            raise ValueError(msg)
        return v


class ExplainRequest(BaseModel):
    """Request for a prediction with SHAP-based explanation.

    Identical to PredictionRequest but semantically distinct: the
    response will include per-feature SHAP contributions.
    """

    readings: Annotated[
        list[SensorReading],
        Field(
            min_length=1,
            max_length=10_000,
            description="List of sensor readings for one equipment unit",
        ),
    ]

    @field_validator("readings")
    @classmethod
    def single_equipment_only(cls, v: list[SensorReading]) -> list[SensorReading]:
        """Ensure all readings belong to the same equipment unit."""
        equipment_ids = {r.equipment_id for r in v}
        if len(equipment_ids) > 1:
            msg = (
                f"All readings must belong to the same equipment_id; "
                f"found {len(equipment_ids)} distinct IDs: {equipment_ids}"
            )
            raise ValueError(msg)
        return v

