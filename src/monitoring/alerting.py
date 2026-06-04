"""Alert lifecycle management with deduplication and escalation.

Designed for industrial predictive-maintenance use-cases where alert fatigue
from duplicate alarms must be suppressed and operators need a structured
escalation path.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from src.utils.logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def _severity_order(cls) -> dict[AlertSeverity, int]:
        return {cls.LOW: 0, cls.MEDIUM: 1, cls.HIGH: 2, cls.CRITICAL: 3}

    def __lt__(self, other: AlertSeverity) -> bool:
        order = self._severity_order()
        return order[self] < order[other]

    def __le__(self, other: AlertSeverity) -> bool:
        order = self._severity_order()
        return order[self] <= order[other]


class AlertType(Enum):
    DATA_QUALITY = "data_quality"
    SENSOR_DRIFT = "sensor_drift"
    MODEL_DRIFT = "model_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_HEALTH = "system_health"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    equipment_id: str | None
    message: str
    details: dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    _fingerprint: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self._fingerprint:
            self._fingerprint = f"{self.alert_type.value}:{self.equipment_id or 'global'}:{self.message}"


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------


class AlertManager:
    """Manages alert creation, deduplication, acknowledgement, and escalation.

    Parameters
    ----------
    dedup_window:
        Time window within which duplicate alerts (same type + equipment +
        message) are suppressed.  Defaults to 30 minutes.
    """

    def __init__(self, dedup_window: timedelta | None = None) -> None:
        self.alerts: list[Alert] = []
        self._dedup_window = dedup_window or timedelta(minutes=30)

    # ------------------------------------------------------------------
    # Create / suppress
    # ------------------------------------------------------------------

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        equipment_id: str | None,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Alert | None:
        now = datetime.now(timezone.utc)

        if self._is_duplicate(alert_type, equipment_id, message, now):
            _logger.debug(
                "Suppressed duplicate alert type=%s equipment=%s",
                alert_type.value,
                equipment_id,
            )
            return None

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            equipment_id=equipment_id,
            message=message,
            details=details or {},
            timestamp=now,
        )

        self.alerts.append(alert)
        _logger.info(
            "Alert created: id=%s type=%s severity=%s equipment=%s",
            alert.alert_id,
            alert_type.value,
            severity.value,
            equipment_id,
        )

        return alert

    # ------------------------------------------------------------------
    # Acknowledge
    # ------------------------------------------------------------------

    def acknowledge(self, alert_id: str, operator: str) -> bool:
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = operator
                alert.acknowledged_at = datetime.now(timezone.utc)
                _logger.info("Alert %s acknowledged by %s", alert_id, operator)
                return True

        _logger.warning("Alert %s not found or already acknowledged", alert_id)
        return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_active_alerts(
        self,
        equipment_id: str | None = None,
        min_severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        results = [a for a in self.alerts if not a.acknowledged]

        if equipment_id is not None:
            results = [a for a in results if a.equipment_id == equipment_id]

        if min_severity is not None:
            results = [a for a in results if a.severity >= min_severity]

        return sorted(results, key=lambda a: a.timestamp, reverse=True)

    def get_alert_summary(self, hours: int = 24) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [a for a in self.alerts if a.timestamp >= cutoff]

        by_type: dict[str, int] = defaultdict(int)
        by_severity: dict[str, int] = defaultdict(int)

        for a in recent:
            by_type[a.alert_type.value] += 1
            by_severity[a.severity.value] += 1

        return {
            "active_count": sum(1 for a in recent if not a.acknowledged),
            "acknowledged_count": sum(1 for a in recent if a.acknowledged),
            "total_count": len(recent),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "window_hours": hours,
        }

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def escalate(self, alert_id: str) -> AlertSeverity | None:
        escalation_map = {
            AlertSeverity.LOW: AlertSeverity.MEDIUM,
            AlertSeverity.MEDIUM: AlertSeverity.HIGH,
            AlertSeverity.HIGH: AlertSeverity.CRITICAL,
        }

        for alert in self.alerts:
            if alert.alert_id != alert_id:
                continue

            new_severity = escalation_map.get(alert.severity)
            if new_severity is None:
                _logger.warning(
                    "Alert %s is already at CRITICAL — cannot escalate further",
                    alert_id,
                )
                return None

            _logger.info(
                "Alert %s escalated from %s to %s",
                alert_id,
                alert.severity.value,
                new_severity.value,
            )

            alert.severity = new_severity
            return new_severity

        _logger.warning("Alert %s not found for escalation", alert_id)
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_duplicate(
        self,
        alert_type: AlertType,
        equipment_id: str | None,
        message: str,
        now: datetime,
    ) -> bool:
        cutoff = now - self._dedup_window

        for alert in self.alerts:
            if alert.timestamp < cutoff:
                continue
            if alert.alert_type != alert_type:
                continue
            if alert.equipment_id != equipment_id:
                continue
            if alert.message == message:
                return True

        return False
