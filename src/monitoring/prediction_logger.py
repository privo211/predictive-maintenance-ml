"""Async prediction and ground-truth logger with in-memory fallback.

When a database URL is provided predictions are persisted to PostgreSQL (or
any SQLAlchemy-compatible backend).  Without a URL the logger falls back to a
simple in-memory list — useful for local development and CI environments.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from utils.logger import get_logger

_logger = get_logger(__name__)


class PredictionLogger:
    """Logs predictions and (optionally) ground-truth labels.

    Parameters
    ----------
    db_url:
        Async database URL.  If *None* the logger operates entirely in
        memory and predictions are not persisted across restarts.
    engine:
        Pre-built async engine (alternative to ``db_url``).  Useful when you
        already have an engine from :func:`src.utils.database.get_engine`.
    """

    def __init__(
        self,
        db_url: str | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        self._store: list[dict[str, Any]] = []

        if engine is not None:
            self.engine: AsyncEngine | None = engine
        elif db_url is not None:
            from sqlalchemy.ext.asyncio import create_async_engine

            self.engine = create_async_engine(db_url, echo=False)
        else:
            self.engine = None

    @property
    def is_persistent(self) -> bool:
        return self.engine is not None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    async def log_prediction(
        self,
        equipment_id: str,
        prediction: float,
        predicted_class: int,
        model_version: str,
        latency_ms: float,
        quality_passed: bool,
    ) -> str:
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        if self.engine is not None:
            await self._log_to_db(
                prediction_id=prediction_id,
                equipment_id=equipment_id,
                prediction=prediction,
                predicted_class=predicted_class,
                model_version=model_version,
                latency_ms=latency_ms,
                quality_passed=quality_passed,
                timestamp=timestamp,
            )
        else:
            self._store.append({
                "prediction_id": prediction_id,
                "equipment_id": equipment_id,
                "prediction": prediction,
                "predicted_class": predicted_class,
                "model_version": model_version,
                "latency_ms": latency_ms,
                "quality_passed": quality_passed,
                "timestamp": timestamp.isoformat(),
                "actual_failure": None,
            })

        _logger.debug("Logged prediction %s for equipment %s", prediction_id, equipment_id)
        return prediction_id

    async def log_ground_truth(self, prediction_id: str, actual_failure: bool) -> bool:
        if self.engine is not None:
            return await self._update_ground_truth_db(prediction_id, actual_failure)

        for entry in self._store:
            if entry["prediction_id"] == prediction_id:
                entry["actual_failure"] = actual_failure
                _logger.debug("Ground truth recorded for prediction %s", prediction_id)
                return True

        _logger.warning("Prediction %s not found for ground-truth update", prediction_id)
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_recent_predictions(
        self,
        equipment_id: str,
        hours: int = 24,
    ) -> list[dict[str, Any]]:
        if self.engine is not None:
            return await self._query_recent_db(equipment_id, hours)

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        return [
            p for p in self._store
            if p["equipment_id"] == equipment_id and p["timestamp"] >= cutoff
        ]

    async def get_prediction_stats(
        self,
        model_version: str | None = None,
        days: int = 7,
    ) -> dict[str, Any]:
        if self.engine is not None:
            return await self._query_stats_db(model_version, days)

        rows = self._store
        if model_version:
            rows = [r for r in rows if r["model_version"] == model_version]

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = [r for r in rows if r["timestamp"] >= cutoff]

        if not rows:
            return {"total_predictions": 0}

        predictions = [r["prediction"] for r in rows]
        classes = [r["predicted_class"] for r in rows]
        latencies = [r["latency_ms"] for r in rows]

        return {
            "total_predictions": len(rows),
            "avg_prediction": sum(predictions) / len(predictions),
            "class_1_rate": sum(classes) / len(classes),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "quality_pass_rate": sum(1 for r in rows if r["quality_passed"]) / len(rows),
        }

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    async def _log_to_db(
        self,
        prediction_id: str,
        equipment_id: str,
        prediction: float,
        predicted_class: int,
        model_version: str,
        latency_ms: float,
        quality_passed: bool,
        timestamp: datetime,
    ) -> None:
        assert self.engine is not None

        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    """INSERT INTO predictions (
                        prediction_id, equipment_id, prediction,
                        predicted_class, model_version, latency_ms,
                        quality_passed, time
                    ) VALUES (
                        :prediction_id, :equipment_id, :prediction,
                        :predicted_class, :model_version, :latency_ms,
                        :quality_passed, :time
                    )"""
                ),
                {
                    "prediction_id": prediction_id,
                    "equipment_id": equipment_id,
                    "prediction": prediction,
                    "predicted_class": predicted_class,
                    "model_version": model_version,
                    "latency_ms": latency_ms,
                    "quality_passed": quality_passed,
                    "time": timestamp,
                },
            )

    async def _update_ground_truth_db(self, prediction_id: str, actual_failure: bool) -> bool:
        assert self.engine is not None

        async with self.engine.begin() as conn:
            result = await conn.execute(
                text(
                    """UPDATE predictions
                       SET actual_failure = :actual_failure,
                           ground_truth_recorded_at = :now
                       WHERE prediction_id = :prediction_id"""
                ),
                {
                    "prediction_id": prediction_id,
                    "actual_failure": actual_failure,
                    "now": datetime.now(timezone.utc),
                },
            )
            updated = result.rowcount

        if updated:
            _logger.debug("Ground truth recorded for prediction %s", prediction_id)
            return True

        _logger.warning("Prediction %s not found for ground-truth update", prediction_id)
        return False

    async def _query_recent_db(
        self,
        equipment_id: str,
        hours: int,
    ) -> list[dict[str, Any]]:
        assert self.engine is not None

        async with self.engine.begin() as conn:
            result = await conn.execute(
                text(
                    """SELECT * FROM predictions
                       WHERE equipment_id = :equipment_id
                         AND timestamp >= :cutoff
                       ORDER BY timestamp DESC"""
                ),
                {
                    "equipment_id": equipment_id,
                    "cutoff": datetime.now(timezone.utc) - timedelta(hours=hours),
                },
            )
            rows = result.fetchall()

        return [dict(row._mapping) for row in rows]

    async def _query_stats_db(
        self,
        model_version: str | None,
        days: int,
    ) -> dict[str, Any]:
        assert self.engine is not None

        base = """SELECT
                      COUNT(*) AS total_predictions,
                      AVG(prediction) AS avg_prediction,
                      AVG(CAST(predicted_class AS FLOAT)) AS class_1_rate,
                      AVG(latency_ms) AS avg_latency_ms,
                      AVG(CAST(quality_passed AS INT)) AS quality_pass_rate
                    FROM predictions
                    WHERE timestamp >= :cutoff"""

        params: dict[str, Any] = {
            "cutoff": datetime.now(timezone.utc) - timedelta(days=days),
        }

        if model_version:
            base += " AND model_version = :model_version"
            params["model_version"] = model_version

        async with self.engine.begin() as conn:
            result = await conn.execute(text(base), params)
            row = result.fetchone()

        if row is None or row.total_predictions == 0:
            return {"total_predictions": 0}

        return {
            "total_predictions": int(row.total_predictions),
            "avg_prediction": float(row.avg_prediction or 0),
            "class_1_rate": float(row.class_1_rate or 0),
            "avg_latency_ms": float(row.avg_latency_ms or 0),
            "quality_pass_rate": float(row.quality_pass_rate or 0),
        }
