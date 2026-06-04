"""
Lightweight feature store for caching computed feature vectors.

Supports SQLite-backed persistent storage for local development
and in-memory dict storage when no database URL is provided.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """Persistent cache for computed feature vectors.

    Stores feature vectors keyed by (equipment_id, timestamp) so that
    expensive feature computations can be reused across training runs.
    Feature vectors are serialized as JSON arrays in SQLite.

    Usage:
        >>> store = FeatureStore("features.db")
        >>> store.store("PUMP-001", pd.Timestamp.now(), features, names)
        >>> df = store.retrieve("PUMP-001", start_time, end_time)
        >>> latest = store.get_latest("PUMP-001")
    """

    def __init__(self, db_url: str | None = None) -> None:
        """Initialize the feature store.

        Args:
            db_url: Path to SQLite database file. If None, uses an
                in-memory dictionary (no persistence across sessions).
        """
        self._db_url = db_url
        self._in_memory: dict[str, dict[str, Any]] = {}

        if self._db_url is not None:
            db_path = Path(self._db_url)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))
            self._init_schema()
            logger.info("FeatureStore initialized with SQLite backend at %s", db_path)
        else:
            self._conn = None
            logger.info("FeatureStore initialized with in-memory backend")

    def _init_schema(self) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_vectors (
                equipment_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                feature_vector TEXT NOT NULL,
                feature_names TEXT NOT NULL,
                PRIMARY KEY (equipment_id, timestamp)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feature_eq_time
            ON feature_vectors (equipment_id, timestamp)
            """
        )
        self._conn.commit()

    def store(
        self,
        equipment_id: str,
        timestamp: pd.Timestamp,
        features: npt.NDArray[np.float64],
        feature_names: list[str],
    ) -> None:
        """Store a feature vector for a given equipment and timestamp.

        Args:
            equipment_id: Equipment unit identifier.
            timestamp: Timestamp associated with this feature vector.
            features: 1-D numpy array of feature values.
            feature_names: Ordered list of feature names.
        """
        ts_str = timestamp.isoformat()

        if self._conn is not None:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO feature_vectors
                    (equipment_id, timestamp, feature_vector, feature_names)
                VALUES (?, ?, ?, ?)
                """,
                (
                    equipment_id,
                    ts_str,
                    json.dumps(features.tolist()),
                    json.dumps(feature_names),
                ),
            )
            self._conn.commit()
        else:
            key = f"{equipment_id}:{ts_str}"
            self._in_memory[key] = {
                "features": features.copy(),
                "feature_names": feature_names.copy(),
            }

    def retrieve(
        self,
        equipment_id: str,
        from_time: pd.Timestamp,
        to_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Retrieve feature vectors for an equipment within a time range.

        Args:
            equipment_id: Equipment unit identifier.
            from_time: Start of time range (inclusive).
            to_time: End of time range (inclusive).

        Returns:
            DataFrame with feature columns indexed by timestamp.
            Returns empty DataFrame if no data found.
        """
        from_str = from_time.isoformat()
        to_str = to_time.isoformat()

        records: list[dict[str, Any]] = []

        if self._conn is not None:
            cursor = self._conn.execute(
                """
                SELECT timestamp, feature_vector, feature_names
                FROM feature_vectors
                WHERE equipment_id = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (equipment_id, from_str, to_str),
            )
            for row in cursor:
                features = np.array(json.loads(row[1]), dtype=np.float64)
                names = json.loads(row[2])
                record = {"timestamp": pd.Timestamp(row[0])}
                record.update(dict(zip(names, features)))
                records.append(record)
        else:
            for key, entry in self._in_memory.items():
                eq_id, ts_str = key.split(":", 1)
                if eq_id != equipment_id:
                    continue
                ts = pd.Timestamp(ts_str)
                if from_time <= ts <= to_time:
                    record = {"timestamp": ts}
                    record.update(
                        dict(zip(entry["feature_names"], entry["features"]))
                    )
                    records.append(record)
            records.sort(key=lambda r: r["timestamp"])

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.set_index("timestamp")
        return df

    def get_latest(
        self, equipment_id: str
    ) -> tuple[npt.NDArray[np.float64], list[str]] | None:
        """Retrieve the most recent feature vector for an equipment.

        Args:
            equipment_id: Equipment unit identifier.

        Returns:
            Tuple of (feature_array, feature_names) or None if no data.
        """
        if self._conn is not None:
            cursor = self._conn.execute(
                """
                SELECT feature_vector, feature_names
                FROM feature_vectors
                WHERE equipment_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (equipment_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            features = np.array(json.loads(row[0]), dtype=np.float64)
            names = json.loads(row[1])
            return features, names
        else:
            matching = [
                (k, v)
                for k, v in self._in_memory.items()
                if k.startswith(f"{equipment_id}:")
            ]
            if not matching:
                return None
            matching.sort(key=lambda kv: kv[0], reverse=True)
            entry = matching[0][1]
            return entry["features"].copy(), entry["feature_names"].copy()

    def close(self) -> None:
        """Close the database connection if using SQLite backend."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("FeatureStore connection closed")

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
