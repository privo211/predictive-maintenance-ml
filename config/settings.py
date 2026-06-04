"""Application settings loaded from environment variables.

Uses pydantic-settings for type-safe configuration management. All settings
can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Type-safe application configuration.

    All values are sourced from environment variables with the prefix
    PMP_ (Predictive Maintenance Platform).  A .env file in the project
    root is automatically loaded if present.
    """

    model_config = SettingsConfigDict(
        env_prefix="PMP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root: Path = Path(__file__).resolve().parent.parent
    feature_pipeline_path: Path = Path("models/feature_pipeline.pkl")
    model_artifact_path: Path = Path("models/xgb_model.json")
    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = (
        "postgresql+asyncpg://pmp:pmp@localhost:5432/predictive_maintenance"
    )
    database_url_sync: str = (
        "postgresql+psycopg2://pmp:pmp@localhost:5432/predictive_maintenance"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # ------------------------------------------------------------------
    # Redis / Cache
    # ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "predictive-maintenance"
    model_stage: str = "Production"
    model_registry_name: str = "failure_predictor"

    # ------------------------------------------------------------------
    # API Server
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False
    api_timeout_keep_alive: int = 30
    api_max_concurrent_connections: int = 1000

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # ------------------------------------------------------------------
    # Monitoring & Drift
    # ------------------------------------------------------------------
    drift_check_interval_hours: int = 24
    drift_min_reference_samples: int = 1000
    drift_min_current_samples: int = 500
    drift_pvalue_threshold: float = 0.05

    # ------------------------------------------------------------------
    # Data Quality Gates
    # ------------------------------------------------------------------
    data_quality_min_completeness: float = 0.98
    data_quality_max_duplicate_rate: float = 0.001
    data_quality_max_outlier_zscore: float = 4.0

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    alert_webhook_url: str = ""
    alert_email_smtp_host: str = ""
    alert_email_from: str = "alerts@predictive-maintenance.local"
    alert_cooldown_minutes: int = 15

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    jwt_secret_key: str = "change-me-in-production-use-opaque-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    feature_window_minutes: int = 60
    feature_rolling_window_sizes: list[int] = [5, 15, 30, 60]
    feature_lag_periods: list[int] = [1, 3, 6, 12]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    inference_batch_size: int = 256
    inference_cache_ttl_seconds: int = 300
    prediction_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Data Generation (dev / CI only)
    # ------------------------------------------------------------------
    synthetic_equipment_count: int = 50
    synthetic_days_of_history: int = 90
    synthetic_failure_rate: float = 0.02
    synthetic_seed: int = 42


# Singleton instance — import this everywhere
settings = Settings()
