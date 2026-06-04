-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- Hypertables
-- =============================================================================

-- Raw sensor data hypertable
CREATE TABLE IF NOT EXISTS raw_sensor_data (
    time            TIMESTAMPTZ NOT NULL,
    equipment_id    TEXT NOT NULL,
    sensor_name     TEXT NOT NULL,
    value           DOUBLE PRECISION,
    health          DOUBLE PRECISION,
    fault_label     INTEGER
);

SELECT create_hypertable('raw_sensor_data', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- Processed features hypertable
CREATE TABLE IF NOT EXISTS processed_features (
    time            TIMESTAMPTZ NOT NULL,
    equipment_id    TEXT NOT NULL,
    feature_vector  DOUBLE PRECISION[],
    feature_names   TEXT[]
);

SELECT create_hypertable('processed_features', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- Predictions hypertable
CREATE TABLE IF NOT EXISTS predictions (
    time                    TIMESTAMPTZ NOT NULL,
    equipment_id            TEXT NOT NULL,
    model_version           TEXT,
    failure_probability     DOUBLE PRECISION,
    predicted_class         INTEGER,
    failure_horizon_hours   DOUBLE PRECISION,
    inference_latency_ms    DOUBLE PRECISION,
    ground_truth            INTEGER,
    shap_values             DOUBLE PRECISION[]
);

SELECT create_hypertable('predictions', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- =============================================================================
-- Relational tables
-- =============================================================================

-- Model version registry
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL,
    version         TEXT NOT NULL,
    stage           TEXT NOT NULL DEFAULT 'Staging',
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    metrics         JSONB
);

-- Alert log
CREATE TABLE IF NOT EXISTS alert_log (
    time            TIMESTAMPTZ NOT NULL DEFAULT now(),
    equipment_id    TEXT NOT NULL,
    alert_type      TEXT NOT NULL,
    severity        TEXT NOT NULL,
    message         TEXT NOT NULL,
    acknowledged    BOOLEAN NOT NULL DEFAULT FALSE
);

-- Drift events log
CREATE TABLE IF NOT EXISTS drift_events (
    time            TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT NOT NULL,
    drift_type      TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    value           DOUBLE PRECISION NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- raw_sensor_data indexes
CREATE INDEX IF NOT EXISTS idx_raw_sensor_equipment
    ON raw_sensor_data (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_raw_sensor_name
    ON raw_sensor_data (sensor_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_raw_sensor_equip_time
    ON raw_sensor_data (equipment_id, sensor_name, time DESC);

-- processed_features indexes
CREATE INDEX IF NOT EXISTS idx_processed_features_equipment
    ON processed_features (equipment_id, time DESC);

-- predictions indexes
CREATE INDEX IF NOT EXISTS idx_predictions_equipment
    ON predictions (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_ground_truth
    ON predictions (equipment_id, ground_truth)
    WHERE ground_truth IS NOT NULL;

-- alert_log indexes
CREATE INDEX IF NOT EXISTS idx_alert_log_equipment
    ON alert_log (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_alert_log_unacknowledged
    ON alert_log (acknowledged, time DESC)
    WHERE acknowledged = FALSE;

-- drift_events indexes
CREATE INDEX IF NOT EXISTS idx_drift_events_version
    ON drift_events (model_version, time DESC);

-- =============================================================================
-- Compression policy
-- =============================================================================

ALTER TABLE raw_sensor_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('raw_sensor_data', INTERVAL '7 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- Continuous aggregates
-- =============================================================================

-- Hourly feature aggregates (rolling 1-hour windows)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_features
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    equipment_id,
    sensor_name,
    AVG(value)                  AS avg_value,
    MIN(value)                  AS min_value,
    MAX(value)                  AS max_value,
    STDDEV(value)               AS stddev_value,
    COUNT(*)                    AS sample_count
FROM raw_sensor_data
GROUP BY bucket, equipment_id, sensor_name
WITH NO DATA;

SELECT add_continuous_aggregate_policy('hourly_features',
    start_offset    => INTERVAL '2 hours',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- Refresh the continuous aggregate to backfill existing data
CALL refresh_continuous_aggregate('hourly_features', NULL, NULL);
