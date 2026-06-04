import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_REGISTRY_PATH = str(
    Path(__file__).resolve().parents[1] / "config" / "sensor_registry.yaml"
)
_MODEL_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
)


@pytest.fixture(scope="session")
def sensor_registry_path():
    return _REGISTRY_PATH


@pytest.fixture(scope="session")
def model_config_path():
    return _MODEL_CONFIG_PATH


@pytest.fixture(scope="session")
def _session_data():
    from data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_normal_only(num_units=1, simulation_hours=0.5, sample_rate_hz=1.0 / 60.0)


@pytest.fixture(scope="session")
def _session_fault_data():
    from data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_fleet(
        num_units=3, simulation_hours=2, sample_rate_hz=1.0 / 120.0, failure_fraction=0.4
    )


@pytest.fixture(scope="session")
def _session_model_data():
    from data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_fleet(
        num_units=3, simulation_hours=2, sample_rate_hz=1.0 / 120.0, failure_fraction=0.3
    )


@pytest.fixture
def sample_sensor_df(_session_data):
    return _session_data.copy()


@pytest.fixture
def sample_sensor_df_with_faults(_session_fault_data):
    return _session_fault_data.copy()


@pytest.fixture
def quality_gate(sensor_registry_path):
    from data.quality_gate import DataQualityGate

    return DataQualityGate(sensor_registry_path=sensor_registry_path)


@pytest.fixture(scope="session")
def _session_pipeline(_session_data):
    from features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_sizes=[5, 10])
    pipeline.fit(_session_data)
    return pipeline


@pytest.fixture
def feature_pipeline(_session_pipeline):
    return _session_pipeline


@pytest.fixture(scope="session")
def _session_model_features(_session_model_data):
    from features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_sizes=[5, 10])
    pipeline.fit(_session_model_data)
    X = pipeline.transform(_session_model_data)

    raw_health = _session_model_data.groupby(
        ["equipment_id", "timestamp"]
    )["health"].first().reset_index()
    raw_health = raw_health.sort_values(["equipment_id", "timestamp"])
    y_raw = (raw_health["health"] < 0.2).astype(int).values
    if len(y_raw) < len(X):
        y_raw = np.tile(y_raw, len(X) // len(y_raw) + 1)
    y = np.array(y_raw[: len(X)], dtype=np.int64)
    return X.astype(np.float64), y, pipeline


@pytest.fixture
def fast_train_data(_session_model_features):
    X, y, pipeline = _session_model_features
    return X, y, pipeline.get_feature_names()


@pytest.fixture(scope="session")
def _session_model_pipeline(_session_model_features):
    _, _, pipeline = _session_model_features
    return pipeline


@pytest.fixture
def trained_model(fast_train_data):
    import xgboost as xgb

    X, y, _ = fast_train_data
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    sw = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=sw,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


@pytest.fixture
def test_client(_session_model_pipeline, trained_model):
    import os

    os.environ.setdefault("PMP_LOG_LEVEL", "ERROR")

    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        app.state.model = trained_model
        app.state.feature_pipeline = _session_model_pipeline
        app.state.demo_mode = False
        app.state.model_version = "test-1.0"
        try:
            from explainability.shap_explainer import SHAPExplainer
            app.state.shap_explainer = SHAPExplainer(trained_model)
        except Exception:
            app.state.shap_explainer = None
        yield client
