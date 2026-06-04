import numpy as np
import pandas as pd

from data.synthetic_generator import SyntheticDataGenerator
from features.feature_pipeline import FeaturePipeline


class TestFullPipeline:
    def test_full_pipeline_data_to_prediction(self, trained_model, fast_train_data):
        X, y, feature_names = fast_train_data
        proba = trained_model.predict_proba(X[:50])
        failure_probs = proba[:, 1]
        assert len(failure_probs) == 50
        assert np.all((failure_probs >= 0.0) & (failure_probs <= 1.0))

    def test_pipeline_rejects_corrupted_data(self, quality_gate):
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_normal_only(num_units=2, simulation_hours=5, sample_rate_hz=1.0 / 60.0)
        df_corrupt = df.copy()
        df_corrupt.loc[0, "value"] = 999999.0

        from data.quality_gate import DataQualityGate

        report = quality_gate.validate(df_corrupt)
        has_violations = not report.stage_results["SchemaGate"].passed or report.total_violations > 0
        assert has_violations

    def test_pipeline_end_to_end_demo_mode(self, quality_gate):
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_normal_only(num_units=1, simulation_hours=3, sample_rate_hz=1.0 / 3600.0)

        report = quality_gate.validate(df)
        assert report.stage_results["SchemaGate"].passed

        pipeline = FeaturePipeline(window_sizes=[5, 10])
        pipeline.fit(df)
        X = pipeline.transform(df)
        assert X.shape[0] > 0
        predictions = np.clip(X[:, 0], 0, 1)
        assert np.all((predictions >= 0) & (predictions <= 1))


class TestFeatureStore:
    def test_feature_store_caching(self, feature_pipeline, sample_sensor_df):
        from features.feature_store import FeatureStore

        store = FeatureStore()
        X = feature_pipeline.transform(sample_sensor_df)
        names = feature_pipeline.get_feature_names()
        eq_id = sample_sensor_df["equipment_id"].iloc[0]
        ts = pd.Timestamp("2026-01-01")

        store.store(eq_id, ts, X[0], names)

        start = pd.Timestamp("2026-01-01")
        end = pd.Timestamp("2026-01-02")
        retrieved = store.retrieve(eq_id, start, end)
        assert len(retrieved) == 1
        assert list(retrieved.columns) == names

        latest = store.get_latest(eq_id)
        assert latest is not None
        features_arr, ret_names = latest
        assert len(features_arr) == len(names)
        assert ret_names == names

        store.close()
