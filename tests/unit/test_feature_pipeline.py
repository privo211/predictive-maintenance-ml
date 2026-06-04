import numpy as np
import pytest


class TestFeaturePipelineFit:
    def test_fit_returns_self(self, sample_sensor_df):
        from features.feature_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(window_sizes=[5, 10])
        result = pipeline.fit(sample_sensor_df)
        assert result is pipeline


class TestFeaturePipelineTransform:
    def test_transform_returns_numpy_array(self, feature_pipeline, sample_sensor_df):
        X = feature_pipeline.transform(sample_sensor_df)
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2

    def test_output_has_expected_shape(self, feature_pipeline, sample_sensor_df):
        X = feature_pipeline.transform(sample_sensor_df)
        n_features = len(feature_pipeline.get_feature_names())
        assert X.shape[1] == n_features
        assert X.shape[0] > 0


class TestFeatureNames:
    def test_names_are_unique_and_descriptive(self, feature_pipeline):
        names = feature_pipeline.get_feature_names()
        assert len(names) == len(set(names))
        assert any("mean" in n for n in names)
        assert any("std" in n for n in names)

    def test_get_feature_names_before_fit_raises(self):
        from features.feature_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(window_sizes=[5])
        with pytest.raises(AttributeError):
            pipeline.get_feature_names()


class TestEdgeCases:
    def test_handles_single_equipment(self, feature_pipeline, sample_sensor_df):
        eq_id = sample_sensor_df["equipment_id"].iloc[0]
        single = sample_sensor_df[sample_sensor_df["equipment_id"] == eq_id]
        X = feature_pipeline.transform(single)
        assert X.shape[0] > 0
        assert X.ndim == 2

    def test_handles_missing_sensors(self, feature_pipeline, sample_sensor_df):
        from features.feature_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(window_sizes=[5, 10])
        pipeline.fit(sample_sensor_df)

        sensors = sorted(sample_sensor_df["sensor_name"].unique())
        present_sensors = sensors[: max(1, len(sensors) // 2)]
        subset = sample_sensor_df[sample_sensor_df["sensor_name"].isin(present_sensors)]
        X = pipeline.transform(subset)
        assert X.shape[1] == len(pipeline.get_feature_names())


class TestFaultDataTransform:
    def test_transform_on_faulty_data(self, feature_pipeline, sample_sensor_df_with_faults):
        X = feature_pipeline.transform(sample_sensor_df_with_faults)
        finite_mask = np.isfinite(X)
        assert finite_mask.any()
