import numpy as np
import pandas as pd

from data.synthetic_generator import SyntheticDataGenerator


class TestSyntheticDataQuality:
    def test_synthetic_data_has_no_nan_values(self):
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_normal_only(num_units=3, simulation_hours=10, sample_rate_hz=1.0 / 60.0)
        nan_counts = df[["timestamp", "equipment_id", "sensor_name", "value"]].isna().sum()
        for col in ["timestamp", "equipment_id", "sensor_name", "value"]:
            assert nan_counts[col] == 0, f"Column {col} has {nan_counts[col]} NaN values"

    def test_synthetic_data_ranges_are_plausible(self):
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_normal_only(num_units=3, simulation_hours=10, sample_rate_hz=1.0 / 60.0)
        sensor_groups = df.groupby("sensor_name")["value"]
        for sensor_name, group in sensor_groups:
            s_min = group.min()
            s_max = group.max()
            assert np.isfinite(s_min), f"{sensor_name} has non-finite min"
            assert np.isfinite(s_max), f"{sensor_name} has non-finite max"
            if "vibration" in sensor_name:
                assert s_max < 50.0, f"{sensor_name} max {s_max} too high for vibration"
            if "temperature" in sensor_name:
                assert s_max < 800.0, f"{sensor_name} max {s_max} too high for temperature"
            if "health" == sensor_name:
                assert 0.0 <= s_min and s_max <= 1.0, f"health out of range [{s_min:.2f}, {s_max:.2f}]"

    def test_fault_labels_are_binary(self):
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_fleet(
            num_units=4, simulation_hours=30, sample_rate_hz=1.0 / 60.0, failure_fraction=0.3
        )
        unique_labels = set(df["fault_label"].unique())
        assert unique_labels.issubset({0, 1}), f"Unexpected fault labels: {unique_labels}"
        assert df["fault_label"].dtype == np.int64

    def test_health_column_range(self):
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate_normal_only(num_units=2, simulation_hours=5, sample_rate_hz=1.0 / 60.0)
        assert df["health"].min() >= 0.0
        assert df["health"].max() <= 1.0
