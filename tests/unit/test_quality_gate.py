import numpy as np
import pandas as pd
import pytest


class TestSchemaGate:
    def test_passes_on_valid_data(self, quality_gate, sample_sensor_df):
        report = quality_gate.validate(sample_sensor_df)
        assert report.stage_results["SchemaGate"].passed
        assert report.stage_results["SchemaGate"].violations == 0

    def test_fails_on_missing_column(self, quality_gate, sample_sensor_df):
        df = sample_sensor_df.drop(columns=["value"])
        report = quality_gate.validate(df)
        assert not report.stage_results["SchemaGate"].passed

    def test_fails_on_extra_column(self, quality_gate, sample_sensor_df):
        df = sample_sensor_df.copy()
        df["bogus"] = 0.0
        report = quality_gate.validate(df)
        assert not report.stage_results["SchemaGate"].passed


class TestRangeGate:
    def test_detects_out_of_range_values(self, quality_gate):
        records = [
            {
                "timestamp": pd.Timestamp("2026-01-01"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 999.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-01"),
                "equipment_id": "EQ-0001",
                "sensor_name": "temperature_a",
                "value": -50.0,
            },
        ]
        df = pd.DataFrame(records)
        report = quality_gate.validate(df)
        assert not report.stage_results["RangeGate"].passed
        assert report.stage_results["RangeGate"].violations >= 2

    def test_passes_within_normal_range(self, quality_gate):
        records = [
            {
                "timestamp": pd.Timestamp("2026-01-01"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 4.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-01"),
                "equipment_id": "EQ-0001",
                "sensor_name": "temperature_a",
                "value": 60.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-01"),
                "equipment_id": "EQ-0001",
                "sensor_name": "rpm",
                "value": 1800.0,
            },
        ]
        df = pd.DataFrame(records)
        report = quality_gate.validate(df)
        assert report.stage_results["RangeGate"].passed


class TestTimestampGate:
    def test_detects_non_monotonic(self, quality_gate):
        records = [
            {
                "timestamp": pd.Timestamp("2026-01-01 10:00:00"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 4.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 09:00:00"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 4.0,
            },
        ]
        df = pd.DataFrame(records)
        report = quality_gate.validate(df)
        assert not report.stage_results["TimestampGate"].passed

    def test_detects_duplicates(self, quality_gate):
        ts = pd.Timestamp("2026-01-01 10:00:00")
        records = [
            {
                "timestamp": ts,
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 4.0,
            },
            {
                "timestamp": ts,
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 4.0,
            },
        ]
        df = pd.DataFrame(records)
        report = quality_gate.validate(df)
        assert not report.stage_results["TimestampGate"].passed


class TestCrossChannelGate:
    def test_detects_mismatch(self, quality_gate):
        records = [
            {
                "timestamp": pd.Timestamp("2026-01-01 10:00:00"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 4.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 10:00:00"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_b",
                "value": 9.0,
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 10:01:00"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_a",
                "value": 4.1,
            },
            {
                "timestamp": pd.Timestamp("2026-01-01 10:01:00"),
                "equipment_id": "EQ-0001",
                "sensor_name": "vibration_x_b",
                "value": 9.1,
            },
        ]
        df = pd.DataFrame(records)
        report = quality_gate.validate(df)
        assert report.stage_results["CrossChannelGate"].violations > 0


class TestDriftGate:
    def test_detects_gradual_shift(self, quality_gate):
        records = []
        for i in range(20):
            ts = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i)
            records.append(
                {
                    "timestamp": ts,
                    "equipment_id": "EQ-0001",
                    "sensor_name": "temperature_a",
                    "value": 75.0,
                }
            )
        df = pd.DataFrame(records)
        report = quality_gate.validate(df)
        assert report.stage_results["DriftGate"].violations > 0


class TestQualityReport:
    def test_report_structure(self, quality_gate, sample_sensor_df):
        report = quality_gate.validate(sample_sensor_df)
        assert isinstance(report.passed, bool)
        assert isinstance(report.stage_results, dict)
        assert isinstance(report.total_violations, int)
        assert isinstance(report.rejected_batch, bool)
        assert isinstance(report.summary, str)
        assert isinstance(report.timestamp, pd.Timestamp)
        for stage in ["SchemaGate", "RangeGate", "TimestampGate", "CrossChannelGate", "DriftGate"]:
            assert stage in report.stage_results
