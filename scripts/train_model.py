#!/usr/bin/env python3
"""
End-to-end training script for the predictive maintenance model.

Usage:
    python scripts/train_model.py [--config config/model_config.yaml]

This script:
1. Generates synthetic training data (or loads existing)
2. Runs data through quality gates
3. Engineers features
4. Trains XGBoost classifier
5. Evaluates against promotion gates
6. Registers model in MLflow registry
7. Saves SHAP explainer
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path for package imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_model")


def _setup_project_path() -> None:
    """Add source root to Python path."""
    src_path = str(_PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _load_quality_gate():
    """Import the full DataQualityGate or return None."""
    try:
        from data.quality_gate import DataQualityGate
        return DataQualityGate
    except ImportError:
        logger.warning("DataQualityGate not available — quality checks skipped.")
        return None


def _optional_feature_pipeline():
    """Optionally import or create a minimal FeaturePipeline."""
    try:
        from features import FeaturePipeline
        return FeaturePipeline
    except ImportError:
        logger.warning(
            "FeaturePipeline not found. Creating minimal feature set from raw sensor values."
        )


class _MinimalFeaturePipeline:
    """Fallback feature pipeline when the full one is not available.

    Pivots long-form sensor data to wide format with simple rolling
    statistics per equipment unit.
    """

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._pivot_and_engineer(df)

    def get_feature_names(self) -> list[str]:
        return getattr(self, "_feature_names", [])

    def _pivot_and_engineer(self, df: pd.DataFrame) -> np.ndarray:
        wide = df.pivot_table(
            index=["timestamp", "equipment_id"],
            columns="sensor_name",
            values="value",
        ).reset_index()

        wide = wide.sort_values(["equipment_id", "timestamp"])

        sensor_cols = [
            c for c in wide.columns
            if c not in ("timestamp", "equipment_id")
        ]

        wide = wide.ffill().bfill().fillna(0)

        features: dict[str, np.ndarray] = {}

        for col in sensor_cols:
            series = wide[col].astype(float).values
            features[f"{col}_raw"] = series
            window = min(10, len(series))
            if window >= 2:
                features[f"{col}_rolling_mean_10"] = pd.Series(series).rolling(window, min_periods=1).mean().values
                features[f"{col}_rolling_std_10"] = pd.Series(series).rolling(window, min_periods=1).std().fillna(0).values

        self._feature_names = list(features.keys())
        return np.column_stack(list(features.values()))


# ------------------------------------------------------------------
# Label extraction
# ------------------------------------------------------------------


def _extract_labels(df: pd.DataFrame, feature_df: pd.DataFrame) -> np.ndarray:
    """Extract binary labels from the DataFrame.

    Uses the health threshold (< 0.2) as failure-imminent label.
    Aggregates per (timestamp, equipment_id) window — if any sensor
    reading in that window has health < 0.2, the window is labelled 1.
    """
    if "health" in df.columns:
        label_df = (
            df.groupby(["timestamp", "equipment_id"])["health"]
            .min()
            .reset_index()
        )
        label_df = label_df.sort_values(["equipment_id", "timestamp"])
        return (label_df["health"] < 0.2).astype(int).values

    # Fallback: use fault_label if available
    if "fault_label" in df.columns:
        label_wide = df.pivot_table(
            index=["timestamp", "equipment_id"],
            values="fault_label",
            aggfunc="max",
        ).reset_index()
        label_wide = label_wide.sort_values(["equipment_id", "timestamp"])
        return label_wide["fault_label"].astype(int).values

    logger.error("No 'health' or 'fault_label' column found in DataFrame.")
    raise ValueError("Cannot extract labels from DataFrame.")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run the full training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train predictive maintenance model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to existing CSV data (skips generation if provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for saved artifacts",
    )
    parser.add_argument(
        "--num-units",
        type=int,
        default=20,
        help="Number of equipment units to simulate",
    )
    parser.add_argument(
        "--simulation-hours",
        type=float,
        default=2000.0,
        help="Hours of simulation data",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Sampling rate in Hz",
    )
    parser.add_argument(
        "--failure-fraction",
        type=float,
        default=0.15,
        help="Fraction of units that fail",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Resolve config path relative to project root
    config_path = str(_PROJECT_ROOT / args.config)
    output_dir = str(_PROJECT_ROOT / args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  PREDICTIVE MAINTENANCE — MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate or load data
    # ------------------------------------------------------------------
    if args.data_path:
        logger.info("Loading data from %s", args.data_path)
        df = pd.read_csv(args.data_path, parse_dates=["timestamp"])
    else:
        logger.info(
            "Generating synthetic fleet: %d units, %.0f hours, %.1f Hz",
            args.num_units,
            args.simulation_hours,
            args.sample_rate,
        )
        from data.synthetic_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=args.seed)
        df = generator.generate_fleet(
            num_units=args.num_units,
            simulation_hours=args.simulation_hours,
            sample_rate_hz=args.sample_rate,
            failure_fraction=args.failure_fraction,
        )
        logger.info(
            "Generated %d records across %d equipment units.",
            len(df),
            df["equipment_id"].nunique(),
        )

    # ------------------------------------------------------------------
    # Step 2: Data quality gates
    # ------------------------------------------------------------------
    DataQualityGate = _load_quality_gate()
    if DataQualityGate is not None:
        quality_gate = DataQualityGate(
            sensor_registry_path=str(_PROJECT_ROOT / "config/sensor_registry.yaml")
        )
        report = quality_gate.validate(df)
        if not report.passed:
            logger.warning("Data quality issues: %s", report.summary)
        else:
            logger.info("Data quality checks PASSED.")
    else:
        logger.info("Data quality checks skipped (module not available).")

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("Engineering features...")

    FeaturePipelineClass = _optional_feature_pipeline()
    if FeaturePipelineClass is None:
        FeaturePipelineClass = _MinimalFeaturePipeline

    pipeline = FeaturePipelineClass()
    X = pipeline.fit_transform(df)
    y = _extract_labels(df, None)
    feature_names = pipeline.get_feature_names()

    logger.info(
        "Feature matrix shape: %s, labels: %d positive / %d total (%.2f%%)",
        X.shape,
        int(np.sum(y == 1)),
        len(y),
        100.0 * np.mean(y),
    )

    if len(feature_names) == 0:
        logger.error("No features extracted. Check feature pipeline.")
        sys.exit(1)

    if np.sum(y == 1) == 0:
        logger.error("No positive labels found. Increase failure_fraction or simulation_hours.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Train model
    # ------------------------------------------------------------------
    logger.info("Training XGBoost classifier...")

    from models.train import ModelTrainer

    trainer = ModelTrainer(config_path=config_path)
    model, explainer, train_metrics = trainer.train(
        X, y, feature_names, experiment_name="predictive_maintenance",
    )

    logger.info("Training complete. Validation metrics: %s", train_metrics)

    # Extract test split for evaluation
    # Re-do the temporal split to get test set
    n = len(y)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_test, y_test = X[val_end:], y[val_end:]

    if len(y_test) == 0:
        logger.warning("Test set is empty — using validation indices instead.")
        X_test, y_test = X[train_end:val_end], y[train_end:val_end]

    # ------------------------------------------------------------------
    # Step 5: Evaluate model
    # ------------------------------------------------------------------
    logger.info("Evaluating model...")

    from models.evaluate import ModelEvaluator

    evaluator = ModelEvaluator()
    eval_metrics = evaluator.evaluate(model, X_test, y_test, feature_names)
    passed, failures = evaluator.check_promotion_gates(eval_metrics)

    report = evaluator.generate_report(
        eval_metrics,
        y_test,
        model.predict(X_test),
        model.predict_proba(X_test)[:, 1],
    )
    logger.info("\n%s", report)

    # ------------------------------------------------------------------
    # Step 6 & 7: Save artifacts
    # ------------------------------------------------------------------
    pipeline_path = f"{output_dir}/feature_pipeline.joblib"
    explainer_path = f"{output_dir}/shap_explainer.joblib"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(explainer, explainer_path)

    logger.info("Feature pipeline saved to %s", pipeline_path)
    logger.info("SHAP explainer saved to %s", explainer_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    if passed:
        logger.info("  TRAINING COMPLETE — ALL PROMOTION GATES PASSED")
    else:
        logger.warning("  TRAINING COMPLETE — SOME GATES FAILED")
        for failure in failures:
            logger.warning("    - %s", failure)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
