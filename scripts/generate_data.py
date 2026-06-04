#!/usr/bin/env python3
"""
Generate synthetic sensor data for predictive maintenance model training.

Usage:
    python scripts/generate_data.py [--output-dir data] [--equipment-count 50]
                                     [--days 90] [--failure-rate 0.02] [--seed 42]

Output:
    CSV file written to ``<output-dir>/sensor_data.csv`` with columns:
    timestamp, equipment_id, equipment_type, sensor_name, value, health,
    fault_label.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic predictive-maintenance sensor data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to write the generated CSV",
    )
    parser.add_argument(
        "--equipment-count",
        type=int,
        default=50,
        help="Number of equipment units to simulate",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of history to generate",
    )
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=0.02,
        help="Fraction of units that experience failure",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    from data.synthetic_generator import SyntheticDataGenerator

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating %d equipment units, %d days, %.1f%% failure rate (seed=%d)",
        args.equipment_count,
        args.days,
        args.failure_rate * 100,
        args.seed,
    )

    generator = SyntheticDataGenerator(seed=args.seed)
    df = generator.generate_fleet(
        num_units=args.equipment_count,
        simulation_hours=args.days * 24.0,
        sample_rate_hz=1.0 / 60.0,
        failure_fraction=args.failure_rate,
    )

    output_path = output_dir / "sensor_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(
        "Wrote %d records (%d equipment units) to %s",
        len(df),
        df["equipment_id"].nunique(),
        output_path,
    )


if __name__ == "__main__":
    main()
