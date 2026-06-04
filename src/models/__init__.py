"""Model training, evaluation, and registry modules.

Exports:
    ModelTrainer: End-to-end XGBoost training with MLflow tracking and SHAP.
    ModelEvaluator: Comprehensive metrics, promotion gates, and diagnostics.
    ModelRegistry: MLflow model registry wrapper for production workflows.
"""

from models.train import ModelTrainer
from models.evaluate import ModelEvaluator
from models.registry import ModelRegistry

__all__ = [
    "ModelTrainer",
    "ModelEvaluator",
    "ModelRegistry",
]
