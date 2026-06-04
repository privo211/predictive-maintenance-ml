"""Model explainability via SHAP and feature attribution.

Exports:
    SHAPExplainer: Exact TreeExplainer wrapper for XGBoost with
                   single-prediction and global importance APIs.
"""

from explainability.shap_explainer import SHAPExplainer

__all__ = ["SHAPExplainer"]
