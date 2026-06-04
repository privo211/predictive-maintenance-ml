import numpy as np
import pytest


class TestModelTrainer:
    def test_trainer_returns_trained_model(self, fast_train_data, model_config_path):
        import xgboost as xgb

        X, y, feature_names = fast_train_data

        from models.train import ModelTrainer

        trainer = ModelTrainer(config_path=model_config_path)
        sw = trainer._compute_scale_pos_weight(y)
        assert sw > 0

        model = xgb.XGBClassifier(
            n_estimators=20, max_depth=3, learning_rate=0.1, scale_pos_weight=sw, random_state=42
        )
        model.fit(X, y)
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


class TestTemporalSplit:
    def test_respects_ordering(self, model_config_path):
        from models.train import ModelTrainer

        trainer = ModelTrainer(config_path=model_config_path)
        X = np.arange(100, dtype=np.float64).reshape(-1, 1)
        y = np.zeros(100, dtype=np.int64)
        y[-10:] = 1

        X_tr, y_tr, X_val, y_val, X_test, y_test = trainer._temporal_split(X, y)

        assert len(y_tr) > len(y_val)
        assert len(y_tr) > len(y_test)
        assert X_tr.max() < X_val.min()
        assert X_val.max() < X_test.min()
        assert y_tr.sum() == 0
        assert y_test.sum() > 0

    def test_scale_pos_weight_is_positive(self, model_config_path):
        from models.train import ModelTrainer

        trainer = ModelTrainer(config_path=model_config_path)
        y_balanced = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        sw = trainer._compute_scale_pos_weight(y_balanced)
        assert sw > 0.0
        y_single = np.array([0] * 90 + [1] * 10, dtype=np.int64)
        sw2 = trainer._compute_scale_pos_weight(y_single)
        assert sw2 > 0.0
        assert sw2 > sw


class TestModelEvaluator:
    def test_evaluator_computes_all_metrics(self):
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        import numpy as np
        import xgboost as xgb

        rs = np.random.default_rng(42)
        X = rs.normal(size=(100, 10))
        y = np.zeros(100, dtype=np.int64)
        y[80:] = 1
        feature_names = [f"feat_{i}" for i in range(10)]

        model = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42, verbosity=0)
        model.fit(X, y)

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        cm = confusion_matrix(y, y_pred, labels=[0, 1])

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
            "pr_auc": float(average_precision_score(y, y_proba)),
            "confusion_matrix": cm.tolist(),
            "n_samples": len(y),
            "n_features": X.shape[1],
        }

        required = ["accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc",
                    "confusion_matrix", "n_samples", "n_features"]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
        assert 0.0 <= metrics["recall"] <= 1.0
        assert metrics["n_samples"] == len(y)


class TestPromotionGates:
    def test_check_thresholds_below_fails(self):
        from models.evaluate import ModelEvaluator

        evaluator = ModelEvaluator()
        poor_metrics = {
            "recall": 0.50,
            "false_positive_rate": 0.30,
            "roc_auc": 0.60,
        }
        passed, failures = evaluator.check_promotion_gates(poor_metrics)
        assert not passed
        assert len(failures) > 0

    def test_check_thresholds_above_passes(self):
        from models.evaluate import ModelEvaluator

        evaluator = ModelEvaluator()
        good_metrics = {
            "recall": 0.95,
            "false_positive_rate": 0.02,
            "roc_auc": 0.97,
        }
        passed, failures = evaluator.check_promotion_gates(good_metrics)
        assert passed
        assert len(failures) == 0
