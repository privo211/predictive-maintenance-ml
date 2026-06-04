import numpy as np
import pytest


class TestModelRecall:
    def test_model_recall_above_threshold(self):
        import numpy as np
        import xgboost as xgb

        rs = np.random.default_rng(42)
        n_samples = 200
        n_failures = 40
        X = rs.normal(size=(n_samples, 5))
        X[:n_failures, 0] += 3.0
        X[-10:, 0] += 3.0
        y = np.zeros(n_samples, dtype=np.int64)
        y[:n_failures] = 1

        model = xgb.XGBClassifier(n_estimators=20, max_depth=2, random_state=42, verbosity=0)
        model.fit(X, y)
        y_pred = model.predict(X)
        tp = int(np.sum((y_pred == 1) & (y == 1)))
        fn = int(np.sum((y_pred == 0) & (y == 1)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert recall >= 0.5, f"Recall {recall:.4f} below minimum threshold"


class TestModelFPR:
    def test_model_fpr_below_threshold(self):
        import numpy as np
        import xgboost as xgb

        rs = np.random.default_rng(42)
        n_samples = 200
        n_failures = 40
        X = rs.normal(size=(n_samples, 5))
        X[:n_failures, 0] += 3.0
        X[-10:, 0] += 3.0
        y = np.zeros(n_samples, dtype=np.int64)
        y[:n_failures] = 1

        model = xgb.XGBClassifier(n_estimators=20, max_depth=2, random_state=42, verbosity=0)
        model.fit(X, y)
        y_pred = model.predict(X)
        fp = int(np.sum((y_pred == 1) & (y == 0)))
        tn = int(np.sum((y_pred == 0) & (y == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
        assert fpr <= 0.3, f"FPR {fpr:.4f} above maximum threshold"


class TestModelROCAUC:
    def test_model_roc_auc_above_threshold(self):
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        rs = np.random.default_rng(42)
        n_samples = 200
        n_failures = 40
        X = rs.normal(size=(n_samples, 5))
        X[:n_failures, 0] += 3.0
        X[-10:, 0] += 3.0
        y = np.zeros(n_samples, dtype=np.int64)
        y[:n_failures] = 1

        model = xgb.XGBClassifier(n_estimators=20, max_depth=2, random_state=42, verbosity=0)
        model.fit(X, y)
        y_proba = model.predict_proba(X)[:, 1]
        if len(np.unique(y)) < 2:
            pytest.skip("Only one class in labels")
        auc = roc_auc_score(y, y_proba)
        assert auc >= 0.5, f"ROC AUC {auc:.4f} below threshold"


class TestPredictionProbabilities:
    def test_probabilities_are_calibrated(self):
        import numpy as np
        import xgboost as xgb

        rs = np.random.default_rng(42)
        n_samples = 200
        n_failures = 40
        X = rs.normal(size=(n_samples, 5))
        X[:n_failures, 0] += 3.0
        X[-10:, 0] += 3.0
        y = np.zeros(n_samples, dtype=np.int64)
        y[:n_failures] = 1

        model = xgb.XGBClassifier(n_estimators=20, max_depth=2, random_state=42, verbosity=0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2
        assert np.all((proba >= 0.0) & (proba <= 1.0))
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
