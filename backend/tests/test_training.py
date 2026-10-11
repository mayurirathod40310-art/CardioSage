import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from cardiosage import config, store
from cardiosage.schema import FEATURE_NAMES, TARGET


def test_training_writes_artifacts(trained):
    vdir = config.MODELS_DIR / trained["version"]
    for name in ("baseline", "logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"):
        assert (vdir / f"{name}.joblib").exists()
    assert (vdir / "metadata.json").exists()
    assert json.loads((config.MODELS_DIR / "current.json").read_text())["version"] == trained["version"]


def test_split_is_stratified_deduplicated_and_disjoint(trained):
    vdir = config.MODELS_DIR / trained["version"]
    tr, te = pd.read_csv(vdir / "train.csv"), pd.read_csv(vdir / "test.csv")
    assert len(tr) + len(te) == trained["dataset"]["unique_rows"] == 302
    merged = pd.concat([tr, te])
    assert not merged.duplicated().any(), "a row appears in both splits (leakage)"
    assert abs(tr[TARGET].mean() - te[TARGET].mean()) < 0.05


def test_selection_uses_training_cv_only_and_excludes_baseline(trained):
    cands = {k: v["cv"]["metrics"]["roc_auc"]["mean"] for k, v in trained["models"].items() if k != "baseline"}
    assert trained["selected"] == max(cands, key=cands.get) or abs(cands[trained["selected"]] - max(cands.values())) < 1e-4
    assert trained["selected"] != "baseline"
    assert "test split was not used" in trained["selection_rule"]


def test_models_beat_baseline(trained):
    base = trained["models"]["baseline"]["test"]["roc_auc"]
    for k, v in trained["models"].items():
        if k != "baseline":
            assert v["test"]["roc_auc"] > base + 0.2
            assert v["cv"]["metrics"]["roc_auc"]["mean"] > 0.8


def test_saved_model_reloads_and_reproduces_reported_test_metrics(trained):
    store.clear_cache()
    vdir = config.MODELS_DIR / trained["version"]
    te = pd.read_csv(vdir / "test.csv")
    for k in ("logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"):
        pipe = joblib.load(vdir / f"{k}.joblib")
        p = pipe.predict_proba(te[FEATURE_NAMES])[:, 1]
        rep = trained["models"][k]["test"]
        assert abs(accuracy_score(te[TARGET], p >= .5) - rep["accuracy"]) < 1e-9
        assert abs(roc_auc_score(te[TARGET], p) - rep["roc_auc"]) < 1e-9


def test_confusion_matrix_adds_up(trained):
    for v in trained["models"].values():
        cm = v["test"]["confusion_matrix"]
        assert sum(cm.values()) == trained["split"]["n_test"]
        assert v["test"]["roc_curve"][0] == {"x": 0.0, "y": 0.0} and v["test"]["roc_curve"][-1]["x"] == 1.0


def test_leakage_check_shows_inflation(trained):
    lk = trained["leakage_check"]
    assert lk["test_rows_also_in_train"] > 0.5
    assert lk["naive_split_accuracy"] > trained["models"]["random_forest"]["test"]["accuracy"]


def test_preprocessing_handles_missing_values_inside_pipeline(trained):
    pipe = joblib.load(config.MODELS_DIR / trained["version"] / "random_forest.joblib")
    row = pd.DataFrame([{"age": 50, "sex": 1, "cp": 2, "trestbps": np.nan, "chol": 200, "fbs": 0, "restecg": 1, "thalach": 150,
                         "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": np.nan, "thal": 2}])
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1
