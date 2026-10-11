"""Leakage-safe training, tuning, evaluation and persistence of all benchmark models."""
from __future__ import annotations
import json
import platform
import time
import warnings
from datetime import datetime
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from . import config, data, db
# SVC(probability=True) is deprecated from scikit-learn 1.9; it is still the right choice on the tested versions.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*", category=FutureWarning)

from .schema import BINARY, CATEGORICAL, FEATURE_NAMES, NUMERIC, ORDINAL, TARGET

ALGORITHMS: dict[str, dict[str, Any]] = {
    "baseline": {"label": "Baseline (class prior)", "why": "Always predicts the majority class; any useful model must beat it."},
    "logistic_regression": {"label": "Logistic Regression", "why": "Transparent linear reference; strong on small tabular data."},
    "random_forest": {"label": "Random Forest", "why": "Captures non-linear interactions and is robust to scaling and outliers."},
    "svm_rbf": {"label": "Support Vector Machine (RBF)", "why": "Flexible smooth boundary; works well with few samples."},
    "gradient_boosting": {"label": "Gradient Boosting", "why": "Sequential tree ensemble; often the strongest tabular learner but prone to overfitting small data."},
}


def make_preprocessor() -> ColumnTransformer:
    """Imputation, scaling and encoding are fitted inside the pipeline, so they only ever see training folds."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("sc", StandardScaler())]), ORDINAL),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
    ])


def _estimators(seed: int) -> dict[str, tuple[Any, dict[str, list]]]:
    return {
        "baseline": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (LogisticRegression(max_iter=5000), {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
                          {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3]}),
        "svm_rbf": (SVC(probability=True, random_state=seed), {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.02]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=seed),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3]}),
    }


def _downsample(xs: np.ndarray, ys: np.ndarray, n: int = 80) -> list[dict[str, float]]:
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [{"x": float(xs[i]), "y": float(ys[i])} for i in idx]


def evaluate(y: np.ndarray, p1: np.ndarray, thr: float = config.DECISION_THRESHOLD) -> dict[str, Any]:
    """All held-out metrics for one model. Positive class is target=1."""
    pred = (p1 >= thr).astype(int)
    out: dict[str, Any] = {
        "n": int(len(y)), "threshold": thr,
        "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)),
        "brier": float(brier_score_loss(y, p1)), "log_loss": float(log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    two = len(np.unique(y)) == 2
    out["roc_auc"] = float(roc_auc_score(y, p1)) if two else None
    out["average_precision"] = float(average_precision_score(y, p1)) if two else None
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    out["per_class"] = {
        "0": {"precision": float(precision_score(y, pred, pos_label=0, zero_division=0)),
              "recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
              "f1": float(f1_score(y, pred, pos_label=0, zero_division=0)), "support": int((y == 0).sum())},
        "1": {"precision": out["precision"], "recall": out["recall"], "f1": out["f1"], "support": int((y == 1).sum())},
    }
    if two:
        fpr, tpr, _ = roc_curve(y, p1)
        out["roc_curve"] = _downsample(fpr, tpr)
        pr, rc, _ = precision_recall_curve(y, p1)
        out["pr_curve"] = _downsample(rc[::-1], pr[::-1])
        try:
            frac, mean_p = calibration_curve(y, p1, n_bins=5, strategy="quantile")
            out["calibration"] = [{"predicted": float(a), "observed": float(b)} for a, b in zip(mean_p, frac)]
        except ValueError:
            out["calibration"] = []
    return out


def bootstrap_ci(y: np.ndarray, p1: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    accs, aucs, f1s = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        pr = (p1[i] >= .5).astype(int)
        accs.append(accuracy_score(y[i], pr)); aucs.append(roc_auc_score(y[i], p1[i])); f1s.append(f1_score(y[i], pr, zero_division=0))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"method": "percentile bootstrap over test rows", "n_boot": len(accs),
            "accuracy": q(accs), "roc_auc": q(aucs), "f1": q(f1s)}


def _cv_summary(res: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc",
             "average_precision": "average_precision", "balanced_accuracy": "balanced_accuracy"}
    out = {k: {"mean": float(res[f"test_{k}"].mean()), "std": float(res[f"test_{k}"].std())} for k in names}
    out["brier"] = {"mean": float(-res["test_neg_brier_score"].mean()), "std": float(res["test_neg_brier_score"].std())}
    return out


def leakage_check(seed: int) -> dict[str, Any]:
    """Quantify how much a naive split on the raw (duplicated) rows inflates results."""
    raw, _ = data.load_raw()
    X, y = raw[FEATURE_NAMES], raw[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
    tr_keys = set(map(tuple, Xtr.values.tolist()))
    overlap = float(np.mean([tuple(r) in tr_keys for r in Xte.values.tolist()]))
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1))])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    return {"model": "random_forest (default)", "rows": int(len(raw)), "test_rows_also_in_train": overlap,
            "naive_split_accuracy": float(accuracy_score(yte, p >= .5)), "naive_split_roc_auc": float(roc_auc_score(yte, p)),
            "note": "Exact copies of test rows appear in the training set, so this estimate is optimistic and is NOT used anywhere else."}


def train_all(progress: Callable[[str, float], None] | None = None, seed: int = config.RANDOM_STATE,
              record: bool = True) -> dict[str, Any]:
    t0 = time.time()
    say = progress or (lambda msg, frac: None)
    say("Loading and validating dataset", 0.02)
    df, rep = data.load_unique()
    X, y = df[FEATURE_NAMES], df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    cv_inner = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed)
    cv_outer = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=seed)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc",
               "average_precision": "average_precision", "balanced_accuracy": "balanced_accuracy", "neg_brier_score": "neg_brier_score"}

    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    vdir = config.MODELS_DIR / version
    vdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    ests = _estimators(seed)
    for i, (name, (est, grid)) in enumerate(ests.items()):
        say(f"Training {ALGORITHMS[name]['label']}", 0.05 + 0.75 * i / len(ests))
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", est)])
        t1 = time.time()
        best_params: dict[str, Any] = {}
        grid_auc = None
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv_inner, n_jobs=1, refit=True)
            gs.fit(Xtr, ytr)
            pipe, grid_auc = gs.best_estimator_, float(gs.best_score_)
            best_params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
        cv_res = cross_validate(pipe, Xtr, ytr, cv=cv_outer, scoring=scoring, n_jobs=1)
        pipe.fit(Xtr, ytr)  # final fit on the full training split only
        p_te = pipe.predict_proba(Xte)[:, 1]
        ev = evaluate(yte, p_te)
        pi = permutation_importance(pipe, Xte, yte, scoring="roc_auc" if name != "baseline" else "accuracy",
                                    n_repeats=20, random_state=seed, n_jobs=1)
        joblib.dump(pipe, vdir / f"{name}.joblib")
        results[name] = {
            "label": ALGORITHMS[name]["label"], "why": ALGORITHMS[name]["why"], "params": best_params,
            "tuning": {"method": f"GridSearchCV, {config.CV_FOLDS}-fold stratified, scoring=roc_auc, training split only",
                       "grid": {k.replace("clf__", ""): v for k, v in grid.items()}, "best_cv_roc_auc": grid_auc},
            "cv": {"scheme": f"{config.CV_FOLDS}-fold x {config.CV_REPEATS} repeats, stratified, training split only",
                   "metrics": _cv_summary(cv_res)},
            "test": ev, "test_ci": bootstrap_ci(yte, p_te) if name != "baseline" else None,
            "permutation_importance": sorted(
                [{"feature": f, "mean": float(m), "std": float(s)} for f, m, s in
                 zip(FEATURE_NAMES, pi.importances_mean, pi.importances_std)], key=lambda r: -r["mean"]),
            "fit_seconds": round(time.time() - t1, 1),
        }

    say("Selecting model on training-only cross-validation", 0.85)
    cands = [k for k in results if k != "baseline"]
    selected = max(cands, key=lambda k: (round(results[k]["cv"]["metrics"]["roc_auc"]["mean"], 4),
                                         results[k]["cv"]["metrics"]["f1"]["mean"]))
    say("Running leakage check", 0.9)
    leak = leakage_check(seed)

    bg = Xtr.sample(n=min(40, len(Xtr)), random_state=seed)
    bg.to_csv(vdir / "background.csv", index=False)
    Xtr.assign(**{TARGET: ytr}).to_csv(vdir / "train.csv", index=False)
    Xte.assign(**{TARGET: yte}).to_csv(vdir / "test.csv", index=False)

    meta = {
        "version": version, "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_seconds": round(time.time() - t0, 1), "seed": seed,
        "selected": selected,
        "selection_rule": "Highest mean ROC-AUC in repeated stratified cross-validation on the training split (ties broken by F1). "
                          "The test split was not used for selection.",
        "dataset": {"sha256": rep["sha256"], "rows_in_file": rep["rows_in_file"], "unique_rows": int(len(df)),
                    "duplicates_removed": rep["duplicates_removed"], "source": config.DATASET_SOURCE},
        "split": {"test_size": config.TEST_SIZE, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                  "train_class_counts": {"0": int((ytr == 0).sum()), "1": int((ytr == 1).sum())},
                  "test_class_counts": {"0": int((yte == 0).sum()), "1": int((yte == 1).sum())},
                  "stratified": True, "deduplicated_before_split": True},
        "models": results, "leakage_check": leak, "observed_ranges": data.observed_ranges(df),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    (vdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (config.MODELS_DIR / "current.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    if record:
        db.init_db()
        db.add_experiment("training", f"Benchmark {version}: {ALGORITHMS[selected]['label']} selected", {
            "selected": selected,
            "models": {k: {"cv_roc_auc": v["cv"]["metrics"]["roc_auc"]["mean"], "test_accuracy": v["test"]["accuracy"],
                           "test_roc_auc": v["test"]["roc_auc"], "test_f1": v["test"]["f1"]} for k, v in results.items()},
            "n_train": len(Xtr), "n_test": len(Xte), "seconds": meta["train_seconds"]},
            model_version=version, config_={"seed": seed, "cv": f"{config.CV_FOLDS}x{config.CV_REPEATS}"})
    say("Done", 1.0)
    return meta


if __name__ == "__main__":  # python -m cardiosage.training
    m = train_all(progress=lambda msg, f: print(f"[{f:4.0%}] {msg}", flush=True))
    print(f"\nSaved {m['version']} in {m['train_seconds']}s; selected = {m['selected']}")
    for k, v in m["models"].items():
        t = v["test"]
        print(f"{k:20s} cvAUC={v['cv']['metrics']['roc_auc']['mean']:.3f}  test acc={t['accuracy']:.3f} auc={t['roc_auc']:.3f} f1={t['f1']:.3f}")
