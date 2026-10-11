"""Loading of persisted model versions and helper prediction functions."""
from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config
from .schema import DISCRETE, FEATURE_NAMES, TARGET


class NoModelError(RuntimeError):
    pass


@dataclass
class Bundle:
    version: str
    meta: dict[str, Any]
    pipelines: dict[str, Any]
    background: pd.DataFrame
    train_x: pd.DataFrame
    test: pd.DataFrame
    nn: NearestNeighbors = field(repr=False, default=None)
    nn_p95: float = 0.0
    _prep_cache: Any = field(repr=False, default=None)

    @property
    def selected(self) -> str:
        return self.meta["selected"]

    def pipeline(self, algo: str | None = None):
        algo = algo or self.selected
        if algo not in self.pipelines:
            raise KeyError(f"Unknown model '{algo}'. Available: {sorted(self.pipelines)}")
        return self.pipelines[algo]


_cache: dict[str, Bundle] = {}
_lock = threading.Lock()


def current_version() -> str | None:
    p = config.MODELS_DIR / "current.json"
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text())["version"]
    except Exception:
        return None
    return v if (config.MODELS_DIR / v / "metadata.json").exists() else None


def list_versions() -> list[dict[str, Any]]:
    out = []
    if not config.MODELS_DIR.exists():
        return out
    cur = current_version()
    for d in sorted(config.MODELS_DIR.iterdir(), reverse=True):
        mf = d / "metadata.json"
        if d.is_dir() and mf.exists():
            m = json.loads(mf.read_text())
            sel = m["models"][m["selected"]]
            out.append({"version": m["version"], "created_at": m["created_at"], "selected": m["selected"],
                        "selected_label": sel["label"], "cv_roc_auc": sel["cv"]["metrics"]["roc_auc"]["mean"],
                        "test_roc_auc": sel["test"]["roc_auc"], "test_accuracy": sel["test"]["accuracy"],
                        "n_train": m["split"]["n_train"], "n_test": m["split"]["n_test"], "active": m["version"] == cur})
    return out


def set_current(version: str) -> None:
    if not (config.MODELS_DIR / version / "metadata.json").exists():
        raise KeyError(f"Model version '{version}' does not exist.")
    (config.MODELS_DIR / "current.json").write_text(json.dumps({"version": version}))


def load_bundle(version: str | None = None) -> Bundle:
    version = version or current_version()
    if not version:
        raise NoModelError("No trained model found. Train one with `python -m cardiosage.training` or POST /api/train.")
    with _lock:
        if version in _cache:
            return _cache[version]
        vdir = config.MODELS_DIR / version
        if not (vdir / "metadata.json").exists():
            raise NoModelError(f"Model version {version} not found in {config.MODELS_DIR}.")
        meta = json.loads((vdir / "metadata.json").read_text())
        pipes = {p.stem: joblib.load(p) for p in vdir.glob("*.joblib")}
        bg = pd.read_csv(vdir / "background.csv")
        tr = pd.read_csv(vdir / "train.csv")
        te = pd.read_csv(vdir / "test.csv")
        b = Bundle(version, meta, pipes, bg[FEATURE_NAMES], tr[FEATURE_NAMES], te)
        # nearest-neighbour index in transformed space -> plausibility of counterfactual candidates
        prep = pipes[meta["selected"]].named_steps["prep"]
        Z = prep.transform(b.train_x)
        b.nn = NearestNeighbors(n_neighbors=2).fit(Z)
        d, _ = b.nn.kneighbors(Z)  # column 0 is the point itself
        b.nn_p95 = float(np.percentile(d[:, 1], 95))
        _cache[version] = b
        return b


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def to_frame(rows: list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Build a correctly-typed frame (discrete features as int, numerics as float)."""
    if isinstance(rows, dict):
        rows = [rows]
    df = pd.DataFrame(rows)[FEATURE_NAMES].copy()
    for c in FEATURE_NAMES:
        df[c] = np.round(df[c].astype(float)).astype(int) if c in DISCRETE or c in ("age", "trestbps", "chol", "thalach") \
            else df[c].astype(float)
    return df


def predict_proba1(bundle: Bundle, rows, algo: str | None = None) -> np.ndarray:
    """P(target=1) from the persisted pipeline."""
    return bundle.pipeline(algo).predict_proba(to_frame(rows))[:, 1]


def predict_one(bundle: Bundle, row: dict[str, Any], algo: str | None = None) -> dict[str, Any]:
    p1 = float(predict_proba1(bundle, row, algo)[0])
    cls = int(p1 >= config.DECISION_THRESHOLD)
    return {"predicted_class": cls, "probability_class1": p1, "probability_class0": 1 - p1,
            "confidence": max(p1, 1 - p1), "threshold": config.DECISION_THRESHOLD}
