"""Side-by-side scenario comparison. Everything is computed by the fitted model; nothing is a causal claim."""
from __future__ import annotations
from typing import Any

import numpy as np

from . import config
from .explain import explain_instance
from .schema import FEATURE_BY_NAME, FEATURE_NAMES
from .store import Bundle, predict_one, predict_proba1


def _sweep(bundle: Bundle, base: dict, name: str, algo: str) -> list[dict[str, float]]:
    f = FEATURE_BY_NAME[name]
    obs = bundle.meta["observed_ranges"][name]
    if f["kind"] == "numeric":
        step = f["step"] or 1
        vals = sorted({round(round(v / step) * step, 2) for v in np.linspace(obs["min"], obs["max"], 40)})
    else:
        vals = [float(v) for v in obs["values"]]
    rows = [{**base, name: v} for v in vals]
    p = predict_proba1(bundle, rows, algo)
    return [{"value": float(v), "probability_class1": float(q)} for v, q in zip(vals, p)]


def compare(bundle: Bundle, a: dict[str, Any], b: dict[str, Any], algo: str | None = None) -> dict[str, Any]:
    algo = algo or bundle.selected
    changed = [n for n in FEATURE_NAMES if float(a[n]) != float(b[n])]
    pa, pb = predict_one(bundle, a, algo), predict_one(bundle, b, algo)
    ea, eb = explain_instance(bundle, a, algo), explain_instance(bundle, b, algo)
    ca = {c["feature"]: c["contribution"] for c in ea["contributions"]}
    cb = {c["feature"]: c["contribution"] for c in eb["contributions"]}
    rows = []
    for n in FEATURE_NAMES:
        rows.append({"feature": n, "label": FEATURE_BY_NAME[n]["label"], "changed": n in changed,
                     "value_a": float(a[n]), "value_b": float(b[n]),
                     "contribution_a": ca[n], "contribution_b": cb[n], "contribution_diff": cb[n] - ca[n]})
    rows.sort(key=lambda r: -abs(r["contribution_diff"]))
    return {
        "algorithm": algo, "changed_features": changed,
        "changes": [{"feature": n, "label": FEATURE_BY_NAME[n]["label"], "unit": FEATURE_BY_NAME[n]["unit"],
                     "from": float(a[n]), "to": float(b[n]), "delta": round(float(b[n]) - float(a[n]), 3),
                     "from_display": (FEATURE_BY_NAME[n]["options"] or {}).get(int(a[n])),
                     "to_display": (FEATURE_BY_NAME[n]["options"] or {}).get(int(b[n]))} for n in changed],
        "a": {**pa, "base_value": ea["base_value"], "explanation_method": ea["method"]},
        "b": {**pb, "base_value": eb["base_value"], "explanation_method": eb["method"]},
        "probability_difference": pb["probability_class1"] - pa["probability_class1"],
        "class_flipped": pa["predicted_class"] != pb["predicted_class"],
        "feature_comparison": rows,
        "sweeps": {n: _sweep(bundle, a, n, algo) for n in changed[:4]},
        "interpretation": "Differences describe how the fitted model responds to edited inputs. They are not predictions of what would "
                          "happen if a real patient's characteristic changed, and several inputs are correlated in real patients.",
    }
