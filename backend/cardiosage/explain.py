"""Local explanations. Primary method: KernelSHAP over the original 13 features (so attributions map to what the
user typed, not to one-hot columns). Fallback: one-feature-at-a-time background substitution."""
from __future__ import annotations
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .schema import FEATURE_BY_NAME, FEATURE_NAMES
from .store import Bundle, predict_proba1, to_frame


def _fmt_value(name: str, v: float) -> str:
    f = FEATURE_BY_NAME[name]
    if f["options"]:
        return f["options"].get(int(round(v)), str(int(round(v))))
    return f"{v:g}" + (f" {f['unit']}" if f["unit"] else "")


def _occlusion(bundle: Bundle, row: dict, algo: str | None) -> tuple[float, np.ndarray]:
    """Replace each feature by every background value in turn; the mean probability change is the attribution.
    Not additive, so it is only used when SHAP fails."""
    x = to_frame(row)
    p = float(predict_proba1(bundle, x, algo)[0])
    bg = bundle.background
    base = float(predict_proba1(bundle, bg, algo).mean())
    contrib = np.zeros(len(FEATURE_NAMES))
    for j, name in enumerate(FEATURE_NAMES):
        tmp = pd.concat([x] * len(bg), ignore_index=True)
        tmp[name] = bg[name].values
        contrib[j] = p - float(predict_proba1(bundle, tmp, algo).mean())
    return base, contrib


def explain_instance(bundle: Bundle, row: dict[str, Any], algo: str | None = None, nsamples: int = 1500,
                     top: int | None = None) -> dict[str, Any]:
    algo = algo or bundle.selected
    x = to_frame(row)
    p1 = float(predict_proba1(bundle, x, algo)[0])
    method, warn = "kernel_shap", None
    try:
        import shap
        bg = bundle.background.values.astype(float)
        cols = FEATURE_NAMES

        def f(arr: np.ndarray) -> np.ndarray:
            return predict_proba1(bundle, pd.DataFrame(arr, columns=cols), algo)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = shap.KernelExplainer(f, bg)
            sv = np.asarray(ex.shap_values(x.values.astype(float), nsamples=nsamples, silent=True)).reshape(-1)
            base = float(np.asarray(ex.expected_value).reshape(-1)[0])
        if not np.all(np.isfinite(sv)):
            raise ValueError("non-finite SHAP values")
    except Exception as exc:  # explanation failure must never break prediction
        method, warn = "occlusion", f"SHAP failed ({type(exc).__name__}: {exc}); showing a simpler substitution-based explanation."
        base, sv = _occlusion(bundle, row, algo)

    items = []
    for j, name in enumerate(FEATURE_NAMES):
        items.append({"feature": name, "label": FEATURE_BY_NAME[name]["label"], "value": float(x.iloc[0][name]),
                      "value_display": _fmt_value(name, float(x.iloc[0][name])), "contribution": float(sv[j]),
                      "direction": 1 if sv[j] > 0 else (0 if sv[j] < 0 else None)})
    items.sort(key=lambda r: -abs(r["contribution"]))
    return {
        "method": method, "warning": warn, "algorithm": algo, "output": "probability of target = 1",
        "base_value": base, "prediction": p1, "sum_of_contributions": float(sv.sum()),
        "additivity_error": float(abs(base + sv.sum() - p1)) if method == "kernel_shap" else None,
        "background_rows": int(len(bundle.background)), "nsamples": nsamples if method == "kernel_shap" else None,
        "contributions": items[:top] if top else items,
        "reading_guide": "Positive contributions push the model's probability of target = 1 up relative to the background average, "
                         "negative ones push it toward target = 0. They describe model behaviour, not causes in a patient.",
    }
