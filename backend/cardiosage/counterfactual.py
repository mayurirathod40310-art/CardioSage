"""Model counterfactual search.

A counterfactual here is an alternative input vector for which THE MODEL's predicted class differs. It is a
statement about the fitted function, not a causal or medical recommendation.

Search: (1) every single-feature change, (2) all feature pairs on thinned grids, (3) triples over the most influential
features. Candidates must keep the desired class with a probability margin, stay inside observed data ranges and valid
categories. The cheapest candidates are then sparsified/shrunk and finally re-verified one by one with a fresh prediction.
"""
from __future__ import annotations
import itertools
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .schema import FEATURE_BY_NAME, FEATURE_NAMES, NUMERIC
from .store import Bundle, predict_proba1, to_frame


def _grid(bundle: Bundle, name: str, current: float, thin: int | None = None) -> list[float]:
    f = FEATURE_BY_NAME[name]
    obs = bundle.meta["observed_ranges"][name]
    if f["kind"] != "numeric":
        vals = [float(v) for v in obs["values"]]
    else:
        lo, hi, step = obs["min"], obs["max"], f["step"] or 1
        pts = list(np.linspace(lo, hi, 25)) + [obs[k] for k in ("p5", "p25", "p50", "p75", "p95")]
        vals = sorted({round(round(v / step) * step, 2) for v in pts})
        vals = [v for v in vals if lo - 1e-9 <= v <= hi + 1e-9]
    vals = [v for v in vals if abs(v - current) > 1e-9]
    if thin and len(vals) > thin:
        idx = np.unique(np.linspace(0, len(vals) - 1, thin).astype(int))
        vals = [vals[i] for i in idx]
    return vals


def _scale(bundle: Bundle, name: str) -> float:
    o = bundle.meta["observed_ranges"][name]
    return max(o["max"] - o["min"], 1e-9) if FEATURE_BY_NAME[name]["kind"] == "numeric" else 1.0


def _cost(bundle: Bundle, base: dict, cand: dict, changed: list[str]) -> float:
    c = 0.0
    for n in changed:
        c += 1.0 + (abs(cand[n] - base[n]) / _scale(bundle, n) if FEATURE_BY_NAME[n]["kind"] == "numeric" else 1.0)
    return c


def _batch(bundle: Bundle, base: dict, combos: list[dict[str, float]], algo: str, chunk: int = 20000) -> np.ndarray:
    out = []
    for i in range(0, len(combos), chunk):
        part = combos[i:i + chunk]
        rows = []
        for ch in part:
            r = dict(base); r.update(ch); rows.append(r)
        out.append(predict_proba1(bundle, rows, algo))
    return np.concatenate(out) if out else np.array([])


def _desired_prob(p1: np.ndarray, desired: int) -> np.ndarray:
    return p1 if desired == 1 else 1 - p1


def generate(bundle: Bundle, instance: dict[str, Any], algo: str | None = None, desired_class: int | None = None,
             locked: list[str] | None = None, max_changes: int = 3, n_results: int = 5, margin: float = 0.05) -> dict[str, Any]:
    algo = algo or bundle.selected
    base = {k: float(instance[k]) for k in FEATURE_NAMES}
    p_orig = float(predict_proba1(bundle, base, algo)[0])
    cur_cls = int(p_orig >= config.DECISION_THRESHOLD)
    desired = (1 - cur_cls) if desired_class is None else int(desired_class)
    locked = [l for l in (locked or []) if l in FEATURE_NAMES]
    free = [n for n in FEATURE_NAMES if n not in locked]
    max_changes = max(1, min(int(max_changes), 3))
    target_p = config.DECISION_THRESHOLD + margin
    result: dict[str, Any] = {
        "algorithm": algo, "original": {"probability_class1": p_orig, "predicted_class": cur_cls}, "desired_class": desired,
        "locked_features": locked, "free_features": free, "max_changes": max_changes, "required_probability": target_p,
        "disclaimer": "These are mathematical counterfactuals of the fitted model: input changes that flip its output. They do not show "
                      "that changing a real patient's characteristic would change their health, and they are not medical advice.",
    }
    if desired == cur_cls:
        result.update(found=False, candidates=[], evaluated=0, reason="The model already predicts the desired class for this input.")
        return result

    grids1 = {n: _grid(bundle, n, base[n]) for n in free}
    combos: list[dict[str, float]] = [{n: v} for n in free for v in grids1[n]]
    singles = len(combos)
    if max_changes >= 2:
        g2 = {n: _grid(bundle, n, base[n], thin=10) for n in free}
        for a, b in itertools.combinations(free, 2):
            combos += [{a: va, b: vb} for va in g2[a] for vb in g2[b]]
    # influence ranking from single-feature moves toward the desired class
    p_single = _desired_prob(_batch(bundle, base, combos[:singles], algo), desired)
    base_d = _desired_prob(np.array([p_orig]), desired)[0]
    infl: dict[str, float] = {}
    for ch, p in zip(combos[:singles], p_single):
        n = next(iter(ch)); infl[n] = max(infl.get(n, 0.0), float(p - base_d))
    if max_changes >= 3:
        top = sorted(free, key=lambda n: -infl.get(n, 0))[:7]
        g3 = {n: _grid(bundle, n, base[n], thin=6) for n in top}
        for tri in itertools.combinations(top, 3):
            combos += [dict(zip(tri, vals)) for vals in itertools.product(*(g3[n] for n in tri))]

    probs = _desired_prob(_batch(bundle, base, combos, algo), desired)
    result["evaluated"] = int(len(combos))
    ok_idx = np.where(probs >= target_p)[0]
    result["best_attempt_probability"] = float(probs.max()) if len(probs) else None
    result["best_attempt_changes"] = combos[int(probs.argmax())] if len(probs) else None
    if len(ok_idx) == 0:
        result.update(found=False, candidates=[], reason=(
            f"No combination of up to {max_changes} unlocked feature changes within the observed data ranges moved the model to class {desired} "
            f"with probability ≥ {target_p:.2f}. The closest attempt reached {probs.max():.2f}. Try unlocking more features, allowing more "
            "changes or lowering the margin."))
        return result

    scored = sorted(((_cost(bundle, base, combos[i], list(combos[i])), int(i)) for i in ok_idx))
    # take the cheapest candidates but keep variety: at most 3 per feature set, 25 in total
    seeds, per_set0 = [], {}
    for _, i in scored:
        k = tuple(sorted(combos[i]))
        if per_set0.get(k, 0) >= 3:
            continue
        per_set0[k] = per_set0.get(k, 0) + 1
        seeds.append(i)
        if len(seeds) >= 25:
            break

    def ok_batch(chs: list[dict[str, float]]) -> np.ndarray:
        return _desired_prob(_batch(bundle, base, chs, algo), desired) >= target_p

    refined: dict[tuple, dict[str, Any]] = {}
    for i in seeds:
        ch = dict(combos[i])
        # 1) sparsify: drop single changes that turn out to be unnecessary (one batched call per round)
        while len(ch) > 1:
            trials = [{k: v for k, v in ch.items() if k != n} for n in ch]
            good = ok_batch(trials)
            if not good.any():
                break
            ch = trials[int(np.argmax(good))]
        # 2) shrink numeric changes toward the original value, keeping the flip (one batched call per feature)
        for n in list(ch):
            f = FEATURE_BY_NAME[n]
            if f["kind"] != "numeric":
                continue
            step = f["step"] or 1
            fracs = np.linspace(0.05, 1.0, 20)
            vals = [round(round((base[n] + t * (ch[n] - base[n])) / step) * step, 2) for t in fracs]
            keep = [(v) for v in dict.fromkeys(vals) if abs(v - base[n]) > 1e-9]
            if not keep:
                continue
            good = ok_batch([{**ch, n: v} for v in keep])
            if good.any():
                ch[n] = keep[int(np.argmax(good))]
        key = tuple(sorted((k, round(v, 2)) for k, v in ch.items()))
        refined[key] = ch

    cands = []
    for ch in refined.values():
        cand = {**base, **ch}
        p1 = float(predict_proba1(bundle, cand, algo)[0])  # independent re-verification
        cls = int(p1 >= config.DECISION_THRESHOLD)
        pdes = p1 if desired == 1 else 1 - p1
        valid = (cls == desired) and pdes >= target_p and all(
            bundle.meta["observed_ranges"][n]["min"] - 1e-9 <= v <= bundle.meta["observed_ranges"][n]["max"] + 1e-9 for n, v in ch.items())
        if not valid:
            continue
        z = bundle.pipeline(algo).named_steps["prep"].transform(to_frame(cand))
        dist = float(bundle.nn.kneighbors(z, n_neighbors=1)[0][0][0])
        cands.append({
            "changes": [{"feature": n, "label": FEATURE_BY_NAME[n]["label"], "kind": FEATURE_BY_NAME[n]["kind"],
                         "unit": FEATURE_BY_NAME[n]["unit"], "from": base[n], "to": v, "delta": round(v - base[n], 3),
                         "from_display": (FEATURE_BY_NAME[n]["options"] or {}).get(int(base[n]), None) if FEATURE_BY_NAME[n]["options"] else None,
                         "to_display": (FEATURE_BY_NAME[n]["options"] or {}).get(int(v), None) if FEATURE_BY_NAME[n]["options"] else None,
                         "relative_to_range": round(abs(v - base[n]) / _scale(bundle, n), 3)} for n, v in ch.items()],
            "n_changes": len(ch), "cost": round(_cost(bundle, base, cand, list(ch)), 4),
            "probability_class1": p1, "predicted_class": cls, "verified": True,
            "within_observed_ranges": True,
            "nearest_training_distance": dist, "typical": bool(dist <= bundle.nn_p95),
            "plausibility_note": ("Close to real training patients" if dist <= bundle.nn_p95 else
                                  "Unusual combination: farther from any real patient than 95% of training patients are from their nearest neighbour"),
            "input": {k: (int(v) if FEATURE_BY_NAME[k]["kind"] != "numeric" or FEATURE_BY_NAME[k]["step"] == 1 else float(v)) for k, v in cand.items()},
        })
    # rank by minimal change; candidates far from any real patient get a small penalty so typical ones surface first
    cands.sort(key=lambda c: c["cost"] + (0.0 if c["typical"] else 0.75))
    picked, per_set = [], {}
    for c in cands:
        k = tuple(sorted(x["feature"] for x in c["changes"]))
        if per_set.get(k, 0) >= 2:
            continue
        per_set[k] = per_set.get(k, 0) + 1
        picked.append(c)
        if len(picked) >= n_results:
            break
    result.update(found=bool(picked), candidates=picked,
                  n_candidates_before_verification=len(refined),
                  reason=None if picked else "Candidates failed verification after refinement.")
    return result
