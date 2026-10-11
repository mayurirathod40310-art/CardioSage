"""Dataset loading, schema normalisation, validation and profiling (all real calculations)."""
from __future__ import annotations
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .schema import (FEATURES, FEATURE_ALIASES, FEATURE_BY_NAME, FEATURE_NAMES, NUMERIC, DISCRETE,
                     TARGET, TARGET_ALIASES)


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or does not fit the expected schema."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map common column-name variants to the canonical schema.
    Returns (frame, mapping original->canonical). Raises DatasetError on missing columns."""
    lookup: dict[str, str] = {}
    for canon, aliases in FEATURE_ALIASES.items():
        for a in aliases:
            lookup[_canon(a)] = canon
    mapping: dict[str, str] = {}
    target_col = None
    for col in df.columns:
        c = _canon(col)
        if c in lookup and lookup[c] not in mapping.values():
            mapping[col] = lookup[c]
        elif c in [_canon(t) for t in TARGET_ALIASES] and target_col is None:
            target_col = col
    if target_col is None:
        raise DatasetError(f"No target column found. Expected one of {TARGET_ALIASES}; got {list(df.columns)}.")
    mapping[target_col] = TARGET
    missing = [f for f in FEATURE_NAMES if f not in mapping.values()]
    if missing:
        raise DatasetError(f"Dataset is missing required feature columns: {missing}. Columns present: {list(df.columns)}.")
    out = df[list(mapping.keys())].rename(columns=mapping)
    return out[FEATURE_NAMES + [TARGET]], mapping


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["sex"].dtype == object:
        df["sex"] = df["sex"].astype(str).str.strip().str.lower().map(
            {"m": 1, "male": 1, "1": 1, "f": 0, "female": 0, "0": 0})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c].replace("?", np.nan), errors="coerce")
    # UCI "num" is 0-4 (severity). Anything > 0 means disease present.
    if df[TARGET].dropna().max() > 1:
        df[TARGET] = (df[TARGET] > 0).astype(float).where(df[TARGET].notna())
    return df


def load_raw(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, normalise and validate. Returns the cleaned frame (duplicates retained) and a load report."""
    path = Path(path or config.DATA_PATH)
    if not path.exists():
        raise DatasetError(f"Dataset not found at {path}. Place heart.csv there or set CARDIOSAGE_DATA.")
    try:
        df0 = pd.read_csv(path)
    except Exception as exc:  # malformed file
        raise DatasetError(f"Could not parse {path}: {exc}") from exc
    df, mapping = normalise_columns(df0)
    df = _coerce(df)
    report: dict[str, Any] = {
        "path": str(path), "sha256": sha256_of(path), "rows_in_file": int(len(df0)),
        "column_mapping": {k: v for k, v in mapping.items() if k != v}, "dropped": {},
    }
    n_missing_target = int(df[TARGET].isna().sum())
    if n_missing_target:
        df = df[df[TARGET].notna()]
        report["dropped"]["missing_target"] = n_missing_target
    # Invalid values (outside physiological limits or undeclared category) become missing, then imputed in-pipeline.
    invalid: dict[str, int] = {}
    for f in FEATURES:
        s = df[f["name"]]
        bad = pd.Series(False, index=df.index)
        if f["hard"]:
            bad = (s < f["hard"][0]) | (s > f["hard"][1])
        if f["options"]:
            bad = bad | (s.notna() & ~s.isin(list(f["options"].keys())))
        if bad.any():
            invalid[f["name"]] = int(bad.sum())
            df.loc[bad, f["name"]] = np.nan
    report["invalid_values_set_missing"] = invalid
    bad_t = ~df[TARGET].isin([0, 1])
    if bad_t.any():
        raise DatasetError(f"Target column contains values other than 0/1: {sorted(df[TARGET].unique())[:10]}")
    df[TARGET] = df[TARGET].astype(int)
    for c in DISCRETE:
        if df[c].notna().all():
            df[c] = df[c].astype(int)
    report["rows_after_validation"] = int(len(df))
    return df.reset_index(drop=True), report


def load_unique(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and drop exact duplicate rows (the training/evaluation population)."""
    df, rep = load_raw(path)
    uniq = df.drop_duplicates().reset_index(drop=True)
    rep["rows_unique"] = int(len(uniq))
    rep["duplicates_removed"] = int(len(df) - len(uniq))
    return uniq, rep


def observed_ranges(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for f in FEATURES:
        s = df[f["name"]].dropna()
        if f["kind"] == "numeric":
            out[f["name"]] = {"min": float(s.min()), "max": float(s.max()),
                              "p5": float(s.quantile(.05)), "p25": float(s.quantile(.25)), "p50": float(s.median()),
                              "p75": float(s.quantile(.75)), "p95": float(s.quantile(.95))}
        else:
            out[f["name"]] = {"values": sorted(int(v) for v in s.unique()),
                              "min": float(s.min()), "max": float(s.max())}
    return out


def _hist(values: pd.Series, bins: np.ndarray) -> list[int]:
    return np.histogram(values, bins=bins)[0].astype(int).tolist()


def label_direction_check(df: pd.DataFrame) -> dict[str, Any]:
    """Test whether target=1 behaves like 'disease present' or 'disease absent'.

    Markers that clinically go with coronary disease: exercise angina, ST depression, vessels coloured, age (+),
    max heart rate (-). If the label is documented as 1=disease, those should correlate positively with target
    (and thalach negatively). We compute the point-biserial correlations and report the direction honestly."""
    risk_up = ["exang", "oldpeak", "ca", "age"]
    corr = {c: float(df[c].corr(df[TARGET])) for c in risk_up + ["thalach"]}
    votes_inverted = sum(corr[c] < 0 for c in risk_up) + (corr["thalach"] > 0)
    total = len(risk_up) + 1
    if votes_inverted == total:
        verdict = "inverted"
        text = ("All five clinical markers move opposite to the 'target=1 means disease' convention: exercise angina, ST depression, "
                "vessel count and age are LOWER, and maximum heart rate is HIGHER, when target=1. In this copy, target=1 therefore "
                "behaves like the absence of angiographic disease and target=0 like its presence. This is inferred from the data only "
                "and should be confirmed against the original source.")
    elif votes_inverted == 0:
        verdict = "as_documented"
        text = "Clinical markers move in the direction expected if target=1 means disease present."
    else:
        verdict = "inconclusive"
        text = "Clinical markers give mixed evidence about the label direction."
    return {"verdict": verdict, "correlations_with_target": corr, "markers_inverted": int(votes_inverted),
            "markers_tested": total, "explanation": text}


def profile(path: Path | None = None) -> dict[str, Any]:
    """Full dataset profile used by the dashboard and explorer. All values are computed here."""
    raw, rep = load_raw(path)
    uniq = raw.drop_duplicates().reset_index(drop=True)
    n_dup = int(len(raw) - len(uniq))
    feats: dict[str, Any] = {}
    for f in FEATURES:
        s = raw[f["name"]]
        su = uniq[f["name"]]
        entry: dict[str, Any] = {"name": f["name"], "label": f["label"], "kind": f["kind"], "unit": f["unit"],
                                 "missing": int(s.isna().sum()), "n_unique": int(su.nunique())}
        if f["kind"] == "numeric":
            q1, q3 = su.quantile(.25), su.quantile(.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_mask = (su < lo) | (su > hi)
            edges = np.linspace(su.min(), su.max(), 13)
            centers = ((edges[:-1] + edges[1:]) / 2).round(2).tolist()
            entry.update({
                "mean": float(su.mean()), "std": float(su.std()), "min": float(su.min()), "max": float(su.max()),
                "median": float(su.median()), "q1": float(q1), "q3": float(q3), "skew": float(su.skew()),
                "outliers": {"count": int(out_mask.sum()), "lower_fence": float(lo), "upper_fence": float(hi),
                             "values": sorted(su[out_mask].round(2).tolist())[:20]},
                "histogram": [{"bin": c, "all": a, "class0": b0, "class1": b1} for c, a, b0, b1 in zip(
                    centers, _hist(su, edges), _hist(su[uniq[TARGET] == 0], edges), _hist(su[uniq[TARGET] == 1], edges))],
            })
        else:
            vc = su.value_counts().sort_index()
            rows = []
            for v, n in vc.items():
                sub = uniq[su == v][TARGET]
                rows.append({"value": int(v), "label": (f["options"] or {}).get(int(v), str(v)), "count": int(n),
                             "class0": int((sub == 0).sum()), "class1": int((sub == 1).sum()),
                             "rate_class1": float((sub == 1).mean())})
            entry["categories"] = rows
        feats[f["name"]] = entry

    cols = FEATURE_NAMES + [TARGET]
    corr_p = uniq[cols].corr(method="pearson")
    corr_s = uniq[cols].corr(method="spearman")
    target_corr = sorted(({"feature": c, "pearson": float(corr_p.loc[c, TARGET]), "spearman": float(corr_s.loc[c, TARGET])}
                          for c in FEATURE_NAMES), key=lambda r: -abs(r["pearson"]))
    # strongest pairs among features
    pairs = []
    for i, a in enumerate(FEATURE_NAMES):
        for b in FEATURE_NAMES[i + 1:]:
            pairs.append({"a": a, "b": b, "pearson": float(corr_p.loc[a, b])})
    pairs.sort(key=lambda r: -abs(r["pearson"]))

    counts_raw = raw[TARGET].value_counts().sort_index()
    counts_u = uniq[TARGET].value_counts().sort_index()
    n_u = len(uniq)
    minority = float(counts_u.min() / n_u)

    warnings: list[dict[str, str]] = []
    if n_dup:
        warnings.append({"level": "high", "code": "duplicates",
                         "message": f"{n_dup} of {len(raw)} rows ({n_dup / len(raw):.0%}) are exact duplicates. Only {n_u} unique records exist, "
                                    "so models are trained and evaluated on de-duplicated data to avoid train/test leakage."})
    if n_u < 500:
        warnings.append({"level": "high", "code": "small_sample",
                         "message": f"Only {n_u} unique records: performance estimates will be wide and unstable."})
    if minority < 0.4:
        warnings.append({"level": "medium", "code": "imbalance", "message": f"Minority class share is {minority:.0%}."})
    if int((uniq["ca"] == 4).sum()):
        warnings.append({"level": "medium", "code": "ca_4",
                         "message": f"ca = 4 appears in {int((uniq['ca'] == 4).sum())} unique rows; documented range is 0-3, so it probably encodes missing."})
    if int((uniq["thal"] == 0).sum()):
        warnings.append({"level": "medium", "code": "thal_0",
                         "message": f"thal = 0 appears in {int((uniq['thal'] == 0).sum())} unique rows and is not a documented result; probably a missing-value code."})
    if rep["invalid_values_set_missing"]:
        warnings.append({"level": "high", "code": "invalid", "message": f"Invalid values were set to missing: {rep['invalid_values_set_missing']}"})
    direction = label_direction_check(uniq)
    if direction["verdict"] == "inverted":
        warnings.append({"level": "high", "code": "label_direction",
                         "message": "The target label appears inverted relative to its common documentation (see Methodology)."})
    warnings.append({"level": "info", "code": "provenance",
                     "message": "The dataset pools patients from four clinical sites; it does not represent any general population."})

    return {
        "source": config.DATASET_SOURCE,
        "file": {"path": rep["path"], "sha256": rep["sha256"], "rows_in_file": rep["rows_in_file"],
                 "column_mapping": rep["column_mapping"]},
        "shape": {"rows": int(len(raw)), "unique_rows": n_u, "duplicate_rows": n_dup, "columns": len(cols)},
        "missing_total": int(raw.isna().sum().sum()),
        "missing_by_column": {c: int(raw[c].isna().sum()) for c in cols},
        "class_distribution": {
            "raw": [{"class": int(k), "count": int(v), "share": float(v / len(raw))} for k, v in counts_raw.items()],
            "unique": [{"class": int(k), "count": int(v), "share": float(v / n_u)} for k, v in counts_u.items()],
        },
        "duplicate_summary": {
            "duplicate_rows": n_dup,
            "max_copies_of_one_row": int(raw.groupby(cols).size().max()),
            "rows_with_a_duplicate": int(raw.duplicated(keep=False).sum()),
        },
        "features": feats,
        "feature_types": {f["name"]: f["kind"] for f in FEATURES},
        "correlation": {"columns": cols, "pearson": corr_p.round(4).values.tolist(), "spearman": corr_s.round(4).values.tolist()},
        "target_correlation": target_corr,
        "top_feature_pairs": pairs[:8],
        "label_direction": direction,
        "warnings": warnings,
        "observed_ranges": observed_ranges(uniq),
        "preview": raw.head(15).to_dict(orient="records"),
    }


@lru_cache(maxsize=4)
def _cached_profile(path_str: str, mtime: float) -> dict[str, Any]:
    return profile(Path(path_str))


def cached_profile() -> dict[str, Any]:
    p = Path(config.DATA_PATH)
    return _cached_profile(str(p), p.stat().st_mtime if p.exists() else 0.0)
