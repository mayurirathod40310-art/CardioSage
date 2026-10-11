import numpy as np
import pandas as pd
import pytest

from cardiosage import config, data
from cardiosage.schema import FEATURE_NAMES, TARGET


def test_real_dataset_loads_and_matches_schema():
    df, rep = data.load_raw()
    assert list(df.columns) == FEATURE_NAMES + [TARGET]
    assert len(df) == rep["rows_in_file"] == 1025
    assert set(df[TARGET].unique()) == {0, 1}
    assert len(rep["sha256"]) == 64


def test_duplicates_detected_and_removed():
    raw, _ = data.load_raw()
    uniq, rep = data.load_unique()
    assert rep["duplicates_removed"] == len(raw) - len(uniq) == 723
    assert not uniq.duplicated().any()


def test_profile_contents_are_computed_not_hardcoded():
    p = data.profile()
    assert p["shape"]["unique_rows"] == 302
    assert sum(c["count"] for c in p["class_distribution"]["raw"]) == 1025
    assert len(p["correlation"]["pearson"]) == 14
    assert p["features"]["age"]["histogram"] and p["features"]["cp"]["categories"]
    assert {w["code"] for w in p["warnings"]} >= {"duplicates", "small_sample"}


def test_label_direction_check_reports_inversion_on_this_copy():
    assert data.profile()["label_direction"]["verdict"] == "inverted"


def test_column_alias_variant_and_multiclass_target(tmp_path):
    df, _ = data.load_raw()
    v = df.rename(columns={"trestbps": "RestingBP", "chol": "Cholesterol", "thalach": "MaxHR", "exang": "ExerciseAngina",
                           "oldpeak": "Oldpeak", "cp": "ChestPainType", "target": "num"})
    v["num"] = v["num"] * 3  # 0/3 -> must be collapsed to 0/1
    p = tmp_path / "variant.csv"; v.to_csv(p, index=False)
    out, rep = data.load_raw(p)
    assert list(out.columns) == FEATURE_NAMES + [TARGET]
    assert set(out[TARGET].unique()) == {0, 1}
    assert rep["column_mapping"]["RestingBP"] == "trestbps"


def test_missing_markers_and_invalid_values_are_flagged(tmp_path):
    df, _ = data.load_raw()
    d = df.head(60).astype(object).copy()
    d.loc[0, "chol"] = "?"
    d.loc[1, "age"] = 400          # physiologically impossible
    d.loc[2, "cp"] = 9             # undeclared category
    p = tmp_path / "bad.csv"; d.to_csv(p, index=False)
    out, rep = data.load_raw(p)
    assert out["chol"].isna().sum() == 1
    assert rep["invalid_values_set_missing"] == {"age": 1, "cp": 1}


def test_missing_target_column_raises(tmp_path):
    df, _ = data.load_raw()
    p = tmp_path / "nt.csv"; df.drop(columns=[TARGET]).to_csv(p, index=False)
    with pytest.raises(data.DatasetError, match="No target column"):
        data.load_raw(p)


def test_missing_feature_column_raises(tmp_path):
    df, _ = data.load_raw()
    p = tmp_path / "nf.csv"; df.drop(columns=["chol"]).to_csv(p, index=False)
    with pytest.raises(data.DatasetError, match="chol"):
        data.load_raw(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(data.DatasetError, match="not found"):
        data.load_raw(tmp_path / "nope.csv")
