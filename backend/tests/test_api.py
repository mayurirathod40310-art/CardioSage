import joblib
import pandas as pd
import pytest

from cardiosage import config
from cardiosage.schema import FEATURE_NAMES


def _direct_p1(trained, row, algo):
    pipe = joblib.load(config.MODELS_DIR / trained["version"] / f"{algo}.joblib")
    return float(pipe.predict_proba(pd.DataFrame([row])[FEATURE_NAMES])[0, 1])


def test_health_reports_real_state(client, trained):
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and j["active_model_version"] == trained["version"] and j["shap_available"]


def test_predict_matches_saved_model(client, trained, example):
    r = client.post("/api/predict", json={"features": example, "save": False})
    assert r.status_code == 200
    j = r.json()
    assert j["algorithm"] == trained["selected"]
    assert j["probability_class1"] == pytest.approx(_direct_p1(trained, example, j["algorithm"]), abs=1e-9)
    assert j["predicted_class"] == int(j["probability_class1"] >= .5)
    assert "not a diagnosis" in j["disclaimer"]
    for m in j["model_agreement"]:
        assert m["probability_class1"] == pytest.approx(_direct_p1(trained, example, m["algorithm"]), abs=1e-9)


def test_explanation_is_additive_and_matches_prediction(client, example):
    j = client.post("/api/predict", json={"features": example, "save": False}).json()
    e = j["explanation"]
    assert e["method"] == "kernel_shap"
    assert e["prediction"] == pytest.approx(j["probability_class1"], abs=1e-9)
    assert e["base_value"] + e["sum_of_contributions"] == pytest.approx(e["prediction"], abs=1e-3)
    assert len(e["contributions"]) == 13


@pytest.mark.parametrize("patch, field", [({"age": 5}, "age"), ({"cp": 7}, "cp"), ({"trestbps": 900}, "trestbps"),
                                         ({"oldpeak": -1}, "oldpeak"), ({"sex": 3}, "sex")])
def test_invalid_inputs_get_field_level_422(client, example, patch, field):
    r = client.post("/api/predict", json={"features": {**example, **patch}})
    assert r.status_code == 422
    assert any(field in e["loc"] for e in r.json()["detail"])


def test_missing_field_is_rejected(client, example):
    example.pop("chol")
    assert client.post("/api/predict", json={"features": example}).status_code == 422


def test_extrapolation_warning(client, example):
    j = client.post("/api/predict", json={"features": {**example, "chol": 650}, "save": False, "explain": False}).json()
    assert any(w["code"] == "extrapolation" for w in j["warnings"])


def test_unknown_algorithm_404(client, example):
    assert client.post("/api/predict", json={"features": example, "algorithm": "nope"}).status_code == 404


def test_counterfactual_candidates_really_flip_the_model(client, trained, example):
    for algo in ("logistic_regression", "random_forest"):
        j = client.post("/api/counterfactual", json={"features": example, "algorithm": algo, "save": False}).json()
        assert j["found"], j.get("reason")
        orig = _direct_p1(trained, example, algo)
        assert (orig >= .5) == bool(j["original"]["predicted_class"])
        for c in j["candidates"]:
            p = _direct_p1(trained, c["input"], algo)          # independent re-evaluation of the saved model
            assert (p >= .5) == bool(j["desired_class"]) and (p >= .5) != (orig >= .5)
            assert p == pytest.approx(c["probability_class1"], abs=1e-9)
            assert c["verified"] and 1 <= c["n_changes"] <= 3
            changed = {k for k in FEATURE_NAMES if c["input"][k] != example[k]}
            assert changed == {x["feature"] for x in c["changes"]}
            assert not changed & {"age", "sex"}                 # locked features untouched
            for x in c["changes"]:
                lo, hi = trained["observed_ranges"][x["feature"]]["min"], trained["observed_ranges"][x["feature"]]["max"]
                assert lo <= x["to"] <= hi


def test_counterfactual_none_found_is_explained(client, example):
    locked = [n for n in FEATURE_NAMES if n != "fbs"]
    j = client.post("/api/counterfactual", json={"features": example, "algorithm": "logistic_regression", "locked_features": locked, "save": False}).json()
    assert j["found"] is False and j["candidates"] == [] and "No combination" in j["reason"]


def test_counterfactual_unknown_locked_feature(client, example):
    r = client.post("/api/counterfactual", json={"features": example, "locked_features": ["bogus"]})
    assert r.status_code == 422


def test_scenario_comparison(client, trained, example):
    b = {**example, "thalach": 170, "exang": 0}
    j = client.post("/api/scenarios/compare", json={"baseline": example, "scenario": b, "save": False}).json()
    algo = j["algorithm"]
    assert set(j["changed_features"]) == {"thalach", "exang"}
    assert j["a"]["probability_class1"] == pytest.approx(_direct_p1(trained, example, algo), abs=1e-9)
    assert j["b"]["probability_class1"] == pytest.approx(_direct_p1(trained, b, algo), abs=1e-9)
    assert j["probability_difference"] == pytest.approx(j["b"]["probability_class1"] - j["a"]["probability_class1"])
    assert set(j["sweeps"]) == {"thalach", "exang"} and len(j["feature_comparison"]) == 13


def test_history_persistence_filter_and_deletion(client, example):
    client.delete("/api/history/predictions")
    ids = [client.post("/api/predict", json={"features": {**example, "age": 50 + i}}).json()["saved_id"] for i in range(3)]
    assert all(ids)
    lst = client.get("/api/history/predictions?sort=id&order=asc").json()
    assert lst["total"] == 3 and [r["id"] for r in lst["items"]] == ids
    assert client.get(f"/api/history/predictions/{ids[0]}").json()["inputs"]["age"] == 50
    assert client.delete(f"/api/history/predictions/{ids[0]}").status_code == 200
    assert client.get(f"/api/history/predictions/{ids[0]}").status_code == 404
    assert client.get("/api/history/predictions").json()["total"] == 2
    assert client.delete("/api/history/predictions").json()["deleted"] == 2
    j = client.post("/api/predict", json={"features": example, "save": False}).json()
    assert j["saved_id"] is None and client.get("/api/history/predictions").json()["total"] == 0


def test_experiments_recorded_for_cf_and_scenario(client, example):
    before = client.get("/api/history/experiments").json()["total"]
    client.post("/api/counterfactual", json={"features": example, "algorithm": "logistic_regression"})
    client.post("/api/scenarios/compare", json={"baseline": example, "scenario": {**example, "chol": 200}})
    items = client.get("/api/history/experiments").json()
    assert items["total"] == before + 2
    kinds = {i["kind"] for i in items["items"]}
    assert {"counterfactual", "scenario"} <= kinds
    eid = items["items"][0]["id"]
    assert client.delete(f"/api/history/experiments/{eid}").status_code == 200
    assert client.get(f"/api/history/experiments/{eid}").status_code == 404


def test_dashboard_and_benchmark_endpoints_return_real_numbers(client, trained):
    o = client.get("/api/overview").json()
    assert o["dataset"]["shape"]["rows"] == 1025 and o["model"]["version"] == trained["version"]
    b = client.get("/api/benchmark").json()
    assert b["models"]["random_forest"]["test"]["accuracy"] == trained["models"]["random_forest"]["test"]["accuracy"]
    r = client.get("/api/reliability").json()
    assert r["evidence_not_available"]["clinical_validation"] is False
    g = client.get("/api/explain/global").json()
    assert len(g["permutation_importance"]) == 13


def test_train_endpoint_conflict_and_status(client):
    from cardiosage import main
    main._train_state.update(status="running")
    try:
        assert client.post("/api/train").status_code == 409
    finally:
        main._train_state.update(status="idle")
    assert client.get("/api/train/status").json()["status"] in ("idle", "done", "error")


def test_unknown_routes_and_cors(client):
    assert client.get("/api/history/predictions/999999").status_code == 404
    r = client.options("/api/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
