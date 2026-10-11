from cardiosage import db


def setup_function(_):
    db.init_db(); db.delete_all_predictions()


def test_prediction_persists_filters_sorts_and_deletes():
    a = db.add_prediction("v1", "random_forest", {"age": 50}, 1, 0.9, [{"feature": "cp", "contribution": 0.1}])
    b = db.add_prediction("v1", "svm_rbf", {"age": 60}, 0, 0.2, [])
    assert db.get_prediction(a)["inputs"] == {"age": 50}
    assert db.list_predictions()["total"] == 2
    assert [r["id"] for r in db.list_predictions(predicted_class=0)["items"]] == [b]
    assert [r["id"] for r in db.list_predictions(sort="probability", order="asc")["items"]] == [b, a]
    assert [r["id"] for r in db.list_predictions(min_p=0.5)["items"]] == [a]
    assert db.delete_prediction(a) and not db.delete_prediction(a)
    assert db.delete_all_predictions() == 1 and db.list_predictions()["total"] == 0


def test_experiments_roundtrip():
    eid = db.add_experiment("scenario", "t", {"x": 1}, "v1", {"c": 2})
    e = db.get_experiment(eid)
    assert e["summary"] == {"x": 1} and e["config"] == {"c": 2}
    assert db.list_experiments(kind="scenario")["total"] >= 1
    assert db.delete_experiment(eid) and db.get_experiment(eid) is None
