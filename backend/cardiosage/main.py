"""CardioSage REST API (FastAPI). Run: uvicorn cardiosage.main:app --port 8000   Docs at /docs"""
from __future__ import annotations
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, config, counterfactual, data, db, scenario, store, training
from .explain import explain_instance
from .schema import CLASS_LABELS, FEATURE_BY_NAME, FEATURES, FEATURE_NAMES
from .store import NoModelError

STARTED = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="CardioSage API", version=__version__, lifespan=lifespan,
              description="Explainable heart-disease modelling with counterfactual analysis. Research/education software, "
                          "NOT a medical device. Predictions are model outputs, not diagnoses.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- errors
@app.exception_handler(NoModelError)
async def _no_model(_: Request, exc: NoModelError):
    return JSONResponse(status_code=409, content={"detail": {"code": "model_not_trained", "message": str(exc)}})


@app.exception_handler(data.DatasetError)
async def _dataset_err(_: Request, exc: data.DatasetError):
    return JSONResponse(status_code=503, content={"detail": {"code": "dataset_unavailable", "message": str(exc)}})


# ---------------------------------------------------------------- schemas
class PatientInput(BaseModel):
    """One record using the dataset's 13 feature columns. Limits are physiological sanity bounds."""
    age: int = Field(..., ge=18, le=100, description="Years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[0, 1, 2, 3] = Field(..., description="Chest pain type")
    trestbps: int = Field(..., ge=70, le=250, description="Resting systolic blood pressure, mmHg")
    chol: int = Field(..., ge=100, le=700, description="Serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: int = Field(..., ge=50, le=230, description="Maximum heart rate, bpm")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression, mm")
    slope: Literal[0, 1, 2] = Field(..., description="Peak exercise ST slope")
    ca: Literal[0, 1, 2, 3, 4] = Field(..., description="Major vessels coloured (4 likely encodes missing)")
    thal: Literal[0, 1, 2, 3] = Field(..., description="Thallium test code (0 likely encodes missing)")

    model_config = {"json_schema_extra": {"example": {
        "age": 57, "sex": 1, "cp": 0, "trestbps": 140, "chol": 241, "fbs": 0, "restecg": 1, "thalach": 123,
        "exang": 1, "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3}}}


class PredictRequest(BaseModel):
    features: PatientInput
    algorithm: Optional[str] = Field(None, description="Model key; defaults to the selected model")
    explain: bool = True
    save: bool = Field(True, description="Store inputs and result in local history")


class ExplainRequest(BaseModel):
    features: PatientInput
    algorithm: Optional[str] = None


class CounterfactualRequest(BaseModel):
    features: PatientInput
    algorithm: Optional[str] = None
    desired_class: Optional[Literal[0, 1]] = Field(None, description="Defaults to the opposite of the current prediction")
    locked_features: list[str] = Field(default_factory=lambda: ["age", "sex"])
    max_changes: int = Field(3, ge=1, le=3)
    n_results: int = Field(5, ge=1, le=8)
    margin: float = Field(0.05, ge=0.0, le=0.3, description="Required probability margin beyond 0.5")
    save: bool = True


class ScenarioRequest(BaseModel):
    baseline: PatientInput
    scenario: PatientInput
    algorithm: Optional[str] = None
    save: bool = True


class TrainRequest(BaseModel):
    seed: int = Field(config.RANDOM_STATE, ge=0, le=2**31 - 1)


# ---------------------------------------------------------------- helpers
def _bundle(version: str | None = None) -> store.Bundle:
    return store.load_bundle(version)


def _algo(b: store.Bundle, algo: str | None) -> str:
    algo = algo or b.selected
    if algo not in b.pipelines:
        raise HTTPException(404, {"code": "unknown_algorithm", "message": f"Unknown algorithm '{algo}'. Available: {sorted(b.pipelines)}"})
    return algo


def _input_warnings(b: store.Bundle, row: dict[str, Any]) -> list[dict[str, str]]:
    w = []
    for n, v in row.items():
        o = b.meta["observed_ranges"][n]
        if FEATURE_BY_NAME[n]["kind"] == "numeric":
            if v < o["min"] or v > o["max"]:
                w.append({"feature": n, "code": "extrapolation",
                          "message": f"{FEATURE_BY_NAME[n]['label']} = {v:g} is outside the range seen in training data ({o['min']:g}–{o['max']:g}); the model is extrapolating."})
        elif int(v) not in o["values"]:
            w.append({"feature": n, "code": "unseen_category", "message": f"{FEATURE_BY_NAME[n]['label']} = {v} never occurs in the training data."})
    if row.get("ca") == 4:
        w.append({"feature": "ca", "code": "missing_code", "message": "ca = 4 probably encodes a missing value in this dataset."})
    if row.get("thal") == 0:
        w.append({"feature": "thal", "code": "missing_code", "message": "thal = 0 probably encodes a missing value in this dataset."})
    return w


def _model_context(b: store.Bundle, algo: str) -> dict[str, Any]:
    m = b.meta["models"][algo]
    return {"cv_roc_auc": m["cv"]["metrics"]["roc_auc"]["mean"], "cv_roc_auc_std": m["cv"]["metrics"]["roc_auc"]["std"],
            "test_accuracy": m["test"]["accuracy"], "test_roc_auc": m["test"]["roc_auc"], "n_test": m["test"]["n"],
            "test_accuracy_ci": (m["test_ci"] or {}).get("accuracy")}


DISCLAIMER = ("This is a statistical model output on a small research dataset, not a diagnosis, risk score or medical advice. "
              "Probabilities are not validated or calibrated against real-world outcomes.")


# ---------------------------------------------------------------- system
@app.get("/api/health", tags=["system"])
def health():
    v = store.current_version()
    ds = config.DATA_PATH.exists()
    try:
        import shap  # noqa: F401
        shap_ok = True
    except Exception:
        shap_ok = False
    try:
        db.init_db(); db_ok = True
    except Exception:
        db_ok = False
    status = "ok" if (ds and v and db_ok) else "degraded"
    return {"status": status, "version": __version__, "uptime_seconds": round(time.time() - STARTED, 1),
            "dataset_available": ds, "model_available": bool(v), "active_model_version": v, "database_ok": db_ok,
            "shap_available": shap_ok, "training": _train_state.copy()}


@app.get("/api/overview", tags=["system"])
def overview():
    prof = data.cached_profile()
    v = store.current_version()
    out: dict[str, Any] = {"health": health(), "dataset": {k: prof[k] for k in ("shape", "class_distribution", "source")},
                           "warnings": prof["warnings"], "label_direction": prof["label_direction"],
                           "recent_experiments": db.list_experiments(limit=5)["items"], "prediction_stats": db.prediction_stats()}
    if v:
        b = _bundle()
        out["model"] = {"version": b.version, "selected": b.selected, "created_at": b.meta["created_at"], "split": b.meta["split"],
                        "models": [{"key": k, "label": m["label"], "cv_roc_auc": m["cv"]["metrics"]["roc_auc"]["mean"],
                                    "test_accuracy": m["test"]["accuracy"], "test_roc_auc": m["test"]["roc_auc"],
                                    "test_f1": m["test"]["f1"]} for k, m in b.meta["models"].items()]}
    return out


# ---------------------------------------------------------------- dataset
@app.get("/api/dataset/summary", tags=["dataset"])
def dataset_summary():
    return data.cached_profile()


@app.get("/api/dataset/schema", tags=["dataset"])
def dataset_schema():
    prof = data.cached_profile()
    return {"features": FEATURES, "class_labels": CLASS_LABELS, "observed_ranges": prof["observed_ranges"],
            "locked_by_default": [f["name"] for f in FEATURES if f["locked"]]}


@app.get("/api/dataset/samples", tags=["dataset"])
def dataset_samples(n: int = Query(5, ge=1, le=20), seed: int = 0):
    """Random real (de-duplicated) records, useful as starting points in the studio."""
    df, _ = data.load_unique()
    s = df.sample(n=min(n, len(df)), random_state=seed)
    return [{"features": {c: (float(r[c]) if c == "oldpeak" else int(r[c])) for c in FEATURE_NAMES}, "target": int(r["target"])}
            for _, r in s.iterrows()]


# ---------------------------------------------------------------- models
@app.get("/api/models", tags=["models"])
def models():
    b = _bundle()
    return {"active_version": b.version, "selected": b.selected, "selection_rule": b.meta["selection_rule"],
            "algorithms": [{"key": k, "label": m["label"], "why": m["why"], "params": m["params"], "selected": k == b.selected,
                            "cv": m["cv"]["metrics"], "test": {kk: vv for kk, vv in m["test"].items() if not isinstance(vv, (list, dict))},
                            "fit_seconds": m["fit_seconds"]} for k, m in b.meta["models"].items()]}


@app.get("/api/models/versions", tags=["models"])
def model_versions():
    return {"versions": store.list_versions()}


@app.post("/api/models/versions/{version}/activate", tags=["models"])
def activate_version(version: str):
    try:
        store.set_current(version)
    except KeyError as e:
        raise HTTPException(404, {"code": "unknown_version", "message": str(e)})
    return {"active_version": version}


@app.get("/api/benchmark", tags=["models"])
def benchmark(version: Optional[str] = None):
    """Full benchmark evidence: CV metrics, held-out metrics, curves, confusion matrices, tuning grids."""
    b = _bundle(version)
    return {"version": b.version, "created_at": b.meta["created_at"], "selected": b.selected, "selection_rule": b.meta["selection_rule"],
            "split": b.meta["split"], "models": b.meta["models"], "leakage_check": b.meta["leakage_check"],
            "dataset": b.meta["dataset"], "environment": b.meta["environment"], "train_seconds": b.meta["train_seconds"]}


_train_state: dict[str, Any] = {"status": "idle", "message": "", "progress": 0.0, "version": None, "error": None}
_train_lock = threading.Lock()


def _run_training(seed: int) -> None:
    def cb(msg: str, frac: float):
        _train_state.update(message=msg, progress=round(frac, 3))
    try:
        m = training.train_all(progress=cb, seed=seed)
        store.clear_cache()
        _train_state.update(status="done", version=m["version"], message="Training finished", progress=1.0, error=None)
    except Exception as exc:
        _train_state.update(status="error", error=f"{type(exc).__name__}: {exc}", message="Training failed")


@app.post("/api/train", status_code=202, tags=["models"])
def train(req: TrainRequest = TrainRequest()):
    """Start a background training run (about a minute). Poll GET /api/train/status."""
    with _train_lock:
        if _train_state["status"] == "running":
            raise HTTPException(409, {"code": "training_in_progress", "message": "A training run is already in progress."})
        _train_state.update(status="running", message="Starting", progress=0.0, version=None, error=None, started_at=time.time())
    threading.Thread(target=_run_training, args=(req.seed,), daemon=True).start()
    return _train_state.copy()


@app.get("/api/train/status", tags=["models"])
def train_status():
    return _train_state.copy()


# ---------------------------------------------------------------- prediction
@app.post("/api/predict", tags=["prediction"])
def predict(req: PredictRequest):
    b = _bundle()
    algo = _algo(b, req.algorithm)
    row = req.features.model_dump()
    pred = store.predict_one(b, row, algo)
    out: dict[str, Any] = {
        "model_version": b.version, "algorithm": algo, "algorithm_label": b.meta["models"][algo]["label"], **pred,
        "predicted_class_label": CLASS_LABELS[pred["predicted_class"]], "input": row,
        "warnings": _input_warnings(b, row), "model_context": _model_context(b, algo), "disclaimer": DISCLAIMER,
        "model_agreement": [{"algorithm": k, "label": b.meta["models"][k]["label"], "probability_class1": float(store.predict_proba1(b, row, k)[0]),
                             "predicted_class": int(store.predict_proba1(b, row, k)[0] >= config.DECISION_THRESHOLD)}
                            for k in b.pipelines if k != "baseline"],
        "explanation": None, "saved_id": None,
    }
    if req.explain:
        out["explanation"] = explain_instance(b, row, algo)
    if req.save:
        top = [{"feature": c["feature"], "contribution": round(c["contribution"], 4)} for c in (out["explanation"] or {}).get("contributions", [])[:5]]
        out["saved_id"] = db.add_prediction(b.version, algo, row, pred["predicted_class"], pred["probability_class1"], top)
    return out


@app.post("/api/explain", tags=["explanation"])
def explain(req: ExplainRequest):
    b = _bundle()
    algo = _algo(b, req.algorithm)
    row = req.features.model_dump()
    return {"input": row, "model_version": b.version, "warnings": _input_warnings(b, row),
            **explain_instance(b, row, algo), **store.predict_one(b, row, algo)}


@app.get("/api/explain/global", tags=["explanation"])
def explain_global(algorithm: Optional[str] = None):
    b = _bundle()
    algo = _algo(b, algorithm)
    prof = data.cached_profile()
    m = b.meta["models"][algo]
    perm = {r["feature"]: r for r in m["permutation_importance"]}
    tc = {r["feature"]: r for r in prof["target_correlation"]}
    return {
        "algorithm": algo, "label": m["label"], "model_version": b.version, "n_test": b.meta["split"]["n_test"],
        "permutation_importance": m["permutation_importance"],
        "comparison": [{"feature": f, "label": FEATURE_BY_NAME[f]["label"], "permutation_importance": perm[f]["mean"],
                        "permutation_std": perm[f]["std"], "pearson_with_target": tc[f]["pearson"]} for f in FEATURE_NAMES],
        "concepts": [
            {"term": "Correlation", "text": "How two columns move together across the dataset (Pearson/Spearman). It says nothing about the model and nothing about cause."},
            {"term": "Feature importance", "text": "How much the model's held-out score drops when a feature is shuffled (permutation importance). It is global, specific to this model, and unstable on small test sets, so read the error bars."},
            {"term": "Prediction explanation", "text": "A per-patient decomposition (KernelSHAP) of why this model output this probability versus the background average. It explains the model, not the patient."},
            {"term": "Causation", "text": "Whether changing a factor would change an outcome. None of the above establishes this; it needs experimental or carefully designed causal evidence."},
        ],
    }


# ---------------------------------------------------------------- counterfactual + scenarios
@app.post("/api/counterfactual", tags=["counterfactual"])
def counterfactual_endpoint(req: CounterfactualRequest):
    b = _bundle()
    algo = _algo(b, req.algorithm)
    row = req.features.model_dump()
    bad = [n for n in req.locked_features if n not in FEATURE_NAMES]
    if bad:
        raise HTTPException(422, {"code": "unknown_feature", "message": f"Unknown locked feature(s): {bad}"})
    t = time.time()
    res = counterfactual.generate(b, row, algo, req.desired_class, req.locked_features, req.max_changes, req.n_results, req.margin)
    res.update(input=row, model_version=b.version, warnings=_input_warnings(b, row), seconds=round(time.time() - t, 2))
    if req.save:
        res["experiment_id"] = db.add_experiment(
            "counterfactual", f"Counterfactual search ({res['original']['predicted_class']}→{res['desired_class']}): "
            f"{'found ' + str(len(res['candidates'])) if res['found'] else 'none found'}",
            {"found": res["found"], "n_candidates": len(res["candidates"]), "evaluated": res["evaluated"],
             "best": res["candidates"][0]["changes"] if res["candidates"] else None, "input": row,
             "original_probability": res["original"]["probability_class1"]},
            model_version=b.version, config_={"algorithm": algo, "locked": req.locked_features, "max_changes": req.max_changes, "margin": req.margin})
    return res


@app.post("/api/scenarios/compare", tags=["scenarios"])
def scenarios_compare(req: ScenarioRequest):
    b = _bundle()
    algo = _algo(b, req.algorithm)
    a, s = req.baseline.model_dump(), req.scenario.model_dump()
    res = scenario.compare(b, a, s, algo)
    res.update(model_version=b.version, warnings=_input_warnings(b, s), baseline=a, scenario=s)
    if req.save:
        res["experiment_id"] = db.add_experiment(
            "scenario", f"What-if: {len(res['changed_features'])} change(s), Δp(target=1) = {res['probability_difference']:+.3f}",
            {"changes": res["changes"], "a": {k: res["a"][k] for k in ("predicted_class", "probability_class1")},
             "b": {k: res["b"][k] for k in ("predicted_class", "probability_class1")}, "class_flipped": res["class_flipped"],
             "baseline": a, "scenario": s}, model_version=b.version, config_={"algorithm": algo})
    return res


# ---------------------------------------------------------------- reliability
@app.get("/api/reliability", tags=["models"])
def reliability(algorithm: Optional[str] = None):
    b = _bundle()
    algo = _algo(b, algorithm)
    m = b.meta["models"][algo]
    cands = sorted(((k, v["cv"]["metrics"]["roc_auc"]) for k, v in b.meta["models"].items() if k != "baseline"), key=lambda kv: -kv[1]["mean"])
    margin = cands[0][1]["mean"] - cands[1][1]["mean"]
    spread = max(c[1]["std"] for c in cands[:2])
    return {
        "algorithm": algo, "label": m["label"], "model_version": b.version, "split": b.meta["split"],
        "test": {k: v for k, v in m["test"].items() if k not in ("roc_curve", "pr_curve")},
        "test_ci": m["test_ci"], "cv": m["cv"], "calibration": m["test"].get("calibration", []),
        "selection": {"ranking": [{"algorithm": k, "cv_roc_auc_mean": v["mean"], "cv_roc_auc_std": v["std"]} for k, v in cands],
                      "margin_between_top_two": margin, "fold_to_fold_std": spread,
                      "statistically_distinguishable": bool(margin > spread),
                      "text": f"The top two models differ by {margin:.3f} AUC while fold-to-fold variation is about {spread:.3f}; "
                              + ("the ranking is not reliable, so treat the top models as equivalent." if margin <= spread else "the leader is ahead by more than the noise level.")},
        "leakage_check": b.meta["leakage_check"],
        "evidence_not_available": {
            "clinical_validation": False, "external_validation": False, "fairness_audit": False, "robustness_audit": False,
            "calibration_in_real_world": False, "prospective_evaluation": False,
            "note": "None of these properties were evaluated, so none are claimed."},
        "limitations": [
            f"Only {b.meta['dataset']['unique_rows']} unique patients; the held-out test set has {b.meta['split']['n_test']} rows, so one extra mistake moves accuracy by about {100 / b.meta['split']['n_test']:.1f} points.",
            "The dataset pools four historical clinical sites and does not represent any current or general population.",
            "Duplicate rows were removed before splitting; results on the raw 1,025 rows would be inflated (see leakage check).",
            "Probabilities are model outputs. Calibration was only inspected on the small test set (5 quantile bins) and is not a medical risk.",
            "The label direction in this copy appears inverted relative to its documentation (see Methodology).",
        ],
    }


# ---------------------------------------------------------------- history
@app.get("/api/history/predictions", tags=["history"])
def history_predictions(limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0), predicted_class: Optional[Literal[0, 1]] = None,
                        algorithm: Optional[str] = None, model_version: Optional[str] = None, min_probability: Optional[float] = Query(None, ge=0, le=1),
                        max_probability: Optional[float] = Query(None, ge=0, le=1), sort: Literal["created_at", "probability", "id", "algorithm"] = "created_at",
                        order: Literal["asc", "desc"] = "desc"):
    return db.list_predictions(limit, offset, predicted_class, algorithm, model_version, None, min_probability, max_probability, sort, order)


@app.get("/api/history/predictions/{pid}", tags=["history"])
def history_prediction(pid: int):
    r = db.get_prediction(pid)
    if not r:
        raise HTTPException(404, {"code": "not_found", "message": f"Prediction {pid} not found."})
    return r


@app.delete("/api/history/predictions/{pid}", tags=["history"])
def history_prediction_delete(pid: int):
    if not db.delete_prediction(pid):
        raise HTTPException(404, {"code": "not_found", "message": f"Prediction {pid} not found."})
    return {"deleted": pid}


@app.delete("/api/history/predictions", tags=["history"])
def history_predictions_clear():
    return {"deleted": db.delete_all_predictions()}


@app.get("/api/history/experiments", tags=["history"])
def history_experiments(limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0),
                        kind: Optional[Literal["training", "counterfactual", "scenario"]] = None, order: Literal["asc", "desc"] = "desc"):
    return db.list_experiments(limit, offset, kind, order)


@app.get("/api/history/experiments/{eid}", tags=["history"])
def history_experiment(eid: int):
    r = db.get_experiment(eid)
    if not r:
        raise HTTPException(404, {"code": "not_found", "message": f"Experiment {eid} not found."})
    return r


@app.delete("/api/history/experiments/{eid}", tags=["history"])
def history_experiment_delete(eid: int):
    if not db.delete_experiment(eid):
        raise HTTPException(404, {"code": "not_found", "message": f"Experiment {eid} not found."})
    return {"deleted": eid}


@app.get("/api/privacy", tags=["history"])
def privacy():
    return {"stored_locally_in": str(config.DB_PATH),
            "prediction_fields": ["timestamp", "model version", "algorithm", "the 13 submitted feature values", "predicted class", "probability", "top 5 contributing features"],
            "not_stored": ["names", "identifiers", "free text", "IP addresses", "device information"],
            "deletion": "Delete single entries or all entries from the History page, or via DELETE /api/history/predictions[/{id}]. Deleting storage/cardiosage.db removes everything."}


# ---------------------------------------------------------------- static frontend (production build)
if config.FRONTEND_DIST.exists() and (config.FRONTEND_DIST / "index.html").exists():
    if (config.FRONTEND_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, {"code": "not_found", "message": "Unknown API route."})
        f = (config.FRONTEND_DIST / path).resolve()
        if path and f.is_file() and config.FRONTEND_DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(config.FRONTEND_DIST / "index.html")
