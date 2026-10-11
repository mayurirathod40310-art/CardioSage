# CardioSage

**Explainable heart-disease modelling, counterfactual analysis and benchmarking.**
A FastAPI + scikit-learn backend and a React/TypeScript interface for studying how tabular models behave on the classic UCI heart-disease data,
with honest evidence about what the numbers do and do not mean.

> **Research and education software. Not a medical device. Outputs are model predictions, not diagnoses, and the probabilities are not validated patient risks.**

## Why it is more than a prediction form
* **Leakage-aware evaluation.** The file has 1,025 rows but only **302 distinct records**. CardioSage de-duplicates before splitting and *measures* the damage of not doing so: a naive split gives **100% accuracy** because 98.5% of test rows are copies of training rows.
* **Label sanity check.** It tests (from the data, live) whether `target = 1` behaves like "disease". In this copy it behaves like the opposite, so the UI speaks of `target = 0 / 1` and says why.
* **Counterfactual engine** that searches single, pair and triple feature changes inside observed ranges and categorical constraints, prunes and shrinks them, and **re-scores every reported candidate with the saved model**. Candidates far from any real patient are flagged. No-solution cases are explained.
* **What-if lab** with side-by-side KernelSHAP contributions and per-feature sensitivity sweeps, all computed by the backend.
* **Benchmark lab**: baseline + Logistic Regression, Random Forest, SVM, Gradient Boosting; nested-in-train tuning and repeated CV, one look at the test set, ROC / PR curves, confusion matrices, bootstrap intervals, calibration bins.
* **Explicit non-claims**: clinical validation, external validation, fairness, robustness and real-world calibration are listed as *not evaluated*.

## Quick start (Windows 10/11, PowerShell)
Prerequisites: **Python 3.10-3.13** (tick "Add to PATH") and **Node.js 18+ LTS**. Internet is needed for the first run only.

```powershell
cd path\to\cardiosage
powershell -ExecutionPolicy Bypass -File .\run.ps1
```
`run.ps1` checks prerequisites, creates `.venv`, installs Python and npm dependencies (only when they changed), builds the UI, trains the models if none exist (about one minute), starts the backend, waits for it to be healthy, prints the URLs and opens the browser. **Ctrl+C stops everything.**

* App: http://127.0.0.1:8000  (the FastAPI backend serves the production build)
* API docs (Swagger): http://127.0.0.1:8000/docs

Options: `-Dev` (also start Vite hot-reload on http://localhost:5173), `-Port 8010`, `-NoBrowser`, `-Retrain`, `-Check` (start, verify health + UI + a real prediction, stop; exit code 0 = OK). Separate setup only: `.\setup.ps1`. On Linux/macOS use `./run.sh` (same steps, same flags minus `-Dev`).

## Pages
Overview (all numbers computed by the backend) | Dataset explorer (summary, distributions, correlations, outliers, quality, provenance) | Prediction studio | Counterfactual analysis | What-if laboratory | ML benchmarking (can retrain, lists/activates versions) | Explainability center | History (filter, sort, view, delete) | Methodology and limitations.

## Architecture
```
data/heart.csv ──► backend/cardiosage/data.py  (schema mapping, validation, profiling)
                   training.py  → models/<version>/{*.joblib, metadata.json, train/test/background.csv}
                   store.py     (loads versions, typed prediction helpers)
                   explain.py   (KernelSHAP on the 13 original features; substitution fallback)
                   counterfactual.py, scenario.py
                   db.py        → storage/cardiosage.db (SQLite)
                   main.py      FastAPI: validated schemas, CORS, serves frontend/dist
frontend/ (Vite + React 18 + TypeScript + Recharts) ──► /api/*
```
Stack: Python 3.12, FastAPI, pydantic v2, scikit-learn 1.8, pandas, NumPy, SHAP, SQLite (stdlib); React 18, Vite 5, Recharts, React Router. Tested versions are in `backend/requirements-tested.txt`.

## API (full interactive docs at `/docs`)
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/overview` | Status, dashboard aggregates |
| GET | `/api/dataset/summary` `/schema` `/samples` | Profile, feature schema, real sample records |
| GET | `/api/models`, `/api/models/versions`, `/api/benchmark`, `/api/reliability` | Model list, versions, full benchmark evidence, reliability evidence |
| POST | `/api/models/versions/{v}/activate` | Switch active version |
| POST/GET | `/api/train`, `/api/train/status` | Background training and progress |
| POST | `/api/predict` | Class, probability, SHAP explanation, model agreement (saves to history unless `save:false`) |
| POST/GET | `/api/explain`, `/api/explain/global` | Local explanation, permutation importance + concepts |
| POST | `/api/counterfactual` | Verified counterfactual candidates |
| POST | `/api/scenarios/compare` | A/B comparison with contributions and sweeps |
| GET/DELETE | `/api/history/predictions[/{id}]`, `/api/history/experiments[/{id}]` | List (filter/sort/page), view, delete |
| GET | `/api/privacy` | What is stored |

Invalid input returns HTTP 422 with per-field messages; unknown ids 404; no trained model 409 `model_not_trained`.

## Training and tests
```powershell
.\.venv\Scripts\python.exe -m cardiosage.training        # run from backend\ ; trains and saves a new version
cd backend;  ..\.venv\Scripts\python.exe -m pytest -q     # 39 tests, trains a real model in a temp dir (~75 s)
cd frontend; npm run build                                # type-check + production build
cd frontend; npm test                                     # 16 UI tests against a RUNNING backend (start run.ps1 first); skipped if it is down
```

## Actual evaluation results
Dataset: 302 distinct records -> **241 train / 61 test** (stratified, seed 42). Cross-validation: 5-fold x 2 repeats on the training split only. Selection used CV ROC-AUC only. Test split was evaluated once. Positive class for precision/recall/F1 is `target = 1`.

| Model | CV ROC-AUC (mean ± SD) | CV accuracy | Test accuracy | Precision | Recall | F1 | Test ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| Baseline (class prior) | 0.500 | 54.4% | 54.1% | 0.541 | 1.000 | 0.702 | 0.500 | 0.248 |
| Logistic Regression (C=0.1) | 0.907 ± 0.050 | 84.4% | 83.6% | 0.848 | 0.848 | 0.848 | 0.911 | 0.124 |
| **Random Forest** (selected) | **0.912 ± 0.044** | 84.8% | 80.3% | 0.784 | 0.879 | 0.829 | 0.897 | 0.142 |
| SVM (RBF) | 0.910 ± 0.047 | 86.5% | 85.2% | 0.853 | 0.879 | 0.866 | 0.897 | 0.128 |
| Gradient Boosting | 0.901 ± 0.047 | 81.9% | 75.4% | 0.765 | 0.788 | 0.776 | 0.856 | 0.159 |

95% bootstrap intervals on the 61-record test set are wide (Random Forest accuracy 70-90%, ROC-AUC 0.81-0.96). **Random Forest won selection by 0.002 AUC against ~0.04 fold-to-fold noise, so the top three models are statistically indistinguishable;** the app states this under Methodology. Hyper-parameters were tuned on the same training folds, so CV numbers are slightly optimistic. Leakage demonstration: naive split of raw rows -> accuracy 1.000 / AUC 1.000 (98.5% of test rows duplicated in training).

## What is stored
Only in a local SQLite file (`storage/cardiosage.db`): timestamp, model version, algorithm, the 13 submitted feature values, predicted class, probability and top-5 contributions (predictions); summaries of training runs, counterfactual searches and what-if comparisons (experiments). No names, IDs, free text, IP addresses or device data; nothing leaves your computer. Delete single entries or everything from the History page, via `DELETE /api/history/predictions`, or by deleting the file. Saving can be skipped per request in the API (`save: false`).

## Verification status (what was actually run)
Built and verified in a **Linux** sandbox (Python 3.12, Node 22) with real commands:
* Dataset retrieved, hashed, profiled; schema and variant handling tested (9 tests).
* Training pipeline executed twice from scratch (identical metrics); models saved, reloaded, and reproduce the reported test metrics exactly (test).
* **39/39 backend tests pass** (data, training, leakage, saved-model reload, predictions vs. directly loaded model, SHAP additivity, 422 validation, counterfactual candidates re-evaluated against the raw saved model, scenarios, SQLite persistence/filter/delete, CORS, error codes).
* `npm run build` (tsc + Vite) passes; **16/16 UI integration tests pass** against the real backend (all 9 pages render real data; clicking Run prediction / Find counterfactuals / Compare scenarios / Delete works; invalid input is blocked).
* `./run.sh --check` from a **clean copy** (no venv, node_modules, build or model): installed, built, trained, started, and verified health + UI + prediction.

**Not verified:** `run.ps1` / `setup.ps1` were written carefully but **never executed**, because no Windows or PowerShell was available. `run.sh` mirrors their logic and was executed. If a PowerShell step fails, its message names the step; run `.\run.ps1 -Check` first. No browser was available, so **no screenshots were captured and visual layout was not inspected in a real browser**; tests ran in jsdom.
To capture screenshots: run the app, open each page, and use Win+Shift+S; save into `docs/screenshots/`.

## Limitations and future work
* 302 distinct patients; small test set; wide uncertainty; no external validation, calibration against outcomes, fairness or robustness evaluation.
* Label direction in this copy appears inverted and could not be checked against the UCI original (host unreachable); category codings for `cp`/`thal` are unverified.
* Counterfactuals are model-level, search is heuristic (not guaranteed minimal) and optimises sparsity, not clinical feasibility.
* KernelSHAP uses a 40-row background; explanations depend on it.
* scikit-learn 1.9+ deprecates `SVC(probability=True)` (warning suppressed); switch to `CalibratedClassifierCV` when upgrading.
* Future: diff against the original UCI files, external datasets, formal calibration, subgroup analysis, DiCE/actionability constraints, authentication for multi-user deployment.
