# Dataset provenance, schema and limitations

**File:** `data/heart.csv` (1,025 rows x 14 columns, SHA-256 `ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1`)

**Origin.** The 14-attribute form of the UCI *Heart Disease* data (Cleveland, Hungarian, Swiss and Long Beach VA sites).
Original citation: Janosi A., Steinbrunn W., Pfisterer M., Detrano R. (1988), UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X.
**Licence of the original:** CC BY 4.0 (as stated by the UCI repository).

**How it was obtained.** `archive.ics.uci.edu` was not reachable from the build environment (outbound access was limited to
GitHub, PyPI and npm), so the file was downloaded from a public GitHub copy:
`https://raw.githubusercontent.com/halekpetigo/BIOF509/master/heart.csv`. That repository states the file is the
UCI heart disease data but ships no licence of its own, so reuse is governed by the original UCI licence.
**The copy could not be diffed against the UCI originals**, so its fidelity rests on the schema and value checks below.

## Verified (automatically, see `backend/tests/test_data.py`)
* 13 feature columns + `target`; no missing cells; target is binary (526 x 1, 499 x 0).
* Every categorical column only contains its documented codes; every numeric column is inside physiological limits.

## Findings that change how the data must be used
| Finding | Evidence | What CardioSage does |
|---|---|---|
| **723 of 1,025 rows are exact duplicates** (only 302 distinct records) | `data.profile()` | De-duplicates before splitting. A naive split of the raw rows puts a copy of 98.5% of test rows in training and scores 100% accuracy. |
| **Label direction looks inverted** relative to the common documentation ("1 = disease") | Exercise angina, ST depression, vessel count and age are all *lower*, and max heart rate *higher*, when `target = 1` (r = -0.44, -0.43, -0.41, -0.22, +0.42). Computed live in `label_direction_check`. | Never calls classes "disease"/"healthy". The UI says `target = 0` / `target = 1` and explains the inferred direction. Must be confirmed against the original source. |
| `ca = 4` (4 unique rows) and `thal = 0` (2 rows) are undocumented values | profile | Kept as observed (treated as codes), warned about, imputation is inside the pipeline. |
| Category codes for `thal` differ between redistributions | domain knowledge | `thal` codes are shown unlabelled. Chest-pain etc. labels follow common documentation and are marked unverified. |

## Column meanings
`age` (years), `sex` (0 female, 1 male), `cp` chest-pain type 0-3, `trestbps` resting systolic BP (mmHg), `chol` serum cholesterol (mg/dl),
`fbs` fasting glucose > 120 mg/dl, `restecg` resting ECG 0-2, `thalach` max heart rate (bpm), `exang` exercise-induced angina,
`oldpeak` ST depression (mm), `slope` peak-exercise ST slope 0-2, `ca` major vessels coloured 0-4, `thal` thallium test code 0-3, `target`.

## Supported variants
Column-name aliases (e.g. `RestingBP`, `trtbps`, `MaxHR`, `thalachh`, `output`, `num`, `HeartDisease`), `?` as missing,
multi-class `num` (0-4, collapsed to >0), and `M`/`F` for sex. Fully string-coded variants (e.g. `ChestPainType = "ATA"`) are **not** supported
and fail with a clear `DatasetError` rather than being guessed. Point `CARDIOSAGE_DATA` at another file to use it.

## Limitations
Historical referral patients from four hospitals; 302 distinct people; no demographic or temporal metadata, so subgroup fairness and drift
cannot be assessed. Results do not generalise to any other population.
