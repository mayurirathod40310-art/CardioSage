"""Feature schema for the 14-attribute heart disease data.

`hard` limits are physiological sanity limits used to reject impossible input.
Observed ranges (min/max in the data) are computed separately and used for
extrapolation warnings and counterfactual search bounds.
"""
from __future__ import annotations
from typing import Any

TARGET = "target"
CLASS_LABELS = {0: "target = 0", 1: "target = 1"}

# name, label, kind, unit, description, options, hard (min,max), step, locked_by_default
_RAW: list[dict[str, Any]] = [
    dict(name="age", label="Age", kind="numeric", unit="years", hard=(18, 100), step=1, locked=True,
         description="Age at examination."),
    dict(name="sex", label="Sex", kind="binary", options={0: "Female", 1: "Male"}, locked=True,
         description="Recorded sex."),
    dict(name="cp", label="Chest pain type", kind="categorical",
         options={0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
         description="Chest pain category (labels follow common documentation; coding in this copy is unverified)."),
    dict(name="trestbps", label="Resting blood pressure", kind="numeric", unit="mmHg", hard=(70, 250), step=1,
         description="Resting systolic blood pressure on admission."),
    dict(name="chol", label="Serum cholesterol", kind="numeric", unit="mg/dl", hard=(100, 700), step=1,
         description="Serum cholesterol."),
    dict(name="fbs", label="Fasting blood sugar > 120 mg/dl", kind="binary", options={0: "No", 1: "Yes"},
         description="Whether fasting blood sugar exceeded 120 mg/dl."),
    dict(name="restecg", label="Resting ECG", kind="categorical",
         options={0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
         description="Resting electrocardiographic result."),
    dict(name="thalach", label="Maximum heart rate", kind="numeric", unit="bpm", hard=(50, 230), step=1,
         description="Maximum heart rate achieved during the exercise test."),
    dict(name="exang", label="Exercise-induced angina", kind="binary", options={0: "No", 1: "Yes"},
         description="Angina provoked by exercise."),
    dict(name="oldpeak", label="ST depression (exercise vs rest)", kind="numeric", unit="mm", hard=(0, 10), step=0.1,
         description="ST depression induced by exercise relative to rest."),
    dict(name="slope", label="Peak exercise ST slope", kind="categorical",
         options={0: "Upsloping", 1: "Flat", 2: "Downsloping"},
         description="Slope of the peak exercise ST segment."),
    dict(name="ca", label="Major vessels coloured (fluoroscopy)", kind="ordinal",
         options={0: "0", 1: "1", 2: "2", 3: "3", 4: "4 (likely missing-value code)"},
         description="Count of major vessels coloured by fluoroscopy. Documented range is 0-3; value 4 appears to encode missing."),
    dict(name="thal", label="Thalassemia test result", kind="categorical",
         options={0: "0 (likely missing-value code)", 1: "Code 1", 2: "Code 2", 3: "Code 3"},
         description="Thallium stress test result. Coding differs between redistributions of this dataset, so codes are shown unlabelled."),
]

FEATURES: list[dict[str, Any]] = []
for f in _RAW:
    f = dict(f)
    f.setdefault("unit", None)
    f.setdefault("options", None)
    f.setdefault("hard", None)
    f.setdefault("step", None)
    f.setdefault("locked", False)
    FEATURES.append(f)

FEATURE_NAMES = [f["name"] for f in FEATURES]
FEATURE_BY_NAME = {f["name"]: f for f in FEATURES}
NUMERIC = [f["name"] for f in FEATURES if f["kind"] == "numeric"]
BINARY = [f["name"] for f in FEATURES if f["kind"] == "binary"]
CATEGORICAL = [f["name"] for f in FEATURES if f["kind"] == "categorical"]
ORDINAL = [f["name"] for f in FEATURES if f["kind"] == "ordinal"]
DISCRETE = BINARY + CATEGORICAL + ORDINAL

# Column-name variants seen in other redistributions of the data.
FEATURE_ALIASES = {
    "age": ["age"], "sex": ["sex", "gender"],
    "cp": ["cp", "chestpain", "chest_pain", "chestpaintype", "chest_pain_type"],
    "trestbps": ["trestbps", "restingbp", "resting_bp", "resting_blood_pressure", "trtbps", "restbp"],
    "chol": ["chol", "cholesterol", "serum_cholesterol"],
    "fbs": ["fbs", "fastingbs", "fasting_blood_sugar"],
    "restecg": ["restecg", "restingecg", "resting_ecg", "rest_ecg"],
    "thalach": ["thalach", "thalch", "thalachh", "maxhr", "max_hr", "max_heart_rate", "thalac"],
    "exang": ["exang", "exerciseangina", "exercise_angina", "exng", "exercise_induced_angina"],
    "oldpeak": ["oldpeak", "st_depression", "stdepression"],
    "slope": ["slope", "st_slope", "stslope", "slp"],
    "ca": ["ca", "caa", "num_major_vessels", "major_vessels"],
    "thal": ["thal", "thall", "thalassemia"],
}
TARGET_ALIASES = ["target", "output", "num", "goal", "condition", "heartdisease", "heart_disease",
                  "hd", "disease", "label", "class", "diagnosis", "angiographic_disease_status"]
