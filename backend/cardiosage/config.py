"""Central configuration. Every path can be overridden through environment variables
so tests can run against temporary directories without touching user data."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("CARDIOSAGE_DATA", ROOT / "data" / "heart.csv"))
MODELS_DIR = Path(os.environ.get("CARDIOSAGE_MODELS", ROOT / "models"))
DB_PATH = Path(os.environ.get("CARDIOSAGE_DB", ROOT / "storage" / "cardiosage.db"))
FRONTEND_DIST = Path(os.environ.get("CARDIOSAGE_FRONTEND", ROOT / "frontend" / "dist"))

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 2
DECISION_THRESHOLD = 0.5

DATASET_SOURCE = {
    "name": "Heart Disease Data Set (14-attribute processed form of the UCI Cleveland/Hungarian/Swiss/Long Beach data)",
    "original_citation": "Janosi A., Steinbrunn W., Pfisterer M., Detrano R. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X",
    "original_license": "CC BY 4.0 (as stated by the UCI Machine Learning Repository)",
    "retrieved_from": "https://raw.githubusercontent.com/halekpetigo/BIOF509/master/heart.csv",
    "retrieval_note": (
        "The UCI host (archive.ics.uci.edu) was not reachable from the build environment, so a public GitHub copy of the "
        "widely redistributed 14-column heart.csv was used. That redistribution carries no licence file of its own; reuse is "
        "governed by the original UCI licence. The copy could not be diffed against the UCI originals."
    ),
}

CORS_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "http://localhost:8000", "http://127.0.0.1:8000",
]
