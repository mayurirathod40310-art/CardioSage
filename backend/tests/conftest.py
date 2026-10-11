import os
import tempfile
from pathlib import Path

# Isolate all writable state BEFORE the package is imported (config reads the environment at import time).
_TMP = Path(tempfile.mkdtemp(prefix="cardiosage-test-"))
os.environ["CARDIOSAGE_MODELS"] = str(_TMP / "models")
os.environ["CARDIOSAGE_DB"] = str(_TMP / "test.db")

import pytest  # noqa: E402

EXAMPLE = {"age": 57, "sex": 1, "cp": 0, "trestbps": 140, "chol": 241, "fbs": 0, "restecg": 1, "thalach": 123,
           "exang": 1, "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3}


@pytest.fixture(scope="session")
def trained():
    """Run the REAL training pipeline once and return its metadata."""
    from cardiosage import db, store, training
    db.init_db()
    meta = training.train_all()
    store.clear_cache()
    return meta


@pytest.fixture(scope="session")
def client(trained):
    from fastapi.testclient import TestClient
    from cardiosage.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def example():
    return dict(EXAMPLE)
