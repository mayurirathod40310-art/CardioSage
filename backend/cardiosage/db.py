"""SQLite persistence for predictions, experiments and model registry.

Stored per prediction: timestamp, model version/algorithm, the 13 clinical feature values that were submitted,
predicted class, probability, top contributing features. No names, IDs, free text or IP addresses are stored.
"""
from __future__ import annotations
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from . import config

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  model_version TEXT NOT NULL,
  algorithm TEXT NOT NULL,
  inputs_json TEXT NOT NULL,
  predicted_class INTEGER NOT NULL,
  probability_class1 REAL NOT NULL,
  top_features_json TEXT,
  source TEXT NOT NULL DEFAULT 'prediction'
);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
CREATE TABLE IF NOT EXISTS experiments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL,
  model_version TEXT,
  title TEXT NOT NULL,
  summary_json TEXT NOT NULL,
  config_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    try:
        with _lock:
            yield c
            c.commit()
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.executescript(SCHEMA)
        for table in ("predictions", "experiments"):
            columns = {row["name"] for row in c.execute(f"PRAGMA table_info({table})")}
            if "user_id" not in columns:
                c.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER REFERENCES users(id)")


def _pred_row(r: sqlite3.Row) -> dict[str, Any]:
    return {"id": r["id"], "created_at": r["created_at"], "model_version": r["model_version"], "algorithm": r["algorithm"],
            "inputs": json.loads(r["inputs_json"]), "predicted_class": r["predicted_class"],
            "probability_class1": r["probability_class1"],
            "top_features": json.loads(r["top_features_json"]) if r["top_features_json"] else [],
            "source": r["source"]}


def add_prediction(model_version: str, algorithm: str, inputs: dict, predicted_class: int, p1: float,
                   top_features: list | None, source: str = "prediction", user_id: int | None = None) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (created_at, model_version, algorithm, inputs_json, predicted_class, probability_class1, top_features_json, source, user_id) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (now(), model_version, algorithm, json.dumps(inputs), int(predicted_class), float(p1),
             json.dumps(top_features or []), source, user_id))
        return int(cur.lastrowid)


_SORTS = {"created_at": "created_at", "probability": "probability_class1", "id": "id", "algorithm": "algorithm"}


def list_predictions(limit: int = 50, offset: int = 0, predicted_class: int | None = None, algorithm: str | None = None,
                     model_version: str | None = None, source: str | None = None, min_p: float | None = None,
                     max_p: float | None = None, sort: str = "created_at", order: str = "desc",
                     user_id: int | None = None) -> dict[str, Any]:
    where, args = [], []
    if user_id is not None: where.append("user_id=?"); args.append(user_id)
    if predicted_class is not None: where.append("predicted_class=?"); args.append(predicted_class)
    if algorithm: where.append("algorithm=?"); args.append(algorithm)
    if model_version: where.append("model_version=?"); args.append(model_version)
    if source: where.append("source=?"); args.append(source)
    if min_p is not None: where.append("probability_class1>=?"); args.append(min_p)
    if max_p is not None: where.append("probability_class1<=?"); args.append(max_p)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    col = _SORTS.get(sort, "created_at")
    direction = "ASC" if order.lower() == "asc" else "DESC"
    with conn() as c:
        total = c.execute(f"SELECT COUNT(*) FROM predictions {w}", args).fetchone()[0]
        rows = c.execute(f"SELECT * FROM predictions {w} ORDER BY {col} {direction}, id DESC LIMIT ? OFFSET ?",
                         [*args, limit, offset]).fetchall()
    return {"total": total, "items": [_pred_row(r) for r in rows]}


def get_prediction(pid: int, user_id: int | None = None) -> dict[str, Any] | None:
    with conn() as c:
        if user_id is None:
            r = c.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
        else:
            r = c.execute("SELECT * FROM predictions WHERE id=? AND user_id=?", (pid, user_id)).fetchone()
    return _pred_row(r) if r else None


def delete_prediction(pid: int, user_id: int | None = None) -> bool:
    with conn() as c:
        if user_id is None:
            cur = c.execute("DELETE FROM predictions WHERE id=?", (pid,))
        else:
            cur = c.execute("DELETE FROM predictions WHERE id=? AND user_id=?", (pid, user_id))
        return cur.rowcount > 0


def delete_all_predictions(user_id: int | None = None) -> int:
    with conn() as c:
        if user_id is None:
            cur = c.execute("DELETE FROM predictions")
        else:
            cur = c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
        return cur.rowcount


def prediction_stats(user_id: int | None = None) -> dict[str, Any]:
    with conn() as c:
        if user_id is None:
            total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            by = c.execute("SELECT predicted_class, COUNT(*) n FROM predictions GROUP BY predicted_class").fetchall()
        else:
            total = c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (user_id,)).fetchone()[0]
            by = c.execute("SELECT predicted_class, COUNT(*) n FROM predictions WHERE user_id=? GROUP BY predicted_class", (user_id,)).fetchall()
    return {"total": total, "by_class": {int(r["predicted_class"]): r["n"] for r in by}}


def add_experiment(kind: str, title: str, summary: dict, model_version: str | None = None,
                   config_: dict | None = None, user_id: int | None = None) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO experiments (created_at, kind, model_version, title, summary_json, config_json, user_id) VALUES (?,?,?,?,?,?,?)",
                        (now(), kind, model_version, title, json.dumps(summary), json.dumps(config_ or {}), user_id))
        return int(cur.lastrowid)


def _exp_row(r: sqlite3.Row) -> dict[str, Any]:
    return {"id": r["id"], "created_at": r["created_at"], "kind": r["kind"], "model_version": r["model_version"],
            "title": r["title"], "summary": json.loads(r["summary_json"]), "config": json.loads(r["config_json"] or "{}")}


def list_experiments(limit: int = 50, offset: int = 0, kind: str | None = None, order: str = "desc",
                     user_id: int | None = None) -> dict[str, Any]:
    conditions, args = [], []
    if kind: conditions.append("kind=?"); args.append(kind)
    if user_id is not None: conditions.append("user_id=?"); args.append(user_id)
    w = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    direction = "ASC" if order.lower() == "asc" else "DESC"
    with conn() as c:
        total = c.execute(f"SELECT COUNT(*) FROM experiments {w}", args).fetchone()[0]
        rows = c.execute(f"SELECT * FROM experiments {w} ORDER BY created_at {direction}, id {direction} LIMIT ? OFFSET ?",
                         [*args, limit, offset]).fetchall()
    return {"total": total, "items": [_exp_row(r) for r in rows]}


def get_experiment(eid: int, user_id: int | None = None) -> dict[str, Any] | None:
    with conn() as c:
        if user_id is None:
            r = c.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
        else:
            r = c.execute("SELECT * FROM experiments WHERE id=? AND user_id=?", (eid, user_id)).fetchone()
    return _exp_row(r) if r else None


def delete_experiment(eid: int, user_id: int | None = None) -> bool:
    with conn() as c:
        if user_id is None:
            cur = c.execute("DELETE FROM experiments WHERE id=?", (eid,))
        else:
            cur = c.execute("DELETE FROM experiments WHERE id=? AND user_id=?", (eid, user_id))
        return cur.rowcount > 0
