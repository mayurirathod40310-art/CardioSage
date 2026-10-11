"""Account and server-side session management for CardioSage."""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pwdlib import PasswordHash

from . import config

_password_hash = PasswordHash.recommended()
SESSION_COOKIE = "cardiosage_session"
SESSION_DAYS = 7


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def init_auth_db() -> None:
    """Create authentication tables without changing existing CardioSage tables."""
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK (role IN ('user', 'admin')),
                created_at TEXT NOT NULL,
                disabled INTEGER NOT NULL DEFAULT 0
                    CHECK (disabled IN (0, 1))
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_sessions_user "
            "ON auth_sessions(user_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry "
            "ON auth_sessions(expires_at)"
        )


def create_user(email: str, password: str) -> dict[str, Any]:
    """Create a normal account. Public registration never grants admin access."""
    email = email.strip().lower()
    if len(email) > 254 or email.count("@") != 1:
        raise ValueError("Enter a valid email address.")
    local, domain = email.split("@")
    if (
        not local
        or not domain
        or "." not in domain
        or any(ch.isspace() for ch in email)
    ):
        raise ValueError("Enter a valid email address.")
    if len(password) < 12 or len(password) > 128:
        raise ValueError("Password must contain 12 to 128 characters.")

    encoded = _password_hash.hash(password)
    try:
        with _connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users (email, password_hash, role, created_at) "
                "VALUES (?, ?, 'user', ?)",
                (email, encoded, _stamp(_now())),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        if "users.email" in str(exc).lower() or "unique" in str(exc).lower():
            raise ValueError("An account with this email already exists.") from exc
        raise
    return {"id": user_id, "email": email, "role": "user"}


def authenticate(email: str, password: str) -> dict[str, Any] | None:
    """Verify credentials without revealing whether an email exists."""
    email = email.strip().lower()
    with _connect() as connection:
        row = connection.execute(
            "SELECT id, email, password_hash, role, disabled "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if row is None:
        # Perform a password hash verification to reduce timing differences.
        _password_hash.hash("not-a-real-user-password")
        return None
    if row["disabled"] or not _password_hash.verify(password, row["password_hash"]):
        return None

    return {"id": row["id"], "email": row["email"], "role": row["role"]}


def create_session(user_id: int) -> tuple[str, str]:
    """Return a one-time raw cookie token and its expiry timestamp."""
    token = secrets.token_urlsafe(32)
    now = _now()
    expires = now + timedelta(days=SESSION_DAYS)
    with _connect() as connection:
        connection.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ?",
            (_stamp(now),),
        )
        connection.execute(
            "INSERT INTO auth_sessions (user_id, token_hash, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, _token_hash(token), _stamp(now), _stamp(expires)),
        )
    return token, _stamp(expires)


def get_session_user(token: str | None) -> dict[str, Any] | None:
    if not token or len(token) > 256:
        return None
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT u.id, u.email, u.role
            FROM auth_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ? AND u.disabled = 0
            """,
            (_token_hash(token), _stamp(_now())),
        ).fetchone()
    return dict(row) if row else None


def revoke_session(token: str | None) -> bool:
    if not token or len(token) > 256:
        return False
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM auth_sessions WHERE token_hash = ?",
            (_token_hash(token),),
        )
        return cursor.rowcount > 0


def revoke_all_sessions(user_id: int) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM auth_sessions WHERE user_id = ?", (user_id,)
        )
        return cursor.rowcount


def bootstrap_admin_from_env() -> dict[str, Any] | None:
    """Create/promote the configured administrator without a public setup route.

    Set CARDIOSAGE_ADMIN_EMAIL and CARDIOSAGE_ADMIN_PASSWORD in the hosting
    environment. Existing accounts are promoted only when the password matches.
    """
    email = os.environ.get("CARDIOSAGE_ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("CARDIOSAGE_ADMIN_PASSWORD", "")

    if not email and not password:
        return None
    if not email or not password:
        raise RuntimeError(
            "Set both CARDIOSAGE_ADMIN_EMAIL and CARDIOSAGE_ADMIN_PASSWORD, "
            "or leave both unset."
        )
    if len(password) < 12 or len(password) > 128:
        raise RuntimeError(
            "CARDIOSAGE_ADMIN_PASSWORD must contain 12 to 128 characters."
        )

    with _connect() as connection:
        row = connection.execute(
            "SELECT id, email, password_hash, disabled FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if row is None:
        user = create_user(email, password)
        user_id = int(user["id"])
    else:
        if row["disabled"]:
            raise RuntimeError(
                "The configured administrator account is disabled; enable it "
                "manually before bootstrapping."
            )
        if not _password_hash.verify(password, row["password_hash"]):
            raise RuntimeError(
                "Admin bootstrap password did not match the existing account. "
                "No role changes were made."
            )
        user_id = int(row["id"])

    with _connect() as connection:
        connection.execute(
            "UPDATE users SET role = 'admin' WHERE id = ?", (user_id,)
        )

    return {"id": user_id, "email": email, "role": "admin"}
