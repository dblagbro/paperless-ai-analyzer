"""
Database layer for user auth, chat sessions, and share management.
SQLite at /app/data/app.db — separate from state.json / vector store.
"""

import sqlite3
import uuid
import logging
from datetime import datetime
from pathlib import Path
from werkzeug.security import generate_password_hash

logger = logging.getLogger(__name__)

DB_PATH = '/app/data/app.db'


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if not exist (idempotent, safe to call on every startup)."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                username       TEXT    NOT NULL UNIQUE,
                password_hash  TEXT    NOT NULL,
                display_name   TEXT,
                role           TEXT    NOT NULL DEFAULT 'basic',
                created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
                last_login     TEXT,
                is_active      INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id            TEXT PRIMARY KEY,
                user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title         TEXT    NOT NULL DEFAULT 'New Chat',
                document_type TEXT,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role       TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chat_shares (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id           TEXT    NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                shared_with_user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                shared_by_user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(session_id, shared_with_user_id)
            );
        """)
    logger.info("Database initialized (tables created if not exist)")


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def get_user_by_username(username: str):
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
        ).fetchone()


def get_user_by_id(user_id: int):
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()


def create_user(username: str, password: str, role: str = 'basic', display_name: str = None):
    pw_hash = generate_password_hash(password)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, display_name) VALUES (?, ?, ?, ?)",
            (username, pw_hash, role, display_name or username)
        )
    logger.info(f"Created user '{username}' with role '{role}'")


def update_last_login(user_id: int):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?", (user_id,)
        )


def update_user(user_id: int, **kwargs):
    """Update role, display_name, or password for a user."""
    allowed = {'role', 'display_name', 'password', 'is_active'}
    updates = {}
    for k, v in kwargs.items():
        if k not in allowed:
            continue
        if k == 'password':
            updates['password_hash'] = generate_password_hash(v)
        else:
            updates[k] = v
    if not updates:
        return
    set_clause = ', '.join(f"{col} = ?" for col in updates)
    values = list(updates.values()) + [user_id]
    with _get_conn() as conn:
        conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)


def list_users():
    with _get_conn() as conn:
        return conn.execute(
            "SELECT id, username, display_name, role, created_at, last_login, is_active FROM users ORDER BY id"
        ).fetchall()


# ---------------------------------------------------------------------------
# Chat sessions CRUD
# ---------------------------------------------------------------------------

def get_sessions(user_id: int):
    """Return own sessions + sessions shared with this user, newest first."""
    with _get_conn() as conn:
        return conn.execute("""
            SELECT cs.*, u.username AS owner_username,
                   CASE WHEN cs.user_id = ? THEN 0 ELSE 1 END AS is_shared
            FROM chat_sessions cs
            JOIN users u ON u.id = cs.user_id
            WHERE cs.user_id = ?
               OR cs.id IN (
                   SELECT session_id FROM chat_shares WHERE shared_with_user_id = ?
               )
            ORDER BY cs.updated_at DESC
        """, (user_id, user_id, user_id)).fetchall()


def get_all_sessions_by_user():
    """Admin view: all sessions grouped by user (returns flat list with owner info)."""
    with _get_conn() as conn:
        return conn.execute("""
            SELECT cs.*, u.username AS owner_username, u.display_name AS owner_display_name
            FROM chat_sessions cs
            JOIN users u ON u.id = cs.user_id
            ORDER BY u.username, cs.updated_at DESC
        """).fetchall()


def create_session(user_id: int, title: str = 'New Chat', document_type: str = None) -> str:
    session_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, title, document_type) VALUES (?, ?, ?, ?)",
            (session_id, user_id, title, document_type)
        )
    return session_id


def get_session(session_id: str):
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()


def get_messages(session_id: str):
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at, id",
            (session_id,)
        ).fetchall()


def append_message(session_id: str, role: str, content: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,)
        )


def update_session_title(session_id: str, title: str):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (title, session_id)
        )


def delete_session(session_id: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


# ---------------------------------------------------------------------------
# Chat sharing
# ---------------------------------------------------------------------------

def share_session(session_id: str, shared_with_user_id: int, shared_by_user_id: int):
    with _get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO chat_shares (session_id, shared_with_user_id, shared_by_user_id)
            VALUES (?, ?, ?)
        """, (session_id, shared_with_user_id, shared_by_user_id))


def unshare_session(session_id: str, shared_with_user_id: int):
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM chat_shares WHERE session_id = ? AND shared_with_user_id = ?",
            (session_id, shared_with_user_id)
        )


def get_session_shares(session_id: str):
    """Return list of (user_id, username) tuples this session is shared with."""
    with _get_conn() as conn:
        return conn.execute("""
            SELECT u.id, u.username, u.display_name
            FROM chat_shares cs
            JOIN users u ON u.id = cs.shared_with_user_id
            WHERE cs.session_id = ?
        """, (session_id,)).fetchall()


def can_access_session(session_id: str, user_id: int) -> bool:
    """Check if user owns or has been shared this session."""
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT 1 FROM chat_sessions
            WHERE id = ? AND (
                user_id = ?
                OR id IN (SELECT session_id FROM chat_shares WHERE shared_with_user_id = ?)
            )
        """, (session_id, user_id, user_id)).fetchone()
        return row is not None
