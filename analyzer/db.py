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
                project_slug  TEXT    NOT NULL DEFAULT 'default',
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

            CREATE TABLE IF NOT EXISTS import_history (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER REFERENCES users(id),
                source            TEXT,
                filename          TEXT,
                original_url      TEXT,
                paperless_doc_id  INTEGER,
                status            TEXT NOT NULL DEFAULT 'uploaded',
                error_msg         TEXT,
                created_at        TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS processed_documents (
                doc_id            INTEGER PRIMARY KEY,
                project_slug      TEXT NOT NULL DEFAULT 'default',
                first_analyzed_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_analyzed_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)

        # Migrations — ADD COLUMN is idempotent-safe (ignored if column exists)
        for _tbl, _col in (
            ("chat_sessions",       "project_slug TEXT NOT NULL DEFAULT 'default'"),
            ("processed_documents", "project_slug TEXT NOT NULL DEFAULT 'default'"),
        ):
            try:
                conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN {_col}")
                logger.info(f"Migration: added {_tbl}.project_slug column")
            except Exception:
                pass  # column already exists

        try:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT ''")
            logger.info("Migration: added users.email column")
        except Exception:
            pass  # column already exists
        for col in (
            "phone TEXT NOT NULL DEFAULT ''",
            "address TEXT NOT NULL DEFAULT ''",
            "github TEXT NOT NULL DEFAULT ''",
            "linkedin TEXT NOT NULL DEFAULT ''",
            "facebook TEXT NOT NULL DEFAULT ''",
            "instagram TEXT NOT NULL DEFAULT ''",
            "other_handles TEXT NOT NULL DEFAULT ''",
            "timezone TEXT NOT NULL DEFAULT ''",
            "job_title TEXT NOT NULL DEFAULT ''",
        ):
            col_name = col.split()[0]
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col}")
                logger.info(f"Migration: added users.{col_name} column")
            except Exception:
                pass  # column already exists

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


def create_user(username: str, password: str, role: str = 'basic', display_name: str = None, email: str = ''):
    pw_hash = generate_password_hash(password)
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, display_name, email) VALUES (?, ?, ?, ?, ?)",
            (username, pw_hash, role, display_name or username, email or '')
        )
    logger.info(f"Created user '{username}' with role '{role}'")


def update_last_login(user_id: int):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?", (user_id,)
        )


def update_user(user_id: int, **kwargs):
    """Update role, display_name, email, password, is_active, or profile fields for a user."""
    allowed = {
        'role', 'display_name', 'email', 'password', 'is_active',
        'phone', 'address', 'github', 'linkedin', 'facebook',
        'instagram', 'other_handles', 'timezone', 'job_title',
    }
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
            "SELECT id, username, display_name, email, role, created_at, last_login, is_active,"
            " phone, address, github, linkedin, facebook, instagram, other_handles, timezone, job_title"
            " FROM users ORDER BY id"
        ).fetchall()


# ---------------------------------------------------------------------------
# Chat sessions CRUD
# ---------------------------------------------------------------------------

def get_sessions(user_id: int, project_slug: str = 'default'):
    """Return own sessions + sessions shared with this user for the given project, newest first."""
    with _get_conn() as conn:
        return conn.execute("""
            SELECT cs.*, u.username AS owner_username,
                   CASE WHEN cs.user_id = ? THEN 0 ELSE 1 END AS is_shared
            FROM chat_sessions cs
            JOIN users u ON u.id = cs.user_id
            WHERE cs.project_slug = ?
              AND (
                   cs.user_id = ?
                   OR cs.id IN (
                       SELECT session_id FROM chat_shares WHERE shared_with_user_id = ?
                   )
              )
            ORDER BY cs.updated_at DESC
        """, (user_id, project_slug, user_id, user_id)).fetchall()


def get_all_sessions_by_user():
    """Admin view: all sessions grouped by user (returns flat list with owner info)."""
    with _get_conn() as conn:
        return conn.execute("""
            SELECT cs.*, u.username AS owner_username, u.display_name AS owner_display_name
            FROM chat_sessions cs
            JOIN users u ON u.id = cs.user_id
            ORDER BY u.username, cs.updated_at DESC
        """).fetchall()


def create_session(user_id: int, title: str = 'New Chat', document_type: str = None,
                   project_slug: str = 'default') -> str:
    session_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, title, document_type, project_slug) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, title, document_type, project_slug)
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


def append_message(session_id: str, role: str, content: str) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = datetime('now') WHERE id = ?",
            (session_id,)
        )
        return cur.lastrowid


def update_message_content(message_id: int, session_id: str, new_content: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chat_messages SET content = ? WHERE id = ? AND session_id = ?",
            (new_content, message_id, session_id)
        )


def delete_messages_from(session_id: str, from_message_id: int) -> None:
    """Delete this message and all subsequent messages in the session."""
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE session_id = ? AND id >= ?",
            (session_id, from_message_id)
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


# ---------------------------------------------------------------------------
# Import history
# ---------------------------------------------------------------------------

def log_import(user_id: int, source: str, filename: str, url: str = None,
               doc_id: int = None, status: str = 'uploaded', error: str = None):
    """Record an import attempt in import_history."""
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO import_history
               (user_id, source, filename, original_url, paperless_doc_id, status, error_msg)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, source, filename, url, doc_id, status, error)
        )


def mark_document_processed(doc_id: int, project_slug: str = 'default') -> None:
    """Record that the AI analyzer has processed a document.  Uses INSERT OR REPLACE so the
    last_analyzed_at timestamp is updated on re-analysis while first_analyzed_at is preserved."""
    if not doc_id:
        return
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT first_analyzed_at FROM processed_documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE processed_documents SET last_analyzed_at = datetime('now'), project_slug = ? WHERE doc_id = ?",
                (project_slug, doc_id)
            )
        else:
            conn.execute(
                "INSERT INTO processed_documents (doc_id, project_slug) VALUES (?, ?)",
                (doc_id, project_slug)
            )


def count_processed_documents(project_slug: str = None) -> int:
    """Return the total number of unique documents ever processed by the AI analyzer.
    If project_slug is given, counts only documents for that project."""
    try:
        with _get_conn() as conn:
            if project_slug:
                row = conn.execute(
                    "SELECT COUNT(*) FROM processed_documents WHERE project_slug = ?", (project_slug,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM processed_documents").fetchone()
            return row[0] if row else 0
    except Exception as e:
        logger.error(f"Failed to count processed documents: {e}")
        return 0


def get_analyzed_doc_ids(project_slug: str = None) -> set:
    """Return the set of doc_ids in the processed_documents table.
    If project_slug is given, returns only IDs for that project."""
    try:
        with _get_conn() as conn:
            if project_slug:
                rows = conn.execute(
                    "SELECT doc_id FROM processed_documents WHERE project_slug = ?", (project_slug,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT doc_id FROM processed_documents").fetchall()
            return {r[0] for r in rows}
    except Exception as e:
        logger.error(f"Failed to get analyzed doc IDs: {e}")
        return set()


def get_import_history(user_id: int, limit: int = 20):
    """Return the most recent imports for a user, newest first."""
    with _get_conn() as conn:
        return conn.execute(
            """SELECT id, source, filename, original_url, paperless_doc_id, status, error_msg, created_at
               FROM import_history
               WHERE user_id = ?
               ORDER BY created_at DESC, id DESC
               LIMIT ?""",
            (user_id, limit)
        ).fetchall()
