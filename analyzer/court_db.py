"""
Court Document Importer — Database helpers.

All DB operations for court credentials, import jobs, and imported-doc tracking.
Tables live in projects.db (shared with project_manager).

Gated by COURT_IMPORT_ENABLED=true.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path('/app/data/projects.db')


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_court_db() -> None:
    """
    Create court import tables if they don't exist.
    Idempotent — safe to call on every startup.
    """
    conn = _get_conn()
    try:
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS court_credentials (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                project_slug        TEXT NOT NULL,
                court_system        TEXT NOT NULL CHECK(court_system IN ('federal','nyscef')),
                username            TEXT,
                password_encrypted  BLOB,
                extra_config_json   TEXT DEFAULT '{}',
                last_tested_at      TEXT,
                last_test_success   INTEGER DEFAULT 0,
                created_at          TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(project_slug, court_system)
            );

            CREATE TABLE IF NOT EXISTS court_import_jobs (
                id              TEXT PRIMARY KEY,
                project_slug    TEXT NOT NULL,
                user_id         INTEGER NOT NULL,
                court_system    TEXT NOT NULL,
                case_number     TEXT NOT NULL,
                case_title      TEXT,
                status          TEXT NOT NULL DEFAULT 'queued'
                                CHECK(status IN ('queued','running','completed','failed','cancelled')),
                total_docs      INTEGER DEFAULT 0,
                imported_docs   INTEGER DEFAULT 0,
                skipped_docs    INTEGER DEFAULT 0,
                failed_docs     INTEGER DEFAULT 0,
                error_message   TEXT,
                job_log_json    TEXT DEFAULT '[]',
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                started_at      TEXT,
                completed_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS court_imported_docs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id           TEXT NOT NULL REFERENCES court_import_jobs(id) ON DELETE CASCADE,
                project_slug     TEXT NOT NULL,
                court_system     TEXT NOT NULL,
                case_number      TEXT NOT NULL,
                doc_sequence     TEXT,
                source_url       TEXT,
                sha256_hash      TEXT,
                filename         TEXT,
                paperless_doc_id INTEGER,
                status           TEXT NOT NULL DEFAULT 'imported'
                                 CHECK(status IN ('imported','skipped','failed')),
                skip_reason      TEXT,
                error_msg        TEXT,
                imported_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_cid_hash ON court_imported_docs(sha256_hash);
            CREATE INDEX IF NOT EXISTS idx_cid_url  ON court_imported_docs(source_url);
            CREATE INDEX IF NOT EXISTS idx_cid_proj ON court_imported_docs(project_slug, case_number);
            CREATE INDEX IF NOT EXISTS idx_cij_proj ON court_import_jobs(project_slug, created_at DESC);
        """)
        conn.commit()
        logger.info("Court import DB schema ready")
    finally:
        conn.close()


# ── Credential helpers ──────────────────────────────────────────────────────

def save_credentials(project_slug: str, court_system: str,
                     username: str, password_encrypted: bytes,
                     extra_config: dict | None = None) -> None:
    """Upsert court credentials for a project."""
    conn = _get_conn()
    try:
        extra_json = json.dumps(extra_config or {})
        conn.execute("""
            INSERT INTO court_credentials
                (project_slug, court_system, username, password_encrypted,
                 extra_config_json, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(project_slug, court_system) DO UPDATE SET
                username           = excluded.username,
                password_encrypted = excluded.password_encrypted,
                extra_config_json  = excluded.extra_config_json,
                updated_at         = excluded.updated_at
        """, (project_slug, court_system, username, password_encrypted, extra_json))
        conn.commit()
    finally:
        conn.close()


def load_credentials(project_slug: str, court_system: str) -> Optional[Dict[str, Any]]:
    """Return credential row or None."""
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT * FROM court_credentials
            WHERE project_slug = ? AND court_system = ?
        """, (project_slug, court_system)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_credentials(project_slug: str) -> List[Dict[str, Any]]:
    """Return all credential rows for a project (no passwords)."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT court_system, username, last_tested_at, last_test_success,
                   extra_config_json, created_at, updated_at
            FROM court_credentials
            WHERE project_slug = ?
            ORDER BY court_system
        """, (project_slug,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_credential_test(project_slug: str, court_system: str, success: bool) -> None:
    conn = _get_conn()
    try:
        conn.execute("""
            UPDATE court_credentials
            SET last_tested_at = datetime('now'),
                last_test_success = ?,
                updated_at = datetime('now')
            WHERE project_slug = ? AND court_system = ?
        """, (1 if success else 0, project_slug, court_system))
        conn.commit()
    finally:
        conn.close()


def delete_credentials(project_slug: str, court_system: str) -> bool:
    conn = _get_conn()
    try:
        cur = conn.execute("""
            DELETE FROM court_credentials
            WHERE project_slug = ? AND court_system = ?
        """, (project_slug, court_system))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Import job helpers ───────────────────────────────────────────────────────

def create_import_job(job_id: str, project_slug: str, user_id: int,
                      court_system: str, case_number: str,
                      case_title: str = '') -> None:
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO court_import_jobs
                (id, project_slug, user_id, court_system, case_number, case_title)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, project_slug, user_id, court_system, case_number, case_title or ''))
        conn.commit()
    finally:
        conn.close()


def update_import_job(job_id: str, **fields) -> None:
    """Update arbitrary fields on an import job. Also accepts log_append=[str,…]."""
    if not fields:
        return

    log_lines = fields.pop('log_append', None)

    conn = _get_conn()
    try:
        if log_lines:
            row = conn.execute(
                "SELECT job_log_json FROM court_import_jobs WHERE id = ?",
                (job_id,)
            ).fetchone()
            if row:
                existing = json.loads(row['job_log_json'] or '[]')
                existing.extend(log_lines)
                # Cap at 200 log lines
                if len(existing) > 200:
                    existing = existing[-200:]
                fields['job_log_json'] = json.dumps(existing)

        if fields:
            set_clause = ', '.join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [job_id]
            conn.execute(
                f"UPDATE court_import_jobs SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()
    finally:
        conn.close()


def get_import_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM court_import_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_import_history(project_slug: str, limit: int = 20) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT id, court_system, case_number, case_title, status,
                   total_docs, imported_docs, skipped_docs, failed_docs,
                   created_at, started_at, completed_at, error_message
            FROM court_import_jobs
            WHERE project_slug = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (project_slug, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Imported doc helpers ─────────────────────────────────────────────────────

def log_court_doc(job_id: str, project_slug: str, court_system: str,
                  case_number: str, status: str,
                  doc_sequence: str = '', source_url: str = '',
                  sha256_hash: str = '', filename: str = '',
                  paperless_doc_id: int | None = None,
                  skip_reason: str = '', error_msg: str = '') -> None:
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO court_imported_docs
                (job_id, project_slug, court_system, case_number, doc_sequence,
                 source_url, sha256_hash, filename, paperless_doc_id,
                 status, skip_reason, error_msg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_id, project_slug, court_system, case_number,
              doc_sequence, source_url, sha256_hash, filename,
              paperless_doc_id, status, skip_reason, error_msg))
        conn.commit()
    finally:
        conn.close()


def url_already_imported(project_slug: str, source_url: str) -> bool:
    """Tier-1 dedup: check if this URL was already successfully imported."""
    if not source_url:
        return False
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT 1 FROM court_imported_docs
            WHERE project_slug = ? AND source_url = ? AND status = 'imported'
            LIMIT 1
        """, (project_slug, source_url)).fetchone()
        return row is not None
    finally:
        conn.close()


def hash_already_imported(project_slug: str, sha256_hash: str) -> bool:
    """Tier-2 dedup: check if a doc with this hash was already imported."""
    if not sha256_hash:
        return False
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT 1 FROM court_imported_docs
            WHERE project_slug = ? AND sha256_hash = ? AND status = 'imported'
            LIMIT 1
        """, (project_slug, sha256_hash)).fetchone()
        return row is not None
    finally:
        conn.close()
