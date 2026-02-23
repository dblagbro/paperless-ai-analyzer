"""
Database layer for Case Intelligence AI.

All CI data is in case_intelligence.db — separate from app.db, projects.db,
and llm_usage.db. Schema is initialized idempotently on every startup.
"""

import sqlite3
import uuid
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

CI_DB_PATH = '/app/data/case_intelligence.db'


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CI_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_ci_db():
    """Create all CI tables and indexes (idempotent)."""
    Path(CI_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ci_runs (
                id                  TEXT PRIMARY KEY,
                project_slug        TEXT NOT NULL,
                user_id             INTEGER NOT NULL,
                created_at          TEXT NOT NULL DEFAULT (datetime('now')),
                started_at          TEXT,
                completed_at        TEXT,
                status              TEXT NOT NULL DEFAULT 'draft',

                role                TEXT NOT NULL DEFAULT 'neutral',
                goal_text           TEXT,
                objectives          TEXT,
                jurisdiction_json   TEXT NOT NULL DEFAULT '{}',

                budget_per_run_usd  REAL DEFAULT 10.0,
                budget_monthly_usd  REAL,
                auto_routing        INTEGER DEFAULT 1,
                max_tier            INTEGER DEFAULT 3,

                current_stage       TEXT,
                progress_pct        REAL DEFAULT 0,
                cost_so_far_usd     REAL DEFAULT 0,
                docs_processed      INTEGER DEFAULT 0,
                docs_total          INTEGER DEFAULT 0,
                error_message       TEXT,
                budget_blocked      INTEGER DEFAULT 0,
                budget_blocked_note TEXT,

                findings_summary    TEXT,
                questions_asked     INTEGER DEFAULT 0,
                proceed_with_assumptions INTEGER DEFAULT 0,
                assumptions_made    TEXT
            );

            CREATE TABLE IF NOT EXISTS ci_run_questions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                question    TEXT NOT NULL,
                is_required INTEGER DEFAULT 0,
                answer      TEXT,
                answered_at TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ci_entities (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                entity_type  TEXT NOT NULL,
                name         TEXT NOT NULL,
                aliases      TEXT,
                role_in_case TEXT,
                attributes   TEXT,
                notes        TEXT,
                provenance   TEXT NOT NULL DEFAULT '[]',
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ci_timeline_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                event_date   TEXT,
                date_approx  INTEGER DEFAULT 0,
                event_type   TEXT,
                description  TEXT NOT NULL,
                significance TEXT DEFAULT 'medium',
                parties      TEXT,
                provenance   TEXT NOT NULL DEFAULT '[]',
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ci_disputed_facts (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                   TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                fact_description         TEXT NOT NULL,
                position_a_label         TEXT,
                position_a_text          TEXT,
                position_b_label         TEXT,
                position_b_text          TEXT,
                supporting_evidence_a    TEXT,
                supporting_evidence_b    TEXT,
                resolution_status        TEXT DEFAULT 'open',
                resolution_notes         TEXT,
                created_at               TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ci_contradictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                description       TEXT NOT NULL,
                severity          TEXT NOT NULL,
                contradiction_type TEXT,
                doc_a_provenance  TEXT NOT NULL DEFAULT '[]',
                doc_b_provenance  TEXT NOT NULL DEFAULT '[]',
                explanation       TEXT,
                suggested_action  TEXT,
                created_at        TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ci_theory_ledger (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id               TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                theory_text          TEXT NOT NULL,
                theory_type          TEXT NOT NULL,
                status               TEXT NOT NULL DEFAULT 'proposed',
                confidence           REAL DEFAULT 0.5,
                supporting_evidence  TEXT,
                counter_evidence     TEXT,
                falsification_report TEXT,
                what_would_change    TEXT,
                role_perspective     TEXT,
                created_at           TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ci_authorities (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id         TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                citation       TEXT NOT NULL,
                authority_type TEXT NOT NULL,
                jurisdiction   TEXT,
                source         TEXT,
                source_url     TEXT,
                retrieval_date TEXT,
                reliability    TEXT DEFAULT 'official',
                excerpt        TEXT,
                relevance_note TEXT,
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS ci_reports (
                id           TEXT PRIMARY KEY,
                run_id       TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                user_id      INTEGER NOT NULL,
                instructions TEXT NOT NULL,
                template     TEXT,
                content      TEXT,
                status       TEXT DEFAULT 'pending',
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS ci_authority_corpus (
                id             TEXT PRIMARY KEY,
                citation       TEXT NOT NULL,
                source         TEXT NOT NULL,
                source_url     TEXT,
                retrieval_date TEXT NOT NULL,
                jurisdiction   TEXT NOT NULL,
                authority_type TEXT NOT NULL,
                reliability    TEXT DEFAULT 'official',
                title          TEXT,
                content_text   TEXT,
                is_embedded    INTEGER DEFAULT 0,
                embedded_at    TEXT,
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_ci_runs_project ON ci_runs(project_slug, status);
            CREATE INDEX IF NOT EXISTS idx_ci_runs_user ON ci_runs(user_id);
            CREATE INDEX IF NOT EXISTS idx_ci_entities_run ON ci_entities(run_id, entity_type);
            CREATE INDEX IF NOT EXISTS idx_ci_timeline_run ON ci_timeline_events(run_id, event_date);
            CREATE INDEX IF NOT EXISTS idx_ci_disputes_run ON ci_disputed_facts(run_id, resolution_status);
            CREATE INDEX IF NOT EXISTS idx_ci_contradictions_run ON ci_contradictions(run_id, severity);
            CREATE INDEX IF NOT EXISTS idx_ci_theories_run ON ci_theory_ledger(run_id, status);
            CREATE INDEX IF NOT EXISTS idx_ci_corpus_source ON ci_authority_corpus(source, jurisdiction);

            -- Manager reports (added for hierarchical orchestrator)
            CREATE TABLE IF NOT EXISTS ci_manager_reports (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                manager_id    TEXT NOT NULL,
                status        TEXT DEFAULT 'pending',
                report_json   TEXT,
                worker_count  INTEGER DEFAULT 0,
                docs_assigned INTEGER DEFAULT 0,
                cost_usd      REAL DEFAULT 0,
                started_at    TEXT,
                completed_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ci_manager_reports_run ON ci_manager_reports(run_id);
        """)

        # Idempotent migrations — new columns for hierarchical orchestrator
        for col in (
            "director_count INTEGER DEFAULT 1",
            "manager_count INTEGER",
            "workers_per_manager INTEGER",
            "notification_email TEXT",
            "notify_on_complete INTEGER DEFAULT 1",
            "notify_on_budget INTEGER DEFAULT 1",
            "last_budget_checkpoint_pct REAL DEFAULT 0",
        ):
            try:
                conn.execute(f"ALTER TABLE ci_runs ADD COLUMN {col}")
                logger.debug(f"CI migration: added ci_runs.{col.split()[0]}")
            except Exception:
                pass  # column already exists

    logger.info("Case Intelligence DB initialized")


# ---------------------------------------------------------------------------
# ci_runs CRUD
# ---------------------------------------------------------------------------

def create_ci_run(project_slug: str, user_id: int, role: str = 'neutral',
                  goal_text: str = '', budget_per_run_usd: float = 10.0,
                  jurisdiction_json: str = '{}', objectives: str = '[]',
                  max_tier: int = 3) -> str:
    run_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO ci_runs (id, project_slug, user_id, role, goal_text,
                budget_per_run_usd, jurisdiction_json, objectives, max_tier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, project_slug, user_id, role, goal_text,
              budget_per_run_usd, jurisdiction_json, objectives, max_tier))
    return run_id


def get_ci_run(run_id: str) -> Optional[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute("SELECT * FROM ci_runs WHERE id = ?", (run_id,)).fetchone()


def list_ci_runs(project_slug: str, user_id: int = None) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        if user_id:
            return conn.execute("""
                SELECT * FROM ci_runs WHERE project_slug = ? AND user_id = ?
                ORDER BY created_at DESC
            """, (project_slug, user_id)).fetchall()
        return conn.execute("""
            SELECT * FROM ci_runs WHERE project_slug = ?
            ORDER BY created_at DESC
        """, (project_slug,)).fetchall()


def update_ci_run(run_id: str, **kwargs):
    """Update any ci_runs column(s)."""
    allowed = {
        'status', 'started_at', 'completed_at', 'role', 'goal_text', 'objectives',
        'jurisdiction_json', 'budget_per_run_usd', 'budget_monthly_usd', 'auto_routing',
        'max_tier', 'current_stage', 'progress_pct', 'cost_so_far_usd', 'docs_processed',
        'docs_total', 'error_message', 'budget_blocked', 'budget_blocked_note',
        'findings_summary', 'questions_asked', 'proceed_with_assumptions', 'assumptions_made',
        # Hierarchical orchestrator columns
        'director_count', 'manager_count', 'workers_per_manager',
        'notification_email', 'notify_on_complete', 'notify_on_budget',
        'last_budget_checkpoint_pct',
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    set_clause = ', '.join(f"{col} = ?" for col in updates)
    with _get_conn() as conn:
        conn.execute(f"UPDATE ci_runs SET {set_clause} WHERE id = ?",
                     list(updates.values()) + [run_id])


def increment_ci_run_cost(run_id: str, cost: float):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE ci_runs SET cost_so_far_usd = cost_so_far_usd + ? WHERE id = ?",
            (cost, run_id)
        )


def delete_ci_run(run_id: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM ci_runs WHERE id = ?", (run_id,))


# ---------------------------------------------------------------------------
# ci_run_questions CRUD
# ---------------------------------------------------------------------------

def add_ci_question(run_id: str, question: str, is_required: bool = False) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO ci_run_questions (run_id, question, is_required) VALUES (?, ?, ?)",
            (run_id, question, 1 if is_required else 0)
        )
        return cur.lastrowid


def get_ci_questions(run_id: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_run_questions WHERE run_id = ? ORDER BY id",
            (run_id,)
        ).fetchall()


def answer_ci_question(question_id: int, answer: str):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE ci_run_questions SET answer = ?, answered_at = datetime('now') WHERE id = ?",
            (answer, question_id)
        )


# ---------------------------------------------------------------------------
# ci_entities CRUD
# ---------------------------------------------------------------------------

def upsert_ci_entity(run_id: str, entity_type: str, name: str,
                     aliases: str = '[]', role_in_case: str = '',
                     attributes: str = '{}', notes: str = '',
                     provenance: str = '[]') -> int:
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO ci_entities (run_id, entity_type, name, aliases, role_in_case,
                attributes, notes, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, entity_type, name, aliases, role_in_case, attributes, notes, provenance))
        return cur.lastrowid


def get_ci_entities(run_id: str, entity_type: str = None) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        if entity_type:
            return conn.execute(
                "SELECT * FROM ci_entities WHERE run_id = ? AND entity_type = ? ORDER BY id",
                (run_id, entity_type)
            ).fetchall()
        return conn.execute(
            "SELECT * FROM ci_entities WHERE run_id = ? ORDER BY entity_type, id",
            (run_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# ci_timeline_events CRUD
# ---------------------------------------------------------------------------

def add_ci_event(run_id: str, description: str, event_date: str = None,
                 date_approx: bool = False, event_type: str = 'other',
                 significance: str = 'medium', parties: str = '[]',
                 provenance: str = '[]') -> int:
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO ci_timeline_events (run_id, event_date, date_approx, event_type,
                description, significance, parties, provenance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, event_date, 1 if date_approx else 0, event_type,
              description, significance, parties, provenance))
        return cur.lastrowid


def get_ci_timeline(run_id: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_timeline_events WHERE run_id = ? ORDER BY event_date, id",
            (run_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# ci_contradictions CRUD
# ---------------------------------------------------------------------------

def add_ci_contradiction(run_id: str, description: str, severity: str,
                         doc_a_provenance: str, doc_b_provenance: str,
                         contradiction_type: str = None, explanation: str = None,
                         suggested_action: str = None) -> int:
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO ci_contradictions (run_id, description, severity, contradiction_type,
                doc_a_provenance, doc_b_provenance, explanation, suggested_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, description, severity, contradiction_type,
              doc_a_provenance, doc_b_provenance, explanation, suggested_action))
        return cur.lastrowid


def get_ci_contradictions(run_id: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_contradictions WHERE run_id = ? ORDER BY severity DESC, id",
            (run_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# ci_disputed_facts CRUD
# ---------------------------------------------------------------------------

def add_ci_disputed_fact(run_id: str, fact_description: str,
                         position_a_label: str = None, position_a_text: str = None,
                         position_b_label: str = None, position_b_text: str = None,
                         supporting_evidence_a: str = '[]',
                         supporting_evidence_b: str = '[]') -> int:
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO ci_disputed_facts (run_id, fact_description, position_a_label,
                position_a_text, position_b_label, position_b_text,
                supporting_evidence_a, supporting_evidence_b)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, fact_description, position_a_label, position_a_text,
              position_b_label, position_b_text,
              supporting_evidence_a, supporting_evidence_b))
        return cur.lastrowid


def get_ci_disputed_facts(run_id: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_disputed_facts WHERE run_id = ? ORDER BY id",
            (run_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# ci_theory_ledger CRUD
# ---------------------------------------------------------------------------

def add_ci_theory(run_id: str, theory_text: str, theory_type: str,
                  role_perspective: str = 'neutral',
                  confidence: float = 0.5,
                  supporting_evidence: str = '[]') -> int:
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO ci_theory_ledger (run_id, theory_text, theory_type,
                role_perspective, confidence, supporting_evidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, theory_text, theory_type, role_perspective,
              confidence, supporting_evidence))
        return cur.lastrowid


def update_ci_theory(theory_id: int, **kwargs):
    allowed = {
        'status', 'confidence', 'counter_evidence', 'falsification_report',
        'what_would_change', 'supporting_evidence',
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    set_clause = ', '.join(f"{col} = ?" for col in updates)
    with _get_conn() as conn:
        conn.execute(f"UPDATE ci_theory_ledger SET {set_clause} WHERE id = ?",
                     list(updates.values()) + [theory_id])


def get_ci_theories(run_id: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_theory_ledger WHERE run_id = ? ORDER BY confidence DESC, id",
            (run_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# ci_authorities CRUD
# ---------------------------------------------------------------------------

def add_ci_authority(run_id: str, citation: str, authority_type: str,
                     jurisdiction: str = None, source: str = None,
                     source_url: str = None, reliability: str = 'official',
                     excerpt: str = None, relevance_note: str = None) -> int:
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO ci_authorities (run_id, citation, authority_type, jurisdiction,
                source, source_url, retrieval_date, reliability, excerpt, relevance_note)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
        """, (run_id, citation, authority_type, jurisdiction, source, source_url,
              reliability, excerpt, relevance_note))
        return cur.lastrowid


def get_ci_authorities(run_id: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_authorities WHERE run_id = ? ORDER BY authority_type, id",
            (run_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# ci_reports CRUD
# ---------------------------------------------------------------------------

def create_ci_report(run_id: str, user_id: int, instructions: str,
                     template: str = None) -> str:
    report_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO ci_reports (id, run_id, user_id, instructions, template)
            VALUES (?, ?, ?, ?, ?)
        """, (report_id, run_id, user_id, instructions, template))
    return report_id


def update_ci_report(report_id: str, content: str, status: str = 'complete'):
    with _get_conn() as conn:
        conn.execute("""
            UPDATE ci_reports SET content = ?, status = ?, completed_at = datetime('now')
            WHERE id = ?
        """, (content, status, report_id))


def get_ci_report(report_id: str) -> Optional[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute("SELECT * FROM ci_reports WHERE id = ?", (report_id,)).fetchone()


def get_ci_reports_for_run(run_id: str) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_reports WHERE run_id = ? ORDER BY created_at DESC",
            (run_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# ci_authority_corpus CRUD
# ---------------------------------------------------------------------------

def upsert_authority_corpus_entry(citation: str, source: str, jurisdiction: str,
                                   authority_type: str, title: str = None,
                                   content_text: str = None, source_url: str = None,
                                   reliability: str = 'official') -> str:
    """Insert or update an authority in the shared corpus. Returns the ID."""
    # Check if this citation + source already exists
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_authority_corpus WHERE citation = ? AND source = ?",
            (citation, source)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_authority_corpus
                SET title = ?, content_text = ?, source_url = ?,
                    retrieval_date = datetime('now'), is_embedded = 0
                WHERE id = ?
            """, (title, content_text, source_url, existing['id']))
            return existing['id']
        corpus_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO ci_authority_corpus
                (id, citation, source, source_url, retrieval_date, jurisdiction,
                 authority_type, reliability, title, content_text)
            VALUES (?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?)
        """, (corpus_id, citation, source, source_url, jurisdiction,
              authority_type, reliability, title, content_text))
        return corpus_id


def mark_authority_embedded(corpus_id: str):
    with _get_conn() as conn:
        conn.execute("""
            UPDATE ci_authority_corpus
            SET is_embedded = 1, embedded_at = datetime('now')
            WHERE id = ?
        """, (corpus_id,))


def get_unembedded_authorities(limit: int = 100) -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute("""
            SELECT * FROM ci_authority_corpus
            WHERE is_embedded = 0 AND content_text IS NOT NULL
            LIMIT ?
        """, (limit,)).fetchall()


def get_authority_corpus_stats() -> dict:
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(is_embedded) as embedded,
                COUNT(DISTINCT source) as sources,
                COUNT(DISTINCT jurisdiction) as jurisdictions
            FROM ci_authority_corpus
        """).fetchone()
        return dict(row) if row else {}


# ---------------------------------------------------------------------------
# ci_manager_reports CRUD  (hierarchical orchestrator)
# ---------------------------------------------------------------------------

def upsert_manager_report(run_id: str, manager_id: str,
                           status: str = 'pending',
                           report_json: str = None,
                           worker_count: int = 0,
                           docs_assigned: int = 0,
                           cost_usd: float = 0.0,
                           started_at: str = None,
                           completed_at: str = None) -> None:
    """Insert or update a manager report row for a CI run."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_manager_reports WHERE run_id = ? AND manager_id = ?",
            (run_id, manager_id)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_manager_reports
                SET status = ?, report_json = ?, worker_count = ?,
                    docs_assigned = ?, cost_usd = ?,
                    started_at = COALESCE(started_at, ?), completed_at = ?
                WHERE run_id = ? AND manager_id = ?
            """, (status, report_json, worker_count, docs_assigned, cost_usd,
                  started_at, completed_at, run_id, manager_id))
        else:
            conn.execute("""
                INSERT INTO ci_manager_reports
                    (run_id, manager_id, status, report_json, worker_count,
                     docs_assigned, cost_usd, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, manager_id, status, report_json, worker_count,
                  docs_assigned, cost_usd, started_at, completed_at))


def get_manager_reports(run_id: str) -> List[Dict[str, Any]]:
    """Return all manager reports for a run."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ci_manager_reports WHERE run_id = ? ORDER BY id",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]
