"""
CI run, run-share, and run-question CRUD operations.
"""

import sqlite3
import uuid
from typing import Optional, List

from analyzer.case_intelligence.db.schema import _get_conn


# ---------------------------------------------------------------------------
# ci_runs CRUD
# ---------------------------------------------------------------------------

def create_ci_run(project_slug: str, user_id: int, role: str = 'neutral',
                  goal_text: str = '', budget_per_run_usd: float = 10.0,
                  jurisdiction_json: str = '{}', objectives: str = '[]',
                  max_tier: int = 3, notification_email: str = '',
                  notify_on_complete: int = 1, notify_on_budget: int = 1,
                  allow_overage_pct: int = 0) -> str:
    run_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO ci_runs (id, project_slug, user_id, role, goal_text,
                budget_per_run_usd, jurisdiction_json, objectives, max_tier,
                notification_email, notify_on_complete, notify_on_budget,
                allow_overage_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, project_slug, user_id, role, goal_text,
              budget_per_run_usd, jurisdiction_json, objectives, max_tier,
              notification_email, notify_on_complete, notify_on_budget,
              allow_overage_pct))
    return run_id


def get_ci_run(run_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM ci_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


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
        # Enhanced progress bar columns
        'tokens_in', 'tokens_out', 'active_managers', 'active_workers',
        # Web research config
        'web_research_config',
        # Budget overage policy
        'allow_overage_pct',
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


def increment_ci_run_docs(run_id: str, count: int = 1):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE ci_runs SET docs_processed = docs_processed + ? WHERE id = ?",
            (count, run_id)
        )


def delete_ci_run(run_id: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM ci_runs WHERE id = ?", (run_id,))


# ---------------------------------------------------------------------------
# ci_run_shares CRUD
# ---------------------------------------------------------------------------

def add_ci_run_share(run_id: str, shared_with: int, shared_by: int):
    """Share a run with another user. Silently ignores duplicate."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO ci_run_shares (run_id, shared_with, shared_by) VALUES (?, ?, ?)",
            (run_id, shared_with, shared_by),
        )


def remove_ci_run_share(run_id: str, shared_with: int):
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM ci_run_shares WHERE run_id=? AND shared_with=?",
            (run_id, shared_with),
        )


def list_ci_run_shares(run_id: str) -> list:
    """Return rows with shared_with, shared_by, shared_at for a run."""
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ci_run_shares WHERE run_id=? ORDER BY shared_at",
            (run_id,),
        ).fetchall()


def get_run_ids_shared_with(user_id: int) -> list:
    """Return list of run_ids that have been shared with the given user."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT run_id FROM ci_run_shares WHERE shared_with=?",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]


def is_run_shared_with(run_id: str, user_id: int) -> bool:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM ci_run_shares WHERE run_id=? AND shared_with=?",
            (run_id, user_id),
        ).fetchone()
        return row is not None


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
