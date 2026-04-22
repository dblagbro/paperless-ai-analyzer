"""
CI analysis CRUD: entities, timeline, contradictions, disputed facts,
theories, web research, and entity merge helpers.
"""

import sqlite3
from typing import Optional, List, Dict, Any

from analyzer.case_intelligence.db.schema import _get_conn


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


def get_ci_entities_active(run_id: str, entity_type: str = None) -> List[sqlite3.Row]:
    """Return only non-merged entities (merged_into IS NULL)."""
    with _get_conn() as conn:
        if entity_type:
            return conn.execute(
                "SELECT * FROM ci_entities "
                "WHERE run_id=? AND entity_type=? AND merged_into IS NULL "
                "ORDER BY entity_type, name",
                (run_id, entity_type)
            ).fetchall()
        return conn.execute(
            "SELECT * FROM ci_entities "
            "WHERE run_id=? AND merged_into IS NULL "
            "ORDER BY entity_type, name",
            (run_id,)
        ).fetchall()


def mark_entity_merged(entity_id: int, canonical_id: int):
    """Mark entity_id as merged into canonical_id."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE ci_entities SET merged_into=? WHERE id=?",
            (canonical_id, entity_id)
        )


def update_entity_aliases(entity_id: int, aliases: str, provenance: str = None):
    """Update aliases (and optionally provenance) on an entity."""
    with _get_conn() as conn:
        if provenance:
            conn.execute(
                "UPDATE ci_entities SET aliases=?, provenance=? WHERE id=?",
                (aliases, provenance, entity_id)
            )
        else:
            conn.execute(
                "UPDATE ci_entities SET aliases=? WHERE id=?",
                (aliases, entity_id)
            )


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
# ci_web_research CRUD  (Phase W — web research results)
# ---------------------------------------------------------------------------

def add_ci_web_research(run_id: str, search_type: str, query: str,
                         source: str, results_json: str = '[]',
                         entity_name: str = None) -> int:
    """Insert a web research result set for a CI run. Returns row id."""
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO ci_web_research (run_id, search_type, query, source, results_json, entity_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, search_type, query, source, results_json, entity_name))
        return cur.lastrowid


def get_ci_web_research(run_id: str,
                         search_type: str = None) -> List[Dict[str, Any]]:
    """Return web research rows for a run, optionally filtered by search_type."""
    with _get_conn() as conn:
        if search_type:
            rows = conn.execute(
                "SELECT * FROM ci_web_research WHERE run_id=? AND search_type=? ORDER BY id",
                (run_id, search_type)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ci_web_research WHERE run_id=? ORDER BY id",
                (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]
