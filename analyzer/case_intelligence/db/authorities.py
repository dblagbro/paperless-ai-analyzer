"""
CI authority and authority corpus CRUD operations.
"""

import sqlite3
import uuid
from typing import List

from analyzer.case_intelligence.db.schema import _get_conn


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
