"""
Database layer for Case Intelligence AI.

All CI data is in case_intelligence.db — separate from app.db, projects.db,
and llm_usage.db. Schema is initialized idempotently on every startup.
"""

import sqlite3
import uuid
import logging
import json
from datetime import datetime, timezone
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

            -- Run sharing
            CREATE TABLE IF NOT EXISTS ci_run_shares (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                shared_with   INTEGER NOT NULL,
                shared_by     INTEGER NOT NULL,
                shared_at     TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(run_id, shared_with)
            );
            CREATE INDEX IF NOT EXISTS idx_ci_run_shares_user ON ci_run_shares(shared_with);

            -- Web research results (Phase W)
            CREATE TABLE IF NOT EXISTS ci_web_research (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                search_type  TEXT NOT NULL,   -- 'legal_authority' | 'entity_background' | 'general'
                query        TEXT NOT NULL,
                source       TEXT NOT NULL,   -- 'courtlistener' | 'caselaw_access' | 'web_search' | etc.
                results_json TEXT,            -- JSON array of result objects
                entity_name  TEXT,            -- populated for entity_background searches
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_web_research_run ON ci_web_research(run_id, search_type);
        """)

        # v3.6.6 new specialist tables
        conn.executescript("""
            -- Forensic accounting report (Tier 3+)
            CREATE TABLE IF NOT EXISTS ci_forensic_report (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                flagged_transactions    TEXT NOT NULL DEFAULT '[]',
                cash_flow_by_party      TEXT NOT NULL DEFAULT '[]',
                balance_discrepancies   TEXT NOT NULL DEFAULT '[]',
                missing_transactions    TEXT NOT NULL DEFAULT '[]',
                transaction_chains      TEXT NOT NULL DEFAULT '[]',
                summary                 TEXT,
                total_exposure_usd      REAL DEFAULT 0,
                created_at              TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_forensic_run ON ci_forensic_report(run_id);

            -- Discovery gap analysis (Tier 3+)
            CREATE TABLE IF NOT EXISTS ci_discovery_gaps (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                missing_doc_types       TEXT NOT NULL DEFAULT '[]',
                custodian_gaps          TEXT NOT NULL DEFAULT '[]',
                spoliation_indicators   TEXT NOT NULL DEFAULT '[]',
                rfp_list                TEXT NOT NULL DEFAULT '[]',
                subpoena_targets        TEXT NOT NULL DEFAULT '[]',
                summary                 TEXT,
                created_at              TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_discovery_run ON ci_discovery_gaps(run_id);

            -- Witness intelligence cards (Tier 4+)
            CREATE TABLE IF NOT EXISTS ci_witness_cards (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                witness_name            TEXT NOT NULL,
                credibility_score       REAL DEFAULT 0.5,
                impeachment_points      TEXT NOT NULL DEFAULT '[]',
                financial_interest      TEXT NOT NULL DEFAULT '{}',
                prior_inconsistencies   TEXT NOT NULL DEFAULT '[]',
                public_record_flags     TEXT NOT NULL DEFAULT '[]',
                deposition_order        INTEGER DEFAULT 99,
                key_questions           TEXT NOT NULL DEFAULT '[]',
                vulnerability_summary   TEXT,
                created_at              TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_witness_run ON ci_witness_cards(run_id);

            -- War room / opposing counsel simulation (Tier 4+)
            CREATE TABLE IF NOT EXISTS ci_war_room (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                opposing_case_summary   TEXT,
                top_dangerous_arguments TEXT NOT NULL DEFAULT '[]',
                client_vulnerabilities  TEXT NOT NULL DEFAULT '[]',
                smoking_guns            TEXT NOT NULL DEFAULT '[]',
                settlement_analysis     TEXT NOT NULL DEFAULT '{}',
                likelihood_pct          REAL DEFAULT 50,
                war_room_memo           TEXT,
                senior_partner_notes    TEXT,
                created_at              TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_warroom_run ON ci_war_room(run_id);
        """)

        # v3.7.2 Tier 5 White Glove tables
        conn.executescript("""
            -- Deep financial forensics (Tier 5)
            CREATE TABLE IF NOT EXISTS ci_deep_forensics (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                beneficial_ownership    TEXT NOT NULL DEFAULT '[]',
                round_trip_transactions TEXT NOT NULL DEFAULT '[]',
                shell_entity_flags      TEXT NOT NULL DEFAULT '[]',
                advanced_structuring    TEXT NOT NULL DEFAULT '[]',
                layering_schemes        TEXT NOT NULL DEFAULT '[]',
                suspicious_clusters     TEXT NOT NULL DEFAULT '[]',
                benford_analysis        TEXT NOT NULL DEFAULT '{}',
                benford_interpretation  TEXT,
                summary                 TEXT,
                risk_score              REAL DEFAULT 0,
                highest_priority_investigation TEXT,
                created_at              TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_deep_forensics_run ON ci_deep_forensics(run_id);

            -- Trial strategy memo (Tier 5)
            CREATE TABLE IF NOT EXISTS ci_trial_strategy (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                opening_theme           TEXT,
                our_narrative           TEXT,
                their_narrative         TEXT,
                witness_order           TEXT NOT NULL DEFAULT '[]',
                key_exhibits            TEXT NOT NULL DEFAULT '[]',
                motions_in_limine       TEXT NOT NULL DEFAULT '[]',
                closing_themes          TEXT NOT NULL DEFAULT '[]',
                jury_profile            TEXT NOT NULL DEFAULT '{}',
                trial_risks             TEXT NOT NULL DEFAULT '[]',
                strategy_memo           TEXT,
                case_type               TEXT DEFAULT 'unknown',
                created_at              TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_trial_strategy_run ON ci_trial_strategy(run_id);

            -- Multi-model comparison (Tier 5)
            CREATE TABLE IF NOT EXISTS ci_multi_model_comparison (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                  TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                anthropic_analysis      TEXT NOT NULL DEFAULT '{}',
                openai_analysis         TEXT NOT NULL DEFAULT '{}',
                agreed_theories         TEXT NOT NULL DEFAULT '[]',
                model_a_only            TEXT NOT NULL DEFAULT '[]',
                model_b_only            TEXT NOT NULL DEFAULT '[]',
                disagreements           TEXT NOT NULL DEFAULT '[]',
                merged_summary          TEXT,
                confidence_in_analysis  REAL DEFAULT 0,
                models_agreement_rate   REAL DEFAULT 0,
                created_at              TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_multimodel_run ON ci_multi_model_comparison(run_id);

            -- Settlement valuation (Tier 5)
            CREATE TABLE IF NOT EXISTS ci_settlement_valuation (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                      TEXT NOT NULL REFERENCES ci_runs(id) ON DELETE CASCADE,
                damages_breakdown           TEXT NOT NULL DEFAULT '[]',
                total_exposure              TEXT NOT NULL DEFAULT '{}',
                comparable_verdict_context  TEXT,
                litigation_cost_model       TEXT NOT NULL DEFAULT '{}',
                fee_shifting_risk           TEXT NOT NULL DEFAULT '{}',
                insurance_flags             TEXT NOT NULL DEFAULT '[]',
                leverage_timeline           TEXT NOT NULL DEFAULT '[]',
                settlement_recommendation   TEXT NOT NULL DEFAULT '{}',
                optimal_settlement_timing   TEXT,
                mediation_strategy          TEXT NOT NULL DEFAULT '{}',
                summary_memo                TEXT,
                created_at                  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ci_settlement_run ON ci_settlement_valuation(run_id);
        """)

        # Idempotent migrations — new columns for hierarchical orchestrator + web research
        for col in (
            "director_count INTEGER DEFAULT 1",
            "manager_count INTEGER",
            "workers_per_manager INTEGER",
            "notification_email TEXT",
            "notify_on_complete INTEGER DEFAULT 1",
            "notify_on_budget INTEGER DEFAULT 1",
            "last_budget_checkpoint_pct REAL DEFAULT 0",
            # Enhanced progress bar columns
            "tokens_in INTEGER DEFAULT 0",
            "tokens_out INTEGER DEFAULT 0",
            "active_managers INTEGER DEFAULT 0",
            "active_workers INTEGER DEFAULT 0",
            # Web research config (Phase W)
            "web_research_config TEXT DEFAULT '{}'",
            # v3.6.6: analysis_tier alias for max_tier (same value, read from max_tier)
            "analysis_tier INTEGER DEFAULT 3",
            # v3.7.1: budget overage policy
            # 0 = hard block at 100% of budget (default)
            # 20 = allow up to 120% of budget before blocking
            # -1 = unlimited overage (budget is a goal only, never blocked)
            "allow_overage_pct INTEGER DEFAULT 0",
        ):
            try:
                conn.execute(f"ALTER TABLE ci_runs ADD COLUMN {col}")
                logger.debug(f"CI migration: added ci_runs.{col.split()[0]}")
            except Exception:
                pass  # column already exists

        # v3.6.6: new ci_theory_ledger columns
        for col in (
            "legal_element_mapping TEXT",
            "theory_legal_memo TEXT",
            "companion_theories TEXT",
            "discovery_needed TEXT",
            "model_source TEXT",
        ):
            try:
                conn.execute(f"ALTER TABLE ci_theory_ledger ADD COLUMN {col}")
            except Exception:
                pass

        # v3.7.2: opposing counsel checklist in war room
        try:
            conn.execute("ALTER TABLE ci_war_room ADD COLUMN opposing_counsel_checklist TEXT NOT NULL DEFAULT '[]'")
        except Exception:
            pass

        # v3.6.6: entity merge support
        try:
            conn.execute("ALTER TABLE ci_entities ADD COLUMN merged_into INTEGER")
        except Exception:
            pass
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ci_entities_merged "
                "ON ci_entities(run_id, merged_into)"
            )
        except Exception:
            pass

    logger.info("Case Intelligence DB initialized")
    recover_orphaned_runs()


def recover_orphaned_runs():
    """Mark any 'running'/'queued' runs as interrupted on startup.

    When the service restarts, orchestrator threads are killed.  Any run
    that was still in-flight at shutdown will be stuck in 'running' forever
    unless we reset it here.  We use 'interrupted' (not 'failed') so the UI
    can offer a Re-run button rather than treating it as an error.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, started_at, progress_pct, docs_processed, docs_total FROM ci_runs "
            "WHERE status IN ('running', 'queued')"
        ).fetchall()
        if not rows:
            return
        now_str = datetime.now(timezone.utc).isoformat()
        for row in rows:
            pct = row['progress_pct'] or 0
            note = (
                f"Run interrupted by service restart at {pct:.0f}% progress "
                f"({row['docs_processed'] or 0}/{row['docs_total'] or 0} docs). "
                "Use Re-run to restart with the same parameters."
            )
            conn.execute(
                """UPDATE ci_runs
                   SET status='interrupted',
                       current_stage='Interrupted',
                       error_message=?,
                       completed_at=?
                   WHERE id=?""",
                (note, now_str, row['id'])
            )
        logger.warning(
            f"recover_orphaned_runs: marked {len(rows)} orphaned run(s) as interrupted: "
            + ', '.join(r['id'] for r in rows)
        )


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
                    docs_assigned = COALESCE(NULLIF(?, 0), docs_assigned, ?),
                    cost_usd = ?,
                    started_at = COALESCE(started_at, ?), completed_at = ?
                WHERE run_id = ? AND manager_id = ?
            """, (status, report_json, worker_count, docs_assigned, docs_assigned,
                  cost_usd, started_at, completed_at, run_id, manager_id))
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


# ---------------------------------------------------------------------------
# ci_forensic_report CRUD  (Tier 3+)
# ---------------------------------------------------------------------------

def upsert_forensic_report(run_id: str,
                            flagged_transactions: str = '[]',
                            cash_flow_by_party: str = '[]',
                            balance_discrepancies: str = '[]',
                            missing_transactions: str = '[]',
                            transaction_chains: str = '[]',
                            summary: str = None,
                            total_exposure_usd: float = 0.0) -> int:
    """Insert or replace the forensic report for a run. Returns row id."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_forensic_report WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_forensic_report
                SET flagged_transactions=?, cash_flow_by_party=?, balance_discrepancies=?,
                    missing_transactions=?, transaction_chains=?, summary=?, total_exposure_usd=?
                WHERE run_id=?
            """, (flagged_transactions, cash_flow_by_party, balance_discrepancies,
                  missing_transactions, transaction_chains, summary, total_exposure_usd, run_id))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_forensic_report
                (run_id, flagged_transactions, cash_flow_by_party, balance_discrepancies,
                 missing_transactions, transaction_chains, summary, total_exposure_usd)
            VALUES (?,?,?,?,?,?,?,?)
        """, (run_id, flagged_transactions, cash_flow_by_party, balance_discrepancies,
              missing_transactions, transaction_chains, summary, total_exposure_usd))
        return cur.lastrowid


def get_forensic_report(run_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ci_forensic_report WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# ci_discovery_gaps CRUD  (Tier 3+)
# ---------------------------------------------------------------------------

def upsert_discovery_gaps(run_id: str,
                           missing_doc_types: str = '[]',
                           custodian_gaps: str = '[]',
                           spoliation_indicators: str = '[]',
                           rfp_list: str = '[]',
                           subpoena_targets: str = '[]',
                           summary: str = None) -> int:
    """Insert or replace the discovery gap analysis for a run. Returns row id."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_discovery_gaps WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_discovery_gaps
                SET missing_doc_types=?, custodian_gaps=?, spoliation_indicators=?,
                    rfp_list=?, subpoena_targets=?, summary=?
                WHERE run_id=?
            """, (missing_doc_types, custodian_gaps, spoliation_indicators,
                  rfp_list, subpoena_targets, summary, run_id))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_discovery_gaps
                (run_id, missing_doc_types, custodian_gaps, spoliation_indicators,
                 rfp_list, subpoena_targets, summary)
            VALUES (?,?,?,?,?,?,?)
        """, (run_id, missing_doc_types, custodian_gaps, spoliation_indicators,
              rfp_list, subpoena_targets, summary))
        return cur.lastrowid


def get_discovery_gaps(run_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ci_discovery_gaps WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# ci_witness_cards CRUD  (Tier 4+)
# ---------------------------------------------------------------------------

def upsert_witness_card(run_id: str, witness_name: str,
                         credibility_score: float = 0.5,
                         impeachment_points: str = '[]',
                         financial_interest: str = '{}',
                         prior_inconsistencies: str = '[]',
                         public_record_flags: str = '[]',
                         deposition_order: int = 99,
                         key_questions: str = '[]',
                         vulnerability_summary: str = None) -> int:
    """Upsert a witness card by run_id + name. Returns row id."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_witness_cards WHERE run_id=? AND witness_name=?",
            (run_id, witness_name)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_witness_cards
                SET credibility_score=?, impeachment_points=?, financial_interest=?,
                    prior_inconsistencies=?, public_record_flags=?, deposition_order=?,
                    key_questions=?, vulnerability_summary=?
                WHERE run_id=? AND witness_name=?
            """, (credibility_score, impeachment_points, financial_interest,
                  prior_inconsistencies, public_record_flags, deposition_order,
                  key_questions, vulnerability_summary, run_id, witness_name))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_witness_cards
                (run_id, witness_name, credibility_score, impeachment_points,
                 financial_interest, prior_inconsistencies, public_record_flags,
                 deposition_order, key_questions, vulnerability_summary)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (run_id, witness_name, credibility_score, impeachment_points,
              financial_interest, prior_inconsistencies, public_record_flags,
              deposition_order, key_questions, vulnerability_summary))
        return cur.lastrowid


def get_witness_cards(run_id: str) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ci_witness_cards WHERE run_id=? ORDER BY deposition_order, id",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# ci_war_room CRUD  (Tier 4+)
# ---------------------------------------------------------------------------

def upsert_war_room(run_id: str,
                     opposing_case_summary: str = None,
                     top_dangerous_arguments: str = '[]',
                     client_vulnerabilities: str = '[]',
                     smoking_guns: str = '[]',
                     settlement_analysis: str = '{}',
                     likelihood_pct: float = 50.0,
                     war_room_memo: str = None,
                     senior_partner_notes: str = None,
                     opposing_counsel_checklist: str = '[]') -> int:
    """Insert or replace the war room report for a run. Returns row id."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_war_room WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_war_room
                SET opposing_case_summary=?, top_dangerous_arguments=?,
                    client_vulnerabilities=?, smoking_guns=?, settlement_analysis=?,
                    likelihood_pct=?, war_room_memo=?, senior_partner_notes=?,
                    opposing_counsel_checklist=?
                WHERE run_id=?
            """, (opposing_case_summary, top_dangerous_arguments,
                  client_vulnerabilities, smoking_guns, settlement_analysis,
                  likelihood_pct, war_room_memo, senior_partner_notes,
                  opposing_counsel_checklist, run_id))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_war_room
                (run_id, opposing_case_summary, top_dangerous_arguments,
                 client_vulnerabilities, smoking_guns, settlement_analysis,
                 likelihood_pct, war_room_memo, senior_partner_notes,
                 opposing_counsel_checklist)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (run_id, opposing_case_summary, top_dangerous_arguments,
              client_vulnerabilities, smoking_guns, settlement_analysis,
              likelihood_pct, war_room_memo, senior_partner_notes,
              opposing_counsel_checklist))
        return cur.lastrowid


def get_war_room(run_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ci_war_room WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


def update_war_room_senior_notes(run_id: str, senior_partner_notes: str):
    """Update only the senior partner notes in war room (Phase 3A output)."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE ci_war_room SET senior_partner_notes=? WHERE run_id=?",
            (senior_partner_notes, run_id)
        )


# ---------------------------------------------------------------------------
# ci_entities — query filtered for non-merged
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ci_deep_forensics CRUD  (Tier 5)
# ---------------------------------------------------------------------------

def upsert_deep_forensics(run_id: str,
                           beneficial_ownership: str = '[]',
                           round_trip_transactions: str = '[]',
                           shell_entity_flags: str = '[]',
                           advanced_structuring: str = '[]',
                           layering_schemes: str = '[]',
                           suspicious_clusters: str = '[]',
                           benford_analysis: str = '{}',
                           benford_interpretation: str = None,
                           summary: str = None,
                           risk_score: float = 0.0,
                           highest_priority_investigation: str = None) -> int:
    """Insert or replace deep forensics report for a run. Returns row id."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_deep_forensics WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_deep_forensics
                SET beneficial_ownership=?, round_trip_transactions=?,
                    shell_entity_flags=?, advanced_structuring=?,
                    layering_schemes=?, suspicious_clusters=?,
                    benford_analysis=?, benford_interpretation=?,
                    summary=?, risk_score=?, highest_priority_investigation=?
                WHERE run_id=?
            """, (beneficial_ownership, round_trip_transactions,
                  shell_entity_flags, advanced_structuring,
                  layering_schemes, suspicious_clusters,
                  benford_analysis, benford_interpretation,
                  summary, risk_score, highest_priority_investigation, run_id))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_deep_forensics
                (run_id, beneficial_ownership, round_trip_transactions,
                 shell_entity_flags, advanced_structuring, layering_schemes,
                 suspicious_clusters, benford_analysis, benford_interpretation,
                 summary, risk_score, highest_priority_investigation)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (run_id, beneficial_ownership, round_trip_transactions,
              shell_entity_flags, advanced_structuring, layering_schemes,
              suspicious_clusters, benford_analysis, benford_interpretation,
              summary, risk_score, highest_priority_investigation))
        return cur.lastrowid


def get_deep_forensics(run_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ci_deep_forensics WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# ci_trial_strategy CRUD  (Tier 5)
# ---------------------------------------------------------------------------

def upsert_trial_strategy(run_id: str,
                           opening_theme: str = None,
                           our_narrative: str = None,
                           their_narrative: str = None,
                           witness_order: str = '[]',
                           key_exhibits: str = '[]',
                           motions_in_limine: str = '[]',
                           closing_themes: str = '[]',
                           jury_profile: str = '{}',
                           trial_risks: str = '[]',
                           strategy_memo: str = None,
                           case_type: str = 'unknown') -> int:
    """Insert or replace trial strategy for a run. Returns row id."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_trial_strategy WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_trial_strategy
                SET opening_theme=?, our_narrative=?, their_narrative=?,
                    witness_order=?, key_exhibits=?, motions_in_limine=?,
                    closing_themes=?, jury_profile=?, trial_risks=?,
                    strategy_memo=?, case_type=?
                WHERE run_id=?
            """, (opening_theme, our_narrative, their_narrative,
                  witness_order, key_exhibits, motions_in_limine,
                  closing_themes, jury_profile, trial_risks,
                  strategy_memo, case_type, run_id))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_trial_strategy
                (run_id, opening_theme, our_narrative, their_narrative,
                 witness_order, key_exhibits, motions_in_limine,
                 closing_themes, jury_profile, trial_risks,
                 strategy_memo, case_type)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (run_id, opening_theme, our_narrative, their_narrative,
              witness_order, key_exhibits, motions_in_limine,
              closing_themes, jury_profile, trial_risks,
              strategy_memo, case_type))
        return cur.lastrowid


def get_trial_strategy(run_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ci_trial_strategy WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# ci_multi_model_comparison CRUD  (Tier 5)
# ---------------------------------------------------------------------------

def upsert_multi_model_comparison(run_id: str,
                                   anthropic_analysis: str = '{}',
                                   openai_analysis: str = '{}',
                                   agreed_theories: str = '[]',
                                   model_a_only: str = '[]',
                                   model_b_only: str = '[]',
                                   disagreements: str = '[]',
                                   merged_summary: str = None,
                                   confidence_in_analysis: float = 0.0,
                                   models_agreement_rate: float = 0.0) -> int:
    """Insert or replace multi-model comparison for a run. Returns row id."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_multi_model_comparison WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_multi_model_comparison
                SET anthropic_analysis=?, openai_analysis=?,
                    agreed_theories=?, model_a_only=?, model_b_only=?,
                    disagreements=?, merged_summary=?,
                    confidence_in_analysis=?, models_agreement_rate=?
                WHERE run_id=?
            """, (anthropic_analysis, openai_analysis,
                  agreed_theories, model_a_only, model_b_only,
                  disagreements, merged_summary,
                  confidence_in_analysis, models_agreement_rate, run_id))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_multi_model_comparison
                (run_id, anthropic_analysis, openai_analysis,
                 agreed_theories, model_a_only, model_b_only,
                 disagreements, merged_summary,
                 confidence_in_analysis, models_agreement_rate)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (run_id, anthropic_analysis, openai_analysis,
              agreed_theories, model_a_only, model_b_only,
              disagreements, merged_summary,
              confidence_in_analysis, models_agreement_rate))
        return cur.lastrowid


def get_multi_model_comparison(run_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ci_multi_model_comparison WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# ci_settlement_valuation CRUD  (Tier 5)
# ---------------------------------------------------------------------------

def upsert_settlement_valuation(run_id: str,
                                 damages_breakdown: str = '[]',
                                 total_exposure: str = '{}',
                                 comparable_verdict_context: str = None,
                                 litigation_cost_model: str = '{}',
                                 fee_shifting_risk: str = '{}',
                                 insurance_flags: str = '[]',
                                 leverage_timeline: str = '[]',
                                 settlement_recommendation: str = '{}',
                                 optimal_settlement_timing: str = None,
                                 mediation_strategy: str = '{}',
                                 summary_memo: str = None) -> int:
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM ci_settlement_valuation WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing:
            conn.execute("""
                UPDATE ci_settlement_valuation
                SET damages_breakdown=?, total_exposure=?, comparable_verdict_context=?,
                    litigation_cost_model=?, fee_shifting_risk=?, insurance_flags=?,
                    leverage_timeline=?, settlement_recommendation=?,
                    optimal_settlement_timing=?, mediation_strategy=?, summary_memo=?
                WHERE run_id=?
            """, (damages_breakdown, total_exposure, comparable_verdict_context,
                  litigation_cost_model, fee_shifting_risk, insurance_flags,
                  leverage_timeline, settlement_recommendation,
                  optimal_settlement_timing, mediation_strategy, summary_memo, run_id))
            return existing['id']
        cur = conn.execute("""
            INSERT INTO ci_settlement_valuation
                (run_id, damages_breakdown, total_exposure, comparable_verdict_context,
                 litigation_cost_model, fee_shifting_risk, insurance_flags,
                 leverage_timeline, settlement_recommendation,
                 optimal_settlement_timing, mediation_strategy, summary_memo)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (run_id, damages_breakdown, total_exposure, comparable_verdict_context,
              litigation_cost_model, fee_shifting_risk, insurance_flags,
              leverage_timeline, settlement_recommendation,
              optimal_settlement_timing, mediation_strategy, summary_memo))
        return cur.lastrowid


def get_settlement_valuation(run_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ci_settlement_valuation WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


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
