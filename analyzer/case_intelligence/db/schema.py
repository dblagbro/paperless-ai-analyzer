"""
Database layer for Case Intelligence AI.

All CI data is in case_intelligence.db — separate from app.db, projects.db,
and llm_usage.db. Schema is initialized idempotently on every startup.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

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
