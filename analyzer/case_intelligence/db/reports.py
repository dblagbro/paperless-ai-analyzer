"""
CI report CRUD: ci_reports, ci_manager_reports, ci_forensic_report,
ci_discovery_gaps, ci_witness_cards, ci_war_room, ci_deep_forensics,
ci_trial_strategy, ci_multi_model_comparison, ci_settlement_valuation.
"""

import sqlite3
import uuid
from typing import Optional, List, Dict, Any

from analyzer.case_intelligence.db.schema import _get_conn


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
