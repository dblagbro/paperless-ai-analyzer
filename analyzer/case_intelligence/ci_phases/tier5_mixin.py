"""CI Tier 5 White Glove — deep forensics, trial strategy, multi-model synthesis

Extracted from orchestrator.py during the v3.9.2 maintainability refactor.
This module exports a mixin class containing a subset of CIOrchestrator methods.

IMPORTANT: this mixin is **not standalone** — its methods reference `self.*`
state (llm_clients, budget_manager, usage_tracker, ...) initialised in
CIOrchestrator.__init__. Do not instantiate directly; it is only ever used
as a base class for CIOrchestrator.
"""
import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from analyzer.case_intelligence.db import (
    get_ci_run, update_ci_run,
    add_ci_question, get_ci_questions,
    get_ci_entities, get_ci_timeline,
    get_ci_contradictions, get_ci_theories, get_ci_authorities,
    get_ci_disputed_facts,
    upsert_ci_entity, add_ci_event, add_ci_contradiction,
    add_ci_disputed_fact, add_ci_theory, update_ci_theory,
    add_ci_authority, increment_ci_run_cost, increment_ci_run_docs,
    upsert_manager_report, get_manager_reports,
    add_ci_web_research, get_ci_web_research,
    upsert_forensic_report, upsert_discovery_gaps,
    upsert_witness_card, upsert_war_room, update_war_room_senior_notes,
    get_ci_entities_active,
    upsert_deep_forensics, upsert_trial_strategy, upsert_multi_model_comparison,
    upsert_settlement_valuation,
    get_war_room, get_witness_cards,
)
from analyzer.case_intelligence.jurisdiction import JurisdictionProfile
from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

DOMAINS = ['entities', 'timeline', 'financial', 'contradictions', 'theories', 'authorities']


class Tier5PhaseMixin:
    """CI Tier 5 White Glove — deep forensics, trial strategy, multi-model synthesis"""

    def _phase_3b_tier5(self, run_id: str, run, documents: List[Dict],
                        specialist_results: Dict,
                        cancel_event: threading.Event):
        """
        Phase 3B: Tier 5 White Glove analysis.

        Runs in parallel:
          2F+: Deep financial forensics (Benford's, beneficial ownership, round-trips)
          3B-T: Trial strategy memo
          3B-M: Multi-model synthesis (Anthropic + OpenAI)
        """
        role = run.get('role', 'neutral')
        goal = run.get('goal_text') or ''

        jurisdiction_name = 'Not specified'
        try:
            jd = json.loads(run.get('jurisdiction_json') or '{}')
            jurisdiction_name = jd.get('display_name', 'Not specified')
        except Exception:
            pass

        entities = [dict(e) for e in get_ci_entities_active(run_id)]
        timeline = [dict(ev) for ev in get_ci_timeline(run_id)]
        contradictions = [dict(c) for c in get_ci_contradictions(run_id)]
        theories = [dict(t) for t in get_ci_theories(run_id)]
        financial = self._extract_financial_facts(run_id)

        run_obj = get_ci_run(run_id)
        findings_summary = run_obj.get('findings_summary') or ''

        # Fetch already-computed specialist data for trial strategy
        war_room_data = get_war_room(run_id) or {}
        witness_data = [dict(w) for w in get_witness_cards(run_id)]

        from analyzer.case_intelligence.db import get_discovery_gaps
        discovery_data = get_discovery_gaps(run_id) or {}

        self._set_status(run_id, 'running',
                         stage='Phase 3B: White Glove analysis (parallel)', progress=96)

        def run_deep_forensics():
            if cancel_event.is_set():
                return
            if not self.budget_manager.check_and_charge(run_id, 'deep_financial_forensics', 0.050):
                return
            try:
                df_result = self.deep_forensics.analyze(
                    run_id=run_id, role=role, goal_text=goal,
                    financial_data=financial,
                    timeline_data=timeline,
                    entities_data=entities,
                )
                upsert_deep_forensics(
                    run_id=run_id,
                    beneficial_ownership=json.dumps(df_result.get('beneficial_ownership', [])),
                    round_trip_transactions=json.dumps(df_result.get('round_trip_transactions', [])),
                    shell_entity_flags=json.dumps(df_result.get('shell_entity_flags', [])),
                    advanced_structuring=json.dumps(df_result.get('advanced_structuring', [])),
                    layering_schemes=json.dumps(df_result.get('layering_schemes', [])),
                    suspicious_clusters=json.dumps(df_result.get('suspicious_clusters', [])),
                    benford_analysis=json.dumps(df_result.get('benford_analysis', {})),
                    benford_interpretation=df_result.get('benford_interpretation'),
                    summary=df_result.get('summary'),
                    risk_score=df_result.get('risk_score', 0),
                    highest_priority_investigation=df_result.get('highest_priority_investigation'),
                )
                increment_ci_run_cost(run_id, 0.050)
                logger.info(f"CI run {run_id}: Phase 3B deep forensics complete")
            except Exception as e:
                logger.warning(f"Phase 3B deep forensics failed: {e}")

        def run_trial_strategy():
            if cancel_event.is_set():
                return
            if not self.budget_manager.check_and_charge(run_id, 'trial_strategy', 0.040):
                return
            try:
                # Parse discovery data JSON fields if they're strings
                disc = {}
                if discovery_data:
                    disc = dict(discovery_data)
                    for field in ('rfp_list', 'missing_doc_types', 'spoliation_indicators'):
                        if isinstance(disc.get(field), str):
                            try:
                                disc[field] = json.loads(disc[field])
                            except Exception:
                                disc[field] = []

                # Parse war room data JSON fields if they're strings
                wr = {}
                if war_room_data:
                    wr = dict(war_room_data)
                    for field in ('top_dangerous_arguments', 'client_vulnerabilities'):
                        if isinstance(wr.get(field), str):
                            try:
                                wr[field] = json.loads(wr[field])
                            except Exception:
                                wr[field] = []

                ts_result = self.trial_strategist.build_strategy(
                    run_id=run_id, role=role, goal_text=goal,
                    jurisdiction=jurisdiction_name,
                    case_summary=findings_summary[:6000],
                    war_room_data=wr or None,
                    witness_data=witness_data,
                    discovery_data=disc or None,
                )
                upsert_trial_strategy(
                    run_id=run_id,
                    opening_theme=ts_result.get('opening_theme'),
                    our_narrative=ts_result.get('our_narrative'),
                    their_narrative=ts_result.get('their_narrative'),
                    witness_order=json.dumps(ts_result.get('witness_order', [])),
                    key_exhibits=json.dumps(ts_result.get('key_exhibits', [])),
                    motions_in_limine=json.dumps(ts_result.get('motions_in_limine', [])),
                    closing_themes=json.dumps(ts_result.get('closing_themes', [])),
                    jury_profile=json.dumps(ts_result.get('jury_profile', {})),
                    trial_risks=json.dumps(ts_result.get('trial_risks', [])),
                    strategy_memo=ts_result.get('strategy_memo'),
                    case_type=ts_result.get('case_type', 'unknown'),
                )
                increment_ci_run_cost(run_id, 0.040)
                logger.info(f"CI run {run_id}: Phase 3B trial strategy complete")
            except Exception as e:
                logger.warning(f"Phase 3B trial strategy failed: {e}")

        def run_multi_model():
            if cancel_event.is_set():
                return
            if not self.budget_manager.check_and_charge(run_id, 'multi_model_synthesis', 0.080):
                return
            try:
                # Build financial summary string
                fin_summary = '\n'.join(
                    f"{f.get('date', '?')}: ${f.get('amount', 0):,} — {f.get('description', '')}"
                    for f in financial[:10] if isinstance(f, dict)
                ) or 'No financial data.'

                mm_result = self.multi_model.synthesize(
                    run_id=run_id, role=role, goal_text=goal,
                    case_summary=findings_summary[:6000],
                    existing_theories=theories,
                    contradictions=contradictions,
                    financial_summary=fin_summary,
                )
                upsert_multi_model_comparison(
                    run_id=run_id,
                    anthropic_analysis=json.dumps(mm_result.get('anthropic_analysis') or {}),
                    openai_analysis=json.dumps(mm_result.get('openai_analysis') or {}),
                    agreed_theories=json.dumps(mm_result.get('agreed_theories', [])),
                    model_a_only=json.dumps(mm_result.get('model_a_only', [])),
                    model_b_only=json.dumps(mm_result.get('model_b_only', [])),
                    disagreements=json.dumps(mm_result.get('disagreements', [])),
                    merged_summary=mm_result.get('merged_summary'),
                    confidence_in_analysis=mm_result.get('confidence_in_analysis', 0.0),
                    models_agreement_rate=mm_result.get('models_agreement_rate', 0.0),
                )
                increment_ci_run_cost(run_id, 0.080)
                logger.info(f"CI run {run_id}: Phase 3B multi-model synthesis complete")
            except Exception as e:
                logger.warning(f"Phase 3B multi-model synthesis failed: {e}")

        def run_settlement_valuation():
            if cancel_event.is_set():
                return
            if not self.budget_manager.check_and_charge(run_id, 'settlement_valuation', 0.030):
                return
            try:
                case_summary = f"{goal}\n\n{findings_summary}"
                sv_result = self.settlement_valuator.valuate(
                    run_id=run_id, role=role, goal_text=goal,
                    jurisdiction=jurisdiction_name,
                    case_summary=case_summary,
                    war_room_data=war_room_data,
                    financial_data=financial,
                    theories=theories,
                )
                upsert_settlement_valuation(
                    run_id=run_id,
                    damages_breakdown=json.dumps(sv_result.get('damages_breakdown', [])),
                    total_exposure=json.dumps(sv_result.get('total_exposure', {})),
                    comparable_verdict_context=sv_result.get('comparable_verdict_context'),
                    litigation_cost_model=json.dumps(sv_result.get('litigation_cost_model', {})),
                    fee_shifting_risk=json.dumps(sv_result.get('fee_shifting_risk', {})),
                    insurance_flags=json.dumps(sv_result.get('insurance_flags', [])),
                    leverage_timeline=json.dumps(sv_result.get('leverage_timeline', [])),
                    settlement_recommendation=json.dumps(sv_result.get('settlement_recommendation', {})),
                    optimal_settlement_timing=sv_result.get('optimal_settlement_timing'),
                    mediation_strategy=json.dumps(sv_result.get('mediation_strategy', {})),
                    summary_memo=sv_result.get('summary_memo'),
                )
                increment_ci_run_cost(run_id, 0.030)
                logger.info(f"CI run {run_id}: Phase 3B settlement valuation complete")
            except Exception as e:
                logger.warning(f"Phase 3B settlement valuation failed: {e}")

        with ThreadPoolExecutor(max_workers=4,
                                thread_name_prefix='ci-tier5') as executor:
            futures = [
                executor.submit(run_deep_forensics),
                executor.submit(run_trial_strategy),
                executor.submit(run_multi_model),
                executor.submit(run_settlement_valuation),
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.warning(f"Phase 3B future failed: {e}")

        logger.info(f"CI run {run_id}: Phase 3B complete")

    # -----------------------------------------------------------------------
    # Phase D2: Director synthesizes report
    # -----------------------------------------------------------------------

