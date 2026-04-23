"""CI Tier 3+ specialist phases — forensic accounting, discovery gaps, witness intelligence

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


class SpecialistPhasesMixin:
    """CI Tier 3+ specialist phases — forensic accounting, discovery gaps, witness intelligence"""

    def _run_specialist_phases(self, run_id: str, run, documents: List[Dict],
                                max_tier: int,
                                cancel_event: threading.Event) -> Dict:
        """
        Run Tier 3+ specialist phases in parallel where possible.

        Phase 2F: Forensic Accounting (Tier 3+)
        Phase 2D: Discovery Gap Analysis (Tier 3+)
        Phase 2W: Witness Intelligence (Tier 4+)
        Phase 2R: War Room (Tier 4+)
        """
        results = {}
        if max_tier < 3:
            return results

        # Fetch all Phase 1 results from DB
        entities = [dict(e) for e in get_ci_entities_active(run_id)]
        timeline = [dict(ev) for ev in get_ci_timeline(run_id)]
        contradictions = [dict(c) for c in get_ci_contradictions(run_id)]
        theories = [dict(t) for t in get_ci_theories(run_id)]

        # Extract financial facts from manager reports
        financial = self._extract_financial_facts(run_id)

        jurisdiction_name = 'Not specified'
        try:
            jd = json.loads(run.get('jurisdiction_json') or '{}')
            jurisdiction_name = jd.get('display_name', 'Not specified')
        except Exception:
            pass

        role = run.get('role', 'neutral')
        goal = run.get('goal_text') or ''

        # ── Phase 2F: Forensic Accounting (Tier 3+) ─────────────────────
        if max_tier >= 3 and not cancel_event.is_set():
            self._set_status(run_id, 'running',
                             stage='Phase 2F: Forensic accounting analysis', progress=86)
            if self.budget_manager.check_and_charge(run_id, 'forensic_accounting', 0.025):
                try:
                    fa_result = self.forensic_accountant.analyze(
                        run_id=run_id, role=role, goal_text=goal,
                        financial_data=financial,
                        timeline_data=timeline,
                        entities_data=entities,
                    )
                    upsert_forensic_report(
                        run_id=run_id,
                        flagged_transactions=json.dumps(fa_result.get('flagged_transactions', [])),
                        cash_flow_by_party=json.dumps(fa_result.get('cash_flow_by_party', [])),
                        balance_discrepancies=json.dumps(fa_result.get('balance_discrepancies', [])),
                        missing_transactions=json.dumps(fa_result.get('missing_transactions', [])),
                        transaction_chains=json.dumps(fa_result.get('transaction_chains', [])),
                        summary=fa_result.get('summary'),
                        total_exposure_usd=fa_result.get('total_documented_exposure_usd', 0),
                    )
                    increment_ci_run_cost(run_id, 0.025)
                    results['forensic'] = fa_result
                    logger.info(f"CI run {run_id}: Phase 2F complete")
                except Exception as e:
                    logger.warning(f"Phase 2F forensic accounting failed: {e}")

        # ── Phase 2D: Discovery Gap Analysis (Tier 3+) ───────────────────
        if max_tier >= 3 and not cancel_event.is_set():
            self._set_status(run_id, 'running',
                             stage='Phase 2D: Discovery gap analysis', progress=87)
            if self.budget_manager.check_and_charge(run_id, 'discovery_gap_analysis', 0.020):
                try:
                    da_result = self.discovery_analyst.analyze(
                        run_id=run_id, role=role, goal_text=goal,
                        jurisdiction=jurisdiction_name,
                        documents=documents,
                        entities=entities,
                        timeline=timeline,
                        financial=financial,
                    )
                    upsert_discovery_gaps(
                        run_id=run_id,
                        missing_doc_types=json.dumps(da_result.get('missing_document_types', [])),
                        custodian_gaps=json.dumps(da_result.get('custodian_gaps', [])),
                        spoliation_indicators=json.dumps(da_result.get('spoliation_indicators', [])),
                        rfp_list=json.dumps(da_result.get('rfp_list', [])),
                        subpoena_targets=json.dumps(da_result.get('subpoena_targets', [])),
                        summary=da_result.get('summary'),
                    )
                    increment_ci_run_cost(run_id, 0.020)
                    results['discovery'] = da_result
                    logger.info(f"CI run {run_id}: Phase 2D complete")
                except Exception as e:
                    logger.warning(f"Phase 2D discovery gap analysis failed: {e}")

        if max_tier < 4:
            return results

        # ── Phase 2W + 2R: Witness Intelligence + War Room (Tier 4+) ────
        # Run in parallel
        def run_witnesses():
            if cancel_event.is_set():
                return
            if not self.budget_manager.check_and_charge(run_id, 'witness_intelligence', 0.15):
                return
            try:
                dossiers = self.witness_analyst.build_dossiers(
                    run_id=run_id, role=role, goal_text=goal,
                    entities=entities, documents=documents,
                    contradictions=contradictions, financial=financial,
                )
                for dossier in dossiers:
                    upsert_witness_card(
                        run_id=run_id,
                        witness_name=dossier.get('witness_name', 'Unknown'),
                        credibility_score=dossier.get('credibility_score', 0.5),
                        impeachment_points=json.dumps(dossier.get('impeachment_points', [])),
                        financial_interest=json.dumps(dossier.get('financial_interest', {})),
                        prior_inconsistencies=json.dumps(dossier.get('prior_inconsistencies', [])),
                        public_record_flags=json.dumps(dossier.get('public_record_flags', [])),
                        deposition_order=dossier.get('recommended_deposition_order', 99),
                        key_questions=json.dumps(dossier.get('deposition_key_questions', [])),
                        vulnerability_summary=dossier.get('vulnerability_summary'),
                    )
                increment_ci_run_cost(run_id, 0.03 * max(len(dossiers), 1))
                results['witnesses'] = dossiers
                logger.info(f"CI run {run_id}: Phase 2W complete ({len(dossiers)} dossiers)")
            except Exception as e:
                logger.warning(f"Phase 2W witness intelligence failed: {e}")

        def run_war_room():
            if cancel_event.is_set():
                return
            if not self.budget_manager.check_and_charge(run_id, 'war_room_strategy', 0.060):
                return
            try:
                wr_result = self.war_room.run_war_room(
                    run_id=run_id, role=role, goal_text=goal,
                    jurisdiction=jurisdiction_name,
                    entities=entities, timeline=timeline, financial=financial,
                    contradictions=contradictions, theories=theories,
                    documents=documents,
                )
                upsert_war_room(
                    run_id=run_id,
                    opposing_case_summary=wr_result.get('opposing_case_summary'),
                    top_dangerous_arguments=json.dumps(
                        wr_result.get('top_3_dangerous_arguments', [])),
                    client_vulnerabilities=json.dumps(
                        wr_result.get('client_vulnerabilities', [])),
                    smoking_guns=json.dumps(wr_result.get('smoking_guns_against_client', [])),
                    settlement_analysis=json.dumps(wr_result.get('settlement_analysis', {})),
                    likelihood_pct=wr_result.get('likelihood_of_success_pct', 50),
                    war_room_memo=wr_result.get('war_room_memo'),
                    opposing_counsel_checklist=json.dumps(
                        wr_result.get('opposing_counsel_checklist', [])),
                )
                increment_ci_run_cost(run_id, 0.060)
                results['war_room'] = wr_result
                logger.info(f"CI run {run_id}: Phase 2R complete")
            except Exception as e:
                logger.warning(f"Phase 2R war room failed: {e}")

        self._set_status(run_id, 'running',
                         stage='Phase 2W/2R: Witness intelligence + War room (parallel)',
                         progress=88)
        with ThreadPoolExecutor(max_workers=2,
                                thread_name_prefix='ci-specialist') as executor:
            futures = [
                executor.submit(run_witnesses),
                executor.submit(run_war_room),
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.warning(f"Specialist phase future failed: {e}")

        return results


    def _extract_financial_facts(self, run_id: str) -> List[Dict]:
        """Extract financial facts from manager reports for specialist phases."""
        try:
            mgr_reports = get_manager_reports(run_id)
            fin = next((r for r in mgr_reports
                        if r.get('manager_id') == 'financial'
                        or r.get('domain') == 'financial'), None)
            if fin and fin.get('report_json'):
                findings = json.loads(fin['report_json'] or '[]')
                result = []
                for f in findings:
                    if isinstance(f, dict) and f.get('content'):
                        # Parse doc_id from source field (e.g. "doc_id=1019")
                        src = f.get('source', '')
                        doc_id = None
                        if 'doc_id=' in src:
                            try:
                                doc_id = int(src.split('doc_id=')[1].split(',')[0].strip())
                            except (ValueError, IndexError):
                                pass
                        result.append({'description': f['content'],
                                       'source': src,
                                       'doc_id': doc_id,
                                       'paperless_doc_id': doc_id})
                return result
        except Exception:
            pass
        return []

    # -----------------------------------------------------------------------
    # Phase 3A: Senior Partner Review (Tier 4+)
    # -----------------------------------------------------------------------

