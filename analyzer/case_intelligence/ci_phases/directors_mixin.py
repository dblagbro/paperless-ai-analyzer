"""CI Director phases — D1 plan, Q clarifying questions, D2 synthesis

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


class DirectorPhasesMixin:
    """CI Director phases — D1 plan, Q clarifying questions, D2 synthesis"""

    def _phase_questions(self, run_id: str, run):
        """Generate clarifying questions (skip if already done)."""
        self._set_status(run_id, 'running', stage='Generating questions', progress=5)
        existing = get_ci_questions(run_id)
        if existing:
            return

        if not self.budget_manager.check_and_charge(run_id, 'generate_questions', 0.001):
            return

        jurisdiction = self._load_jurisdiction(run)
        jurisdiction_name = jurisdiction.display_name if jurisdiction else 'Not specified'
        doc_count = self._estimate_doc_count(run)

        prompt = QUESTIONS_GENERATION_PROMPT.format(
            goal_text=run['goal_text'] or 'Analyze all documents',
            role=run['role'],
            jurisdiction=jurisdiction_name,
            doc_count=doc_count,
        )
        result = self._call_llm_simple(prompt, 'openai', 'gpt-4o-mini', 1000,
                                        'ci:generate_questions')
        if not result:
            return
        try:
            data = json.loads(result)
            for q in data.get('questions', []):
                add_ci_question(run_id,
                                question=q.get('question', ''),
                                is_required=q.get('is_required', False))
            update_ci_run(run_id, questions_asked=len(data.get('questions', [])))
        except Exception as e:
            logger.warning(f"Question parse error: {e}")

    # -----------------------------------------------------------------------
    # Phase D1: Director plans manager assignments
    # -----------------------------------------------------------------------


    def _director_d1_plan(self, run_id: str, run, documents: list,
                           manager_count: int) -> List[Dict]:
        """
        Ask Director LLM to plan manager assignments.
        Falls back to a deterministic plan on failure.
        """
        doc_ids = [d.get('id', 0) for d in documents]
        run_obj = get_ci_run(run_id)
        budget_remaining = (run_obj['budget_per_run_usd'] or 1.0) - (run_obj['cost_so_far_usd'] or 0)
        jurisdiction = self._load_jurisdiction(run)

        # Check for prior CI findings on this project (Director D1 awareness)
        prior_ci_note = ''
        try:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=run.get('project_slug', 'default'))
            if vs.enabled and run.get('goal_text'):
                prior_hits = vs.search(query=run['goal_text'], n_results=15)
                ci_prior = [h for h in prior_hits
                            if h.get('document_type') == 'ci_finding']
                if ci_prior:
                    prior_ci_note = (
                        f"\nNote: Prior CI analysis on this project found "
                        f"{len(ci_prior)} relevant findings — "
                        f"focus on new angles not already covered.\n"
                    )
        except Exception:
            pass

        # Try LLM planning (best-effort)
        prompt = DIRECTOR_D1_PROMPT.format(
            case_name=run.get('case_name') or run.get('goal_text', 'Case')[:40],
            role=run['role'],
            goal_text=(run['goal_text'] or 'General analysis') + prior_ci_note,
            jurisdiction=jurisdiction.display_name if jurisdiction else 'Not specified',
            doc_count=len(doc_ids),
            doc_ids_sample=str(doc_ids[:20]),
            budget=budget_remaining,
            manager_count=manager_count,
        )

        llm_plan = None
        if self.budget_manager.check_and_charge(run_id, 'director_d1_plan', 0.005):
            result = self._call_llm_simple(prompt, 'openai', 'gpt-4o', 2000,
                                            'ci:director_d1_plan')
            if result:
                # Extract JSON from fenced block if present
                cleaned = result
                if '```json' in result:
                    cleaned = result.split('```json')[1].split('```')[0].strip()
                elif '```' in result:
                    cleaned = result.split('```')[1].split('```')[0].strip()
                try:
                    parsed = json.loads(cleaned)
                    if 'managers' in parsed and isinstance(parsed['managers'], list):
                        llm_plan = parsed['managers']
                        logger.info(f"CI run {run_id}: Director D1 plan has "
                                    f"{len(llm_plan)} managers")
                        if parsed.get('director_notes'):
                            update_ci_run(run_id,
                                         current_stage=f"Director: {parsed['director_notes'][:200]}")
                except Exception as e:
                    logger.warning(f"Director D1 plan parse error: {e}")

        if llm_plan:
            # Ensure all 6 domains are present
            covered = {m['domain'] for m in llm_plan}
            for domain in DOMAINS:
                if domain not in covered:
                    llm_plan.append({'domain': domain, 'doc_ids': doc_ids,
                                     'instructions': f'Analyze all documents for {domain}'})
            return llm_plan

        # Fallback: deterministic plan — all 6 domains, all docs each
        logger.info(f"CI run {run_id}: using deterministic Director D1 fallback")
        return [
            {'domain': 'entities',       'doc_ids': doc_ids,
             'instructions': 'Extract all named persons, organizations, places, and key identifiers'},
            {'domain': 'timeline',       'doc_ids': doc_ids,
             'instructions': 'Extract all dates, events, and chronological facts'},
            {'domain': 'financial',      'doc_ids': doc_ids,
             'instructions': 'Extract all financial figures, transactions, and monetary claims'},
            {'domain': 'contradictions', 'doc_ids': doc_ids,
             'instructions': 'Find contradictions, inconsistencies, and disputed facts across documents'},
            {'domain': 'theories',       'doc_ids': doc_ids,
             'instructions': 'Identify factual and legal theories supported by the evidence'},
            {'domain': 'authorities',    'doc_ids': doc_ids,
             'instructions': 'Identify legal authorities, statutes, cases cited or applicable'},
        ]

    # -----------------------------------------------------------------------
    # Phase Managers: parallel domain analysis
    # -----------------------------------------------------------------------


    def _director_d2_synthesize(self, run_id: str, run,
                                  manager_reports: List[Dict],
                                  doc_ids: List[int]):
        """Build and store the scientific paper report from manager findings."""
        try:
            from analyzer.case_intelligence.report_generator import generate_report
            case_name = (run.get('case_name') or
                         (run['goal_text'] or 'Case')[:60] if run else 'Case')
            report_md = generate_report(
                manager_reports=manager_reports,
                case_name=case_name,
                doc_ids=doc_ids,
                run_id=run_id,
            )
            update_ci_run(run_id, findings_summary=report_md)
            logger.info(f"CI run {run_id}: Director D2 report generated "
                        f"({len(report_md)} chars)")
        except Exception as e:
            logger.warning(f"Director D2 report generation failed: {e}")
            # Fallback: legacy findings summary
            try:
                summary = self._legacy_findings_summary(run_id, run)
                if summary:
                    update_ci_run(run_id, findings_summary=summary)
            except Exception as e2:
                logger.warning(f"Legacy findings summary also failed: {e2}")

