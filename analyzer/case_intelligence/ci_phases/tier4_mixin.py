"""CI Tier 4 — Senior Partner review phase

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


class Tier4PhaseMixin:
    """CI Tier 4 — Senior Partner review phase"""

    def _phase_senior_partner_review(self, run_id: str, run):
        """
        Phase 3A: Senior partner challenges the D2 analysis.
        Stores result as senior_partner_notes in ci_war_room.
        """
        if not self.budget_manager.check_and_charge(run_id, 'senior_partner_review', 0.030):
            return

        try:
            self._set_status(run_id, 'running',
                             stage='Phase 3A: Senior partner review', progress=95)
            # Build analysis summary from D2 findings
            run_obj = get_ci_run(run_id)
            findings_summary = run_obj.get('findings_summary') or ''
            theories = get_ci_theories(run_id)
            contradictions = get_ci_contradictions(run_id)

            analysis_summary = f"""FINDINGS SUMMARY:
{findings_summary[:4000]}

THEORIES ({len(theories)}):
{chr(10).join(f"- [{t['theory_type']}] {t['theory_text'][:150]} (confidence {t.get('confidence',0):.0%})" for t in theories[:8])}

CONTRADICTIONS ({len(contradictions)}):
{chr(10).join(f"- [{c['severity']}] {c['description']}" for c in contradictions[:8])}
"""

            review = self.war_room.run_senior_partner_review(
                run_id=run_id,
                role=run.get('role', 'neutral'),
                goal_text=run.get('goal_text') or '',
                analysis_summary=analysis_summary,
            )

            if review:
                senior_notes = review.get('senior_partner_notes', '')
                missed = review.get('missed_issues', [])
                most_important = review.get('single_most_important_finding', '')
                if missed or most_important:
                    notes_text = (
                        f"**Senior Partner Review**\n\n"
                        f"**Most Important Finding:** {most_important}\n\n"
                        f"**Missed Issues:**\n" +
                        '\n'.join(f"- {m['issue']}: {m.get('why_significant', '')}"
                                  for m in missed[:5]) +
                        f"\n\n{senior_notes}"
                    )
                    update_war_room_senior_notes(run_id, notes_text)
                increment_ci_run_cost(run_id, 0.030)
                logger.info(f"CI run {run_id}: Phase 3A complete")
        except Exception as e:
            logger.warning(f"Phase 3A senior partner review failed: {e}")

    # -----------------------------------------------------------------------
    # Phase 3B: White Glove (Tier 5)
    # -----------------------------------------------------------------------

