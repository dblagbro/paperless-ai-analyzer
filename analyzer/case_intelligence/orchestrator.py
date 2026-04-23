"""
CI Orchestrator — Director/Manager/Worker hierarchical run controller.

Architecture (v3.7.2 5-tier):
  Director D1: reads run config + doc list → produces manager_plan JSON
  Phase 1M:   Entity merge pass (all tiers)
  Manager (N parallel, one per domain): splits docs → spawns Workers → aggregates
  Phase 2F:   Forensic Accounting (Tier 3+)
  Phase 2D:   Discovery Gap Analysis (Tier 3+)
  Phase 2W:   Witness Intelligence (Tier 4+)
  Phase 2R:   War Room / Opposing Counsel (Tier 4+)
  Director D2: synthesizes all manager_reports → scientific paper report
  Phase 3A:   Senior Partner Review (Tier 4+)
  Phase 3B:   White Glove — Deep Forensics + Trial Strategy + Multi-Model (Tier 5)
  Director D3: Paperless write-back + marks run complete

Budget checkpoints fire at 50/70/80/90% and optionally invoke notification callbacks.
"""

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable

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
from analyzer.case_intelligence.budget_manager import BudgetManager
from analyzer.case_intelligence.entity_extractor import EntityExtractor
from analyzer.case_intelligence.timeline_builder import TimelineBuilder
from analyzer.case_intelligence.financial_extractor import FinancialExtractor
from analyzer.case_intelligence.contradiction_engine import ContradictionEngine
from analyzer.case_intelligence.theory_planner import TheoryPlanner, AdversarialTester
from analyzer.case_intelligence.jurisdiction import JurisdictionProfile
from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

DOMAINS = ['entities', 'timeline', 'financial', 'contradictions', 'theories', 'authorities']

# ---------------------------------------------------------------------------
# Director prompts
# ---------------------------------------------------------------------------

DIRECTOR_D1_PROMPT = """You are the Director AI of a legal Case Intelligence system.

Case: {case_name}
Role: {role}
Goal: {goal_text}
Jurisdiction: {jurisdiction}
Documents available: {doc_count}
Document IDs: {doc_ids_sample}
Budget remaining: ${budget:.4f}
Manager count: {manager_count}

Your task: produce a JSON manager_plan that assigns document analysis work to
{manager_count} domain managers.

Domains available: entities, timeline, financial, contradictions, theories, authorities

Return ONLY valid JSON:
{{
  "managers": [
    {{
      "domain": "entities",
      "doc_ids": [list of doc IDs to analyze],
      "instructions": "Focus on persons, organizations, key dates, roles in the dispute"
    }},
    ...
  ],
  "director_notes": "Brief strategic note about case analysis approach"
}}

Assign ALL document IDs across managers. A doc ID may appear in multiple domains.
Contradictions/theories managers should get ALL doc IDs for cross-document analysis.
"""

QUESTIONS_GENERATION_PROMPT = """You are a legal analyst preparing to analyze documents for a case.

Goal: {goal_text}
Role: {role}
Jurisdiction: {jurisdiction}
Documents: {doc_count}

Generate 3-5 clarifying questions that would significantly improve analysis quality.
Mark the most important as required.

Respond in JSON:
{{
  "questions": [
    {{"question": "Question text", "is_required": true, "purpose": "Why this matters"}}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


from analyzer.case_intelligence.ci_phases.directors_mixin import DirectorPhasesMixin
from analyzer.case_intelligence.ci_phases.managers_mixin import ManagerPhasesMixin
from analyzer.case_intelligence.ci_phases.specialist_mixin import SpecialistPhasesMixin
from analyzer.case_intelligence.ci_phases.tier4_mixin import Tier4PhaseMixin
from analyzer.case_intelligence.ci_phases.tier5_mixin import Tier5PhaseMixin
from analyzer.case_intelligence.ci_phases.writeback_mixin import WritebackPhasesMixin
from analyzer.case_intelligence.ci_phases.utils_mixin import OrchestratorUtilsMixin

class CIOrchestrator(
    DirectorPhasesMixin,
    ManagerPhasesMixin,
    SpecialistPhasesMixin,
    Tier4PhaseMixin,
    Tier5PhaseMixin,
    WritebackPhasesMixin,
    OrchestratorUtilsMixin,
):
    """
    Hierarchical CI run lifecycle: Director → Managers → Workers.
    Called by CIJobManager in a background thread.

    Implementation is split across mixin files in `ci_phases/` for maintainability.
    """

    def __init__(self, llm_clients: dict, paperless_client=None,
                 usage_tracker=None, cohere_api_key: str = None,
                 budget_notification_cb: Callable = None,
                 completion_notification_cb: Callable = None):
        """
        Args:
            llm_clients: dict mapping 'openai'/'anthropic' → LLMClient
            paperless_client: PaperlessClient for fetching documents + write-back
            usage_tracker: LLMUsageTracker
            cohere_api_key: For authority corpus embeddings
            budget_notification_cb: fn(run_id, pct, cost_so_far, projected, budget, status)
            completion_notification_cb: fn(run_id)
        """
        self.llm_clients = llm_clients
        self.paperless_client = paperless_client
        self.usage_tracker = usage_tracker
        self.cohere_api_key = cohere_api_key
        self.budget_notification_cb = budget_notification_cb
        self.completion_notification_cb = completion_notification_cb
        self.budget_manager = BudgetManager()

        # Per-run token accumulation (reset in execute_run)
        self._token_lock = threading.Lock()
        self._tokens_in = 0
        self._tokens_out = 0

        # Extractors (used by workers)
        self.entity_extractor = EntityExtractor(llm_clients, usage_tracker)
        self.timeline_builder = TimelineBuilder(llm_clients, usage_tracker)
        self.financial_extractor = FinancialExtractor(llm_clients, usage_tracker)
        self.contradiction_engine = ContradictionEngine(llm_clients, usage_tracker)
        self.theory_planner = TheoryPlanner(llm_clients, usage_tracker)
        self.adversarial_tester = AdversarialTester(llm_clients, usage_tracker)

        # v3.6.6 specialist modules (lazy-imported, tier-gated)
        from analyzer.case_intelligence.entity_merger import EntityMerger
        from analyzer.case_intelligence.forensic_accountant import ForensicAccountant
        from analyzer.case_intelligence.discovery_analyst import DiscoveryAnalyst
        from analyzer.case_intelligence.witness_analyst import WitnessAnalyst
        from analyzer.case_intelligence.war_room import WarRoom, TrialStrategist, SettlementValuator
        self.entity_merger = EntityMerger(llm_clients, usage_tracker)
        self.forensic_accountant = ForensicAccountant(llm_clients, usage_tracker)
        self.discovery_analyst = DiscoveryAnalyst(llm_clients, usage_tracker)
        self.witness_analyst = WitnessAnalyst(llm_clients, usage_tracker)
        self.war_room = WarRoom(llm_clients, usage_tracker)
        self.trial_strategist = TrialStrategist(llm_clients, usage_tracker)
        self.settlement_valuator = SettlementValuator(llm_clients, usage_tracker)

        # v3.7.2 Tier 5 modules
        from analyzer.case_intelligence.deep_financial_forensics import DeepFinancialForensics
        from analyzer.case_intelligence.multi_model_synthesis import MultiModelSynthesis
        self.deep_forensics = DeepFinancialForensics(llm_clients, usage_tracker)
        self.multi_model = MultiModelSynthesis(llm_clients, usage_tracker)

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------


    def execute_run(self, run_id: str, cancel_event: threading.Event = None):
        """Main entry — called from CIJobManager background thread."""
        if cancel_event is None:
            cancel_event = threading.Event()

        try:
            with self._token_lock:
                self._tokens_in = 0
                self._tokens_out = 0
            self._set_status(run_id, 'running', stage='Starting', progress=1)
            logger.info(f"CI run {run_id} starting (hierarchical mode)")

            run = get_ci_run(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found in database")

            # ── Phase Q: Clarifying questions ──────────────────────────────
            if not self._is_cancelled(cancel_event, run_id):
                self._phase_questions(run_id, run)

            run = get_ci_run(run_id)  # re-fetch after questions
            questions = get_ci_questions(run_id)
            unanswered_required = [
                q for q in questions if q['is_required'] and not q['answer']
            ]
            if unanswered_required and not run['proceed_with_assumptions']:
                self._set_status(run_id, 'running',
                                 stage='Awaiting answers to clarifying questions',
                                 progress=10)
                logger.info(f"CI run {run_id}: waiting for {len(unanswered_required)} answers")
                return

            # ── Phase D1: Director plans manager assignments ───────────────
            if self._is_cancelled(cancel_event, run_id):
                return

            self._set_status(run_id, 'running', stage='Director planning', progress=12)
            documents = self._fetch_case_documents(run)
            total_docs = len(documents)
            update_ci_run(run_id, docs_total=total_docs)

            if total_docs == 0:
                logger.warning(f"CI run {run_id}: no documents found")
                self._finalize_empty(run_id)
                return

            # Determine manager/worker counts from run config (auto if None)
            manager_count = self._resolve_manager_count(run, total_docs)
            workers_per_mgr = self._resolve_workers_per_manager(run, manager_count)

            logger.info(f"CI run {run_id}: {total_docs} docs, "
                        f"{manager_count} managers, {workers_per_mgr} workers/mgr")

            manager_plan = self._director_d1_plan(
                run_id, run, documents, manager_count
            )

            # ── Phase Managers: parallel domain analysis ───────────────────
            if self._is_cancelled(cancel_event, run_id):
                return

            self._set_status(run_id, 'running', stage='Managers running (parallel)', progress=20)
            manager_reports = self._run_all_managers(
                run_id, run, manager_plan, documents, workers_per_mgr, cancel_event
            )

            # ── Phase 2F/2D/2W/2R: Specialist analysis (tier-gated) ───────────
            run = get_ci_run(run_id)  # re-fetch to get max_tier
            max_tier = run.get('max_tier', 3)
            specialist_results = {}
            if not self._is_cancelled(cancel_event, run_id):
                specialist_results = self._run_specialist_phases(
                    run_id, run, documents, max_tier, cancel_event
                )

            # ── Phase D2: Director synthesizes report ──────────────────────
            if self._is_cancelled(cancel_event, run_id):
                return

            self._set_status(run_id, 'running', stage='Director synthesizing report', progress=90)
            self._director_d2_synthesize(run_id, run, manager_reports,
                                          [d.get('id', 0) for d in documents])

            # ── Phase 3A: Senior Partner Review (Tier 4+) ──────────────────
            if not self._is_cancelled(cancel_event, run_id) and max_tier >= 4:
                self._phase_senior_partner_review(run_id, run)

            # ── Phase 3B: White Glove (Tier 5) ─────────────────────────────
            if not self._is_cancelled(cancel_event, run_id) and max_tier >= 5:
                self._phase_3b_tier5(run_id, run, documents, specialist_results,
                                     cancel_event)

            # ── Phase D3: Write-back + finalize ────────────────────────────
            if not self._is_cancelled(cancel_event, run_id):
                self._set_status(run_id, 'running', stage='Writing back to Paperless', progress=96)
                try:
                    self._paperless_writeback(run_id, run)
                except Exception as e:
                    logger.warning(f"Paperless write-back failed: {e}")

                with self._token_lock:
                    ti, to = self._tokens_in, self._tokens_out
                update_ci_run(run_id, status='completed',
                              completed_at=datetime.now(timezone.utc).isoformat(),
                              current_stage='Completed',
                              progress_pct=100,
                              active_managers=0, active_workers=0,
                              tokens_in=ti, tokens_out=to)
                logger.info(f"CI run {run_id} completed")

                # Embed CI findings into vector store for AI Chat + Director awareness
                try:
                    self._embed_ci_run_findings(run_id, get_ci_run(run_id))
                except Exception as _e:
                    logger.warning(f"CI findings embedding failed (non-fatal): {_e}")

                if self.completion_notification_cb:
                    try:
                        self.completion_notification_cb(run_id)
                    except Exception as e:
                        logger.warning(f"Completion notification failed: {e}")

        except Exception as e:
            logger.error(f"CI run {run_id} failed: {e}", exc_info=True)
            update_ci_run(run_id, status='failed',
                          error_message=str(e)[:1000],
                          current_stage='Failed')

    # -----------------------------------------------------------------------
    # Phase Q: Clarifying questions
    # -----------------------------------------------------------------------

