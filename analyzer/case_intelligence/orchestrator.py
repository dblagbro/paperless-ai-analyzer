"""
CI Orchestrator — Director/Manager/Worker hierarchical run controller.

Architecture (v3.6.6 5-tier):
  Director D1: reads run config + doc list → produces manager_plan JSON
  Phase 1M:   Entity merge pass (all tiers)
  Manager (N parallel, one per domain): splits docs → spawns Workers → aggregates
  Phase 2F:   Forensic Accounting (Tier 3+)
  Phase 2D:   Discovery Gap Analysis (Tier 3+)
  Phase 2W:   Witness Intelligence (Tier 4+)
  Phase 2R:   War Room / Opposing Counsel (Tier 4+)
  Director D2: synthesizes all manager_reports → scientific paper report
  Phase 3A:   Senior Partner Review (Tier 4+)
  Director D3: Paperless write-back + marks run complete

Budget checkpoints fire every 10% and optionally invoke notification callbacks.
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

class CIOrchestrator:
    """
    Hierarchical CI run lifecycle: Director → Managers → Workers.
    Called by CIJobManager in a background thread.
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
        from analyzer.case_intelligence.war_room import WarRoom
        self.entity_merger = EntityMerger(llm_clients, usage_tracker)
        self.forensic_accountant = ForensicAccountant(llm_clients, usage_tracker)
        self.discovery_analyst = DiscoveryAnalyst(llm_clients, usage_tracker)
        self.witness_analyst = WitnessAnalyst(llm_clients, usage_tracker)
        self.war_room = WarRoom(llm_clients, usage_tracker)

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

    def _run_all_managers(self, run_id: str, run, manager_plan: List[Dict],
                           all_docs: list, workers_per_mgr: int,
                           cancel_event: threading.Event) -> List[Dict]:
        """
        Run managers in two sequential phases so cross-document analysis has data:

        Phase 1 (parallel): entities, timeline, financial — per-document extraction
        Phase 2 (parallel): contradictions, theories, authorities — reads Phase 1 results
        """
        doc_map: Dict[int, Dict] = {d.get('id', 0): d for d in all_docs}
        manager_reports: List[Dict] = []
        lock = threading.Lock()

        # Split plan into the two phases
        phase1_specs = [s for s in manager_plan
                        if s.get('domain') in ('entities', 'timeline', 'financial')]
        phase2_specs = [s for s in manager_plan
                        if s.get('domain') in ('contradictions', 'theories', 'authorities')]

        total_p1 = len(phase1_specs)
        total_p2 = len(phase2_specs)
        completed_p1 = [0]
        completed_p2 = [0]

        def run_manager_task(manager_spec):
            domain = manager_spec.get('domain', 'unknown')
            doc_ids_for_manager = manager_spec.get('doc_ids', [])
            instructions = manager_spec.get('instructions', '')
            spec_case_context = manager_spec.get('case_context')

            if cancel_event.is_set():
                return None

            logger.info(f"CI run {run_id}: Manager [{domain}] starting "
                        f"({len(doc_ids_for_manager)} docs)")
            upsert_manager_report(run_id, domain, status='running',
                                   docs_assigned=len(doc_ids_for_manager),
                                   started_at=datetime.now(timezone.utc).isoformat())
            try:
                report = self._run_manager(
                    run_id, run, domain, doc_ids_for_manager,
                    instructions, doc_map, workers_per_mgr, cancel_event,
                    case_context=spec_case_context,
                )
                upsert_manager_report(
                    run_id, domain, status='completed',
                    report_json=json.dumps(report.get('findings', [])),
                    worker_count=report.get('worker_count', 0),
                    docs_assigned=len(doc_ids_for_manager),
                    cost_usd=report.get('cost_usd', 0),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                logger.info(f"CI run {run_id}: Manager [{domain}] done — "
                            f"{len(report.get('findings', []))} findings")
                return report
            except Exception as e:
                logger.error(f"Manager [{domain}] failed: {e}", exc_info=True)
                upsert_manager_report(run_id, domain, status='failed')
                return {'domain': domain, 'findings': [], 'cost_usd': 0}

        # ── Phase 1: extraction (parallel) ─────────────────────────────────
        if phase1_specs:
            self._set_status(run_id, 'running',
                             stage='Phase 1: Extracting entities, timeline, financial',
                             progress=20,
                             active_managers=total_p1,
                             active_workers=total_p1 * workers_per_mgr)
            with ThreadPoolExecutor(max_workers=min(total_p1, 3),
                                     thread_name_prefix='ci-manager-p1') as executor:
                futures = {
                    executor.submit(run_manager_task, spec): spec
                    for spec in phase1_specs
                }
                for future in as_completed(futures):
                    if cancel_event.is_set():
                        break
                    result = future.result()
                    if result:
                        with lock:
                            manager_reports.append(result)
                            completed_p1[0] += 1
                            pct = 20 + int(completed_p1[0] / max(total_p1, 1) * 40)
                            remaining_p1 = total_p1 - completed_p1[0]
                            with self._token_lock:
                                ti, to = self._tokens_in, self._tokens_out
                            self._set_status(
                                run_id, 'running',
                                stage=(f'Phase 1: {completed_p1[0]}/{total_p1} '
                                       f'extraction managers done'),
                                progress=pct,
                                active_managers=remaining_p1,
                                tokens_in=ti, tokens_out=to,
                            )
                            self._check_budget_checkpoint(run_id, pct)

        if cancel_event.is_set():
            return manager_reports

        # ── Phase 1M: Entity merge pass (all tiers) ──────────────────────
        try:
            self._set_status(run_id, 'running',
                             stage='Phase 1M: Merging duplicate entities', progress=60)
            merge_result = self.entity_merger.merge_run_entities(run_id)
            if merge_result.get('merged', 0):
                logger.info(
                    f"CI run {run_id}: Phase 1M merged {merge_result['merged']} duplicates, "
                    f"{merge_result['total']} active entities"
                )
        except Exception as e:
            logger.warning(f"Phase 1M entity merge failed (non-fatal): {e}")

        # ── Phase W: Web research (between extraction and synthesis) ──────
        self._phase_web_research(run_id, run, cancel_event)

        if cancel_event.is_set():
            return manager_reports

        # ── Phase 2: cross-document analysis (parallel, after extraction) ──
        if phase2_specs:
            # Build war room briefing from Phase 1 results
            case_context = self._build_case_context(run_id)
            logger.info(
                f"CI run {run_id}: war room briefing built — "
                f"{case_context['entity_count']} entities, {case_context['event_count']} events, "
                f"{len(case_context['financial'])} financial facts. "
                f"Starting Phase 2 (contradictions/theories/authorities)."
            )
            # Inject into Phase 2 specs so each manager gets the briefing
            for spec in phase2_specs:
                spec['case_context'] = case_context
            self._set_status(run_id, 'running',
                             stage='Phase 2: Contradictions, theories, authorities',
                             progress=62,
                             active_managers=total_p2,
                             active_workers=0)
            with ThreadPoolExecutor(max_workers=min(total_p2, 3),
                                     thread_name_prefix='ci-manager-p2') as executor:
                futures = {
                    executor.submit(run_manager_task, spec): spec
                    for spec in phase2_specs
                }
                for future in as_completed(futures):
                    if cancel_event.is_set():
                        break
                    result = future.result()
                    if result:
                        with lock:
                            manager_reports.append(result)
                            completed_p2[0] += 1
                            pct = 62 + int(completed_p2[0] / max(total_p2, 1) * 23)
                            remaining_p2 = total_p2 - completed_p2[0]
                            with self._token_lock:
                                ti, to = self._tokens_in, self._tokens_out
                            self._set_status(
                                run_id, 'running',
                                stage=(f'Phase 2: {completed_p2[0]}/{total_p2} '
                                       f'analysis managers done'),
                                progress=pct,
                                active_managers=remaining_p2,
                                tokens_in=ti, tokens_out=to,
                            )
                            self._check_budget_checkpoint(run_id, pct)

        return manager_reports

    # -----------------------------------------------------------------------
    # Phase W: Web Research
    # -----------------------------------------------------------------------

    def _phase_web_research(self, run_id: str, run,
                             cancel_event: threading.Event) -> None:
        """
        Phase W — Web Research.

        Runs between Phase 1 (extraction) and Phase 2 (synthesis).
        Searches free and optional paid sources for:
          1. Legal authorities / case law (injected into Phase 2 authorities manager)
          2. Entity background / character research (injected into theories prompt)
          3. General web developments (injected into findings summary)

        Skipped entirely if web_research_config is absent or not enabled.
        """
        try:
            wrc = json.loads(run.get('web_research_config') or '{}')
        except Exception:
            wrc = {}

        if not wrc.get('enabled'):
            return

        self._set_status(run_id, 'running',
                         stage='Phase W: Web research', progress=59)
        logger.info(f"CI run {run_id}: Phase W starting (web research)")

        from analyzer.case_intelligence.web_researcher import WebResearcher
        researcher = WebResearcher(wrc)

        # Jurisdiction display name
        jd_name = 'Not specified'
        try:
            jd_name = json.loads(run.get('jurisdiction_json') or '{}').get(
                'display_name', 'Not specified')
        except Exception:
            pass

        role = run.get('role', 'neutral')
        goal = run.get('goal_text') or ''

        # ── 1. Legal authority / case law search ────────────────────────────
        if wrc.get('legal_search', True) and goal:
            try:
                results = researcher.search_legal_authorities(
                    query=goal[:250],
                    jurisdiction=jd_name,
                    role=role,
                    max_results=10,
                )
                if results:
                    add_ci_web_research(
                        run_id=run_id,
                        search_type='legal_authority',
                        query=goal[:200],
                        source='web_research',
                        results_json=json.dumps(results),
                    )
                    logger.info(f"CI run {run_id}: Web authority search returned "
                                f"{len(results)} results")
            except Exception as e:
                logger.warning(f"CI Phase W legal authority search failed: {e}")

        if cancel_event.is_set():
            return

        # ── 2. Entity background / character research ────────────────────────
        if wrc.get('entity_research', True):
            entities = get_ci_entities(run_id)
            # Prioritise persons and orgs, limit to 12 to stay within API rate limits
            key_entities = [e for e in entities
                            if e['entity_type'] in ('person', 'org',
                                                     'individual', 'organization')][:12]
            for entity in key_entities:
                if cancel_event.is_set():
                    break
                try:
                    result = researcher.research_entity(
                        name=entity['name'],
                        entity_type=entity['entity_type'],
                        role_in_case=entity.get('role_in_case', ''),
                        run_role=role,
                    )
                    if result.get('court_history') or result.get('news_mentions'):
                        add_ci_web_research(
                            run_id=run_id,
                            search_type='entity_background',
                            query=entity['name'],
                            source='web_research',
                            results_json=json.dumps(result),
                            entity_name=entity['name'],
                        )
                except Exception as e:
                    logger.warning(
                        f"CI Phase W entity research failed for "
                        f"'{entity['name']}': {e}")

        if cancel_event.is_set():
            return

        # ── 3. General web search (recent developments, news) ───────────────
        if wrc.get('general_search', True) and goal:
            try:
                query = f"legal {goal[:150]} recent developments 2024 2025"
                results = researcher.search_general(query, max_results=5)
                if results:
                    add_ci_web_research(
                        run_id=run_id,
                        search_type='general',
                        query=query[:200],
                        source='web_search',
                        results_json=json.dumps(results),
                    )
            except Exception as e:
                logger.warning(f"CI Phase W general search failed: {e}")

        logger.info(f"CI run {run_id}: Phase W complete")

    def _build_case_context(self, run_id: str) -> dict:
        """
        Build a compact Phase 1 findings brief for injection into Phase 2 agents.
        This is the 'war room briefing' — all Phase 2 agents receive this before starting,
        giving them full awareness of what Phase 1 workers found without context exhaustion.
        """
        entities = [dict(e) for e in get_ci_entities(run_id)]
        events   = [dict(ev) for ev in get_ci_timeline(run_id)]
        financial_lines: List[str] = []
        try:
            mgr_reports = get_manager_reports(run_id)
            fin = next((r for r in mgr_reports
                        if r.get('manager_id') == 'financial'
                        or r.get('domain') == 'financial'), None)
            if fin and fin.get('report_json'):
                fin_findings = json.loads(fin['report_json'] or '[]')
                financial_lines = [f['content'] for f in fin_findings[:20]
                                   if isinstance(f, dict) and f.get('content')]
        except Exception:
            pass
        return {
            'entities':     [{'name': e['name'], 'type': e['entity_type'],
                               'role': e.get('role_in_case', '') or ''}
                              for e in entities[:40]],
            'timeline':     [{'date': ev.get('event_date'), 'event': ev['description'],
                               'sig': ev.get('significance', 'medium')}
                              for ev in events[:40]],
            'financial':    financial_lines,
            'entity_count': len(entities),
            'event_count':  len(events),
        }

    def _run_manager(self, run_id: str, run, domain: str,
                      doc_ids: List[int], instructions: str,
                      doc_map: Dict[int, Dict], workers_per_mgr: int,
                      cancel_event: threading.Event,
                      case_context: dict = None) -> Dict:
        """
        Run one domain manager: split docs into worker batches (per-doc extraction),
        then handle cross-doc analysis (contradictions, theories, authorities).
        """
        findings: List[Dict] = []
        total_cost = 0.0
        worker_count = 0

        # ── Per-document worker phase (for entity/timeline/financial domains) ──
        if domain in ('entities', 'timeline', 'financial'):
            batch_size = max(1, min(3, math.ceil(len(doc_ids) / max(workers_per_mgr, 1))))
            batches = [doc_ids[i:i+batch_size]
                       for i in range(0, len(doc_ids), batch_size)]

            with ThreadPoolExecutor(max_workers=min(workers_per_mgr, 10),
                                     thread_name_prefix=f'ci-worker-{domain}') as executor:
                worker_futures = {
                    executor.submit(self._run_worker, run_id, run, domain,
                                    batch, instructions, doc_map): batch
                    for batch in batches
                }
                for f in as_completed(worker_futures):
                    if cancel_event.is_set():
                        break
                    try:
                        w_result = f.result()
                        if w_result:
                            findings.extend(w_result.get('findings', []))
                            total_cost += w_result.get('cost_usd', 0)
                            worker_count += 1
                    except Exception as e:
                        logger.warning(f"Worker failed in manager [{domain}]: {e}")

        # ── Cross-document analysis (contradictions/theories/authorities) ────
        elif domain == 'contradictions':
            findings, total_cost = self._manager_contradictions(
                run_id, run, doc_ids, doc_map, cancel_event, case_context=case_context
            )
            worker_count = max(1, len(doc_ids))

        elif domain == 'theories':
            findings, total_cost = self._manager_theories(
                run_id, run, doc_ids, doc_map, cancel_event, case_context=case_context
            )
            worker_count = 1

        elif domain == 'authorities':
            findings, total_cost = self._manager_authorities(
                run_id, run, cancel_event
            )
            worker_count = 1

        return {
            'domain': domain,
            'findings': findings,
            'cost_usd': total_cost,
            'worker_count': worker_count,
        }

    def _run_worker(self, run_id: str, run, domain: str,
                    doc_ids: List[int], instructions: str,
                    doc_map: Dict[int, Dict]) -> Dict:
        """
        Process a batch of documents for a single domain.
        Returns {'findings': [...], 'cost_usd': float}.
        """
        findings: List[Dict] = []
        cost = 0.0

        for doc_id in doc_ids:
            doc = doc_map.get(doc_id)
            if not doc:
                continue

            title = doc.get('title', f'Document {doc_id}')
            content = (doc.get('content', '') or '').strip()

            if len(content) < 50:
                increment_ci_run_docs(run_id, 1)
                continue

            # Prepend vector store AI analysis context if available.
            # The standard analysis pipeline produces full-document summaries
            # (not truncated) — prepending them means extractors see the whole
            # document's key facts even when the raw OCR is thousands of chars long.
            vs_type  = doc.get('vs_document_type', '') or ''
            vs_brief = doc.get('vs_brief_summary',  '') or ''
            vs_full  = doc.get('vs_full_summary',   '') or ''
            enrichment_parts = []
            if vs_type and vs_type not in ('unknown', 'other'):
                enrichment_parts.append(f"Document Type: {vs_type}")
            if vs_brief:
                enrichment_parts.append(f"AI Summary: {vs_brief}")
            if vs_full and vs_full != vs_brief:
                enrichment_parts.append(f"Detailed Analysis: {vs_full}")
            if enrichment_parts:
                content = (
                    '[PRIOR AI ANALYSIS — use as additional context for extraction]\n'
                    + '\n'.join(enrichment_parts)
                    + '\n\n[RAW DOCUMENT TEXT]\n'
                    + content
                )

            try:
                if domain == 'entities':
                    entities = self.entity_extractor.extract(
                        doc_id, title, content, run_id
                    )
                    if entities:
                        cost += self.entity_extractor.task_def.estimated_cost_per_doc
                    for ent in entities:
                        upsert_ci_entity(
                            run_id=run_id,
                            entity_type=ent.get('entity_type', 'person'),
                            name=ent.get('name', ''),
                            aliases=json.dumps(ent.get('aliases', [])),
                            role_in_case=ent.get('role_in_case', ''),
                            attributes=json.dumps(ent.get('attributes', {})),
                            notes=ent.get('notes', ''),
                            provenance=json.dumps(ent.get('provenance', [])),
                        )
                        findings.append({
                            'content': f"{ent.get('entity_type','entity')}: {ent.get('name','')}",
                            'source': f'doc_id={doc_id}',
                            'confidence': 'high',
                        })

                elif domain == 'timeline':
                    events = self.timeline_builder.extract(
                        doc_id, title, content, run_id
                    )
                    if events:
                        cost += self.timeline_builder.task_def.estimated_cost_per_doc
                    for ev in events:
                        add_ci_event(
                            run_id=run_id,
                            description=ev.get('description', ''),
                            event_date=ev.get('event_date'),
                            date_approx=bool(ev.get('date_approx', False)),
                            event_type=ev.get('event_type', 'other'),
                            significance=ev.get('significance', 'medium'),
                            parties=json.dumps(ev.get('parties', [])),
                            provenance=json.dumps(ev.get('provenance', [])),
                        )
                        findings.append({
                            'content': (
                                f"{ev.get('event_date','unknown')}: "
                                f"{ev.get('description','')[:120]}"
                            ),
                            'source': f'doc_id={doc_id}',
                            'confidence': (
                                'high' if ev.get('significance') in ('critical', 'high')
                                else 'medium'
                            ),
                        })

                elif domain == 'financial':
                    result = self.financial_extractor.extract(
                        doc_id, title, content, run_id
                    )
                    facts = result.get('financial_facts', [])
                    if facts:
                        cost += self.financial_extractor.task_def.estimated_cost_per_doc
                    for fact in facts:
                        amt = fact.get('amount_usd') or fact.get('amount')
                        label = fact.get('description') or fact.get('label', '')
                        findings.append({
                            'content': f"{label}: {amt}" if amt else str(label),
                            'source': f'doc_id={doc_id}',
                            'confidence': 'medium',
                        })

            except Exception as e:
                logger.warning(f"Worker [{domain}] error on doc {doc_id}: {e}")

            increment_ci_run_docs(run_id, 1)
            if cost > 0:
                increment_ci_run_cost(run_id, cost)
                cost = 0.0

        return {'findings': findings, 'cost_usd': cost}

    def _manager_contradictions(self, run_id: str, run, doc_ids: List[int],
                                  doc_map: Dict[int, Dict],
                                  cancel_event: threading.Event,
                                  case_context: dict = None) -> tuple:
        """Contradiction and disputed-facts detection across docs."""
        from collections import defaultdict
        findings: List[Dict] = []
        cost = 0.0

        # Load Phase 1 entities and events from DB, grouped by doc
        all_entities_db = [dict(e) for e in get_ci_entities(run_id)]
        all_events_db   = [dict(ev) for ev in get_ci_timeline(run_id)]
        ents_by_doc: Dict[int, list] = defaultdict(list)
        evts_by_doc: Dict[int, list] = defaultdict(list)
        for e in all_entities_db:
            for p in (json.loads(e.get('provenance', '[]') or '[]'))[:1]:
                doc_id_p = p.get('paperless_doc_id') or p.get('doc_id')
                if doc_id_p is not None:
                    ents_by_doc[int(doc_id_p)].append(
                        {'name': e['name'], 'type': e['entity_type']}
                    )
        for ev in all_events_db:
            for p in (json.loads(ev.get('provenance', '[]') or '[]'))[:1]:
                doc_id_p = p.get('paperless_doc_id') or p.get('doc_id')
                if doc_id_p is not None:
                    evts_by_doc[int(doc_id_p)].append(
                        {'date': ev.get('event_date'), 'desc': ev['description']}
                    )

        documents = [
            {'doc_id':   doc_map[did].get('id', did),
             'title':    doc_map[did].get('title', f'Doc {did}'),
             'content':  doc_map[did].get('content', '') or '',
             'entities': ents_by_doc.get(did, []),
             'events':   evts_by_doc.get(did, []),
             'financial_facts': []}
            for did in doc_ids if did in doc_map
        ]
        if len(documents) < 2:
            return findings, cost

        if self.budget_manager.check_and_charge(run_id, 'contradiction_detection', 0.015):
            try:
                result = self.contradiction_engine.detect_contradictions(
                    documents=documents, role=run['role'],
                    goal_text=run['goal_text'] or ''
                )
                for c in result.get('contradictions', []):
                    add_ci_contradiction(
                        run_id=run_id,
                        description=c.get('description', ''),
                        severity=c.get('severity', 'medium'),
                        doc_a_provenance=c.get('doc_a_provenance', '[]'),
                        doc_b_provenance=c.get('doc_b_provenance', '[]'),
                        contradiction_type=c.get('contradiction_type'),
                        explanation=c.get('explanation'),
                        suggested_action=c.get('suggested_action'),
                    )
                    prov = c.get('doc_a_provenance', '')
                    findings.append({
                        'content': c.get('description', ''),
                        'source': str(prov)[:100],
                        'confidence': 'high' if c.get('severity') in ('high', 'critical') else 'medium',
                    })
                cost += 0.015
            except Exception as e:
                logger.warning(f"Contradiction detection error: {e}")

        if self.budget_manager.check_and_charge(run_id, 'disputed_facts_matrix', 0.010):
            try:
                facts = self.contradiction_engine.build_disputed_facts_matrix(
                    documents=documents, role=run['role']
                )
                for fact in facts:
                    add_ci_disputed_fact(
                        run_id=run_id,
                        fact_description=fact.get('fact_description', ''),
                        position_a_label=fact.get('position_a_label'),
                        position_a_text=fact.get('position_a_text'),
                        position_b_label=fact.get('position_b_label'),
                        position_b_text=fact.get('position_b_text'),
                        supporting_evidence_a=json.dumps(fact.get('supporting_evidence_a', [])),
                        supporting_evidence_b=json.dumps(fact.get('supporting_evidence_b', [])),
                    )
                cost += 0.010
            except Exception as e:
                logger.warning(f"Disputed facts matrix error: {e}")

        return findings, cost

    def _manager_theories(self, run_id: str, run, doc_ids: List[int],
                            doc_map: Dict[int, Dict],
                            cancel_event: threading.Event,
                            case_context: dict = None) -> tuple:
        """Theory generation and adversarial testing."""
        # Ensure run is a dict (sqlite3.Row doesn't support .get())
        if not isinstance(run, dict):
            run = dict(run)
        findings: List[Dict] = []
        cost = 0.0

        entities = get_ci_entities(run_id)
        events = get_ci_timeline(run_id)
        contradictions = get_ci_contradictions(run_id)
        authorities_list = get_ci_authorities(run_id)

        def _first_doc_id(provenance_json):
            """Return first paperless_doc_id from a provenance JSON string, or None."""
            try:
                items = json.loads(provenance_json or '[]')
                for p in items:
                    did = p.get('paperless_doc_id')
                    if did:
                        return int(did)
            except Exception:
                pass
            return None

        entities_summary = '\n'.join(
            f"- [{e['entity_type']}] {e['name']}: {e['role_in_case'] or ''}"
            + (f" [Doc #{_first_doc_id(e['provenance'])}]"
               if _first_doc_id(e['provenance']) else '')
            for e in entities[:20]
        )
        timeline_summary = '\n'.join(
            f"- {ev['event_date'] or 'unknown'} [{ev['significance']}]: {ev['description']}"
            + (f" [Doc #{_first_doc_id(ev['provenance'])}]"
               if _first_doc_id(ev['provenance']) else '')
            for ev in events[:30]
        )
        contradictions_summary = '\n'.join(
            f"- [{c['severity']}] {c['description']}" for c in contradictions[:10]
        )
        authorities_summary = '\n'.join(
            f"- {a['citation']}: {a['relevance_note'] or ''}"
            for a in authorities_list[:10]
        )

        # Augment entities_summary with web research background (Phase W)
        try:
            web_entity_rows = get_ci_web_research(run_id, search_type='entity_background')
            web_bg_lines = []
            for wr in web_entity_rows:
                r = json.loads(wr.get('results_json') or '{}')
                if r.get('summary'):
                    web_bg_lines.append(
                        f"- {r.get('name', wr.get('entity_name', '?'))} "
                        f"[web background]: {r['summary']}"
                    )
            if web_bg_lines:
                entities_summary += (
                    '\n\nWEB RESEARCH — ENTITY BACKGROUND:\n'
                    + '\n'.join(web_bg_lines[:10])
                )
        except Exception:
            pass

        documents = [doc_map[did] for did in doc_ids if did in doc_map]

        # Use war room financial data if available (avoids LLM re-extraction)
        financial_lines = []
        if case_context and case_context.get('financial'):
            financial_lines = [f"- {f}" for f in case_context['financial']]
            # Also override entities_summary with war room brief (always freshest data)
            ctx_entities = '\n'.join(
                f"- [{e['type']}] {e['name']}: {e['role']}"
                + (f" [Doc #{e['doc_id']}]" if e.get('doc_id') else '')
                for e in case_context['entities'][:40]
            )
            if ctx_entities:
                entities_summary = ctx_entities
        else:
            # Fallback: re-extract financial from docs (legacy path)
            for doc in documents[:5]:
                fin = self.financial_extractor.extract(
                    doc.get('id', 0), doc.get('title', ''),
                    (doc.get('content', '') or '')[:3000], run_id
                ) if doc.get('content') else {}
                for ff in fin.get('financial_facts', [])[:3]:
                    financial_lines.append(f"- {ff.get('description', '')}: {ff.get('amount_raw', '')}")

        jurisdiction_name = 'Not specified'
        try:
            jd = json.loads(run.get('jurisdiction_json') or '{}')
            jurisdiction_name = jd.get('display_name', 'Not specified')
        except Exception:
            pass

        if not self.budget_manager.check_and_charge(run_id, 'theory_generation', 0.05):
            return findings, cost

        # Build web research summary for theories
        web_research_summary = ''
        try:
            web_rows = get_ci_web_research(run_id)
            web_lines = []
            for wr in web_rows[:5]:
                results_data = json.loads(wr.get('results_json') or '[]')
                if isinstance(results_data, dict) and results_data.get('summary'):
                    web_lines.append(f"- {results_data['summary'][:200]}")
                elif isinstance(results_data, list):
                    for item in results_data[:2]:
                        if isinstance(item, dict) and item.get('title'):
                            web_lines.append(f"- {item.get('title', '')}: {item.get('excerpt', '')[:150]}")
            web_research_summary = '\n'.join(web_lines)
        except Exception:
            pass

        try:
            theories = self.theory_planner.generate_theories(
                role=run['role'],
                goal_text=run['goal_text'] or '',
                jurisdiction=jurisdiction_name,
                entities_summary=entities_summary,
                timeline_summary=timeline_summary,
                financial_summary='\n'.join(financial_lines) or 'None',
                contradictions_summary=contradictions_summary,
                authorities_summary=authorities_summary,
                web_research_summary=web_research_summary,
                run_id=run_id,
            )
            cost += 0.05
        except Exception as e:
            logger.warning(f"Theory generation error: {e}")
            return findings, cost

        docs_for_adversarial = documents[:5]
        for theory in theories:
            if cancel_event.is_set():
                break
            theory_id = add_ci_theory(
                run_id=run_id,
                theory_text=theory.get('theory_text', ''),
                theory_type=theory.get('theory_type', 'factual'),
                role_perspective=theory.get('role_perspective', run['role']),
                confidence=theory.get('confidence', 0.5),
                supporting_evidence=json.dumps(theory.get('supporting_evidence', [])),
            )
            findings.append({
                'content': theory.get('theory_text', ''),
                'source': f'theory_id={theory_id}',
                'confidence': 'high' if (theory.get('confidence') or 0) >= 0.7 else 'medium',
            })

            if self.budget_manager.check_and_charge(run_id, 'adversarial_testing', 0.05):
                try:
                    test_result = self.adversarial_tester.test_theory(
                        theory_text=theory.get('theory_text', ''),
                        confidence=theory.get('confidence', 0.5),
                        supporting_evidence=theory.get('supporting_evidence', []),
                        counter_docs=[{
                            'doc_id': d.get('id', 0), 'title': d.get('title', ''),
                            'content': d.get('content', ''), 'entities': [],
                            'events': [], 'financial_facts': []
                        } for d in docs_for_adversarial],
                        theory_id=theory_id,
                    )
                    update_ci_theory(
                        theory_id,
                        status=test_result.get('recommended_status', 'uncertain'),
                        confidence=test_result.get('revised_confidence', theory.get('confidence', 0.5)),
                        falsification_report=test_result.get('falsification_report', ''),
                        what_would_change=test_result.get('what_would_change', ''),
                        counter_evidence=json.dumps(test_result.get('counter_evidence', [])),
                    )
                    cost += 0.05
                except Exception as e:
                    logger.warning(f"Adversarial test error for theory {theory_id}: {e}")

        # Second pass: opposing perspective (budget utilization + opposing-side awareness)
        if not cancel_event.is_set():
            opposing_role_map = {
                'plaintiff': 'defense', 'defense': 'plaintiff',
                'prosecution': 'defense', 'defendant': 'plaintiff',
                'neutral': 'defense',
            }
            opposing_role = opposing_role_map.get(run['role'], 'defense')
            if self.budget_manager.check_and_charge(run_id, 'opposing_theory_generation', 0.05):
                try:
                    opp_theories = self.theory_planner.generate_theories(
                        role=opposing_role,
                        goal_text=(f"Build the strongest case for {opposing_role} against: "
                                   f"{run['goal_text'] or ''}"),
                        jurisdiction=jurisdiction_name,
                        entities_summary=entities_summary,
                        timeline_summary=timeline_summary,
                        financial_summary='\n'.join(financial_lines) or 'None',
                        contradictions_summary=contradictions_summary,
                        authorities_summary=authorities_summary,
                        web_research_summary=web_research_summary,
                        run_id=run_id,
                    )
                    cost += 0.05
                    for opp_theory in opp_theories:
                        if cancel_event.is_set():
                            break
                        opp_id = add_ci_theory(
                            run_id=run_id,
                            theory_text=opp_theory.get('theory_text', ''),
                            theory_type=opp_theory.get('theory_type', 'factual'),
                            role_perspective=opposing_role,
                            confidence=opp_theory.get('confidence', 0.5),
                            supporting_evidence=json.dumps(
                                opp_theory.get('supporting_evidence', [])),
                        )
                        findings.append({
                            'content': (f"[{opposing_role.upper()} THEORY] "
                                        f"{opp_theory.get('theory_text', '')}"),
                            'source':  f'theory_id={opp_id}',
                            'confidence': 'medium',
                        })
                except Exception as e:
                    logger.warning(f"Opposing theory pass error: {e}")

        return findings, cost

    def _manager_authorities(self, run_id: str, run,
                               cancel_event: threading.Event) -> tuple:
        """Legal authority retrieval."""
        findings: List[Dict] = []
        cost = 0.0
        try:
            from analyzer.case_intelligence.authority_retriever import AuthorityRetriever
            jurisdiction = self._load_jurisdiction(run)
            retriever = AuthorityRetriever(cohere_api_key=self.cohere_api_key)
            if not retriever.enabled or not jurisdiction:
                return findings, cost

            entities_text = ' '.join(
                e['name'] for e in get_ci_entities(run_id)[:10]
            )
            query = (f"{run['goal_text'] or ''} "
                     f"{entities_text} "
                     f"{jurisdiction.baseline_framework}")

            results = retriever.search(
                query=query,
                jurisdiction_jurisdictions=jurisdiction.authority_jurisdictions,
                n_results=10,
            )
            for auth in results:
                authority_id = add_ci_authority(
                    run_id=run_id,
                    citation=auth['citation'],
                    authority_type='binding' if auth.get('reliability') == 'official' else 'persuasive',
                    jurisdiction=auth.get('jurisdiction'),
                    source=auth.get('source'),
                    source_url=auth.get('source_url'),
                    reliability=auth.get('reliability', 'official'),
                    excerpt=auth.get('excerpt'),
                    relevance_note=f"Score: {auth.get('relevance_score', 0):.2f}",
                )
                findings.append({
                    'content': auth['citation'],
                    'source': auth.get('source', ''),
                    'confidence': 'high' if auth.get('reliability') == 'official' else 'medium',
                    'type': 'binding' if auth.get('reliability') == 'official' else 'persuasive',
                })
        except Exception as e:
            logger.warning(f"Authority retrieval error: {e}")

        # Inject web research legal authorities (from Phase W)
        try:
            web_auth_rows = get_ci_web_research(run_id, search_type='legal_authority')
            for wr in web_auth_rows:
                for item in json.loads(wr.get('results_json') or '[]'):
                    citation = item.get('citation') or item.get('title') or 'Unknown'
                    add_ci_authority(
                        run_id=run_id,
                        citation=citation,
                        authority_type=item.get('authority_type', 'persuasive'),
                        jurisdiction=item.get('court', ''),
                        source=item.get('source', 'web'),
                        source_url=item.get('url', ''),
                        reliability=item.get('reliability', 'persuasive'),
                        excerpt=item.get('excerpt', '')[:400],
                        relevance_note=f"Web search: {wr.get('query', '')[:80]}",
                    )
                    findings.append({
                        'content': citation,
                        'source': item.get('source', 'web'),
                        'confidence': 'medium',
                        'type': item.get('authority_type', 'persuasive'),
                    })
            if web_auth_rows:
                logger.info(f"CI run {run_id}: injected web authorities from "
                            f"{len(web_auth_rows)} web research rows")
        except Exception as e:
            logger.warning(f"Web authority injection error: {e}")

        return findings, cost

    # -----------------------------------------------------------------------
    # Phase 2F/2D/2W/2R: Specialist analysis (tier 3+/4+)
    # -----------------------------------------------------------------------

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
    # Phase D2: Director synthesizes report
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

    def _legacy_findings_summary(self, run_id: str, run) -> Optional[str]:
        """Fallback key-findings summary (legacy format)."""
        entities = get_ci_entities(run_id)
        events = get_ci_timeline(run_id)
        contradictions = get_ci_contradictions(run_id)
        theories = get_ci_theories(run_id)

        critical_events = [
            ev for ev in events if ev['significance'] in ('critical', 'high')
        ][:10]

        entities_text = '\n'.join(
            f"- {e['name']} ({e['entity_type']})" for e in entities[:15]
        )
        critical_events_text = '\n'.join(
            f"- {ev['event_date'] or 'unknown'}: {ev['description']}"
            for ev in critical_events
        )
        contradictions_text = '\n'.join(
            f"- [{c['severity']}] {c['description']}" for c in contradictions[:10]
        )
        theories_text = '\n'.join(
            f"- [{t['status']} {int((t['confidence'] or 0)*100)}%] {t['theory_text'][:150]}"
            for t in theories[:8]
        )

        prompt = f"""Compile a KEY FINDINGS SUMMARY for this legal case analysis.

ROLE: {run['role']}
GOAL: {run['goal_text'] or 'Not specified'}

ENTITIES ({len(entities)}):
{entities_text}

CRITICAL TIMELINE ({len(critical_events)} events):
{critical_events_text or 'None'}

CONTRADICTIONS ({len(contradictions)}):
{contradictions_text or 'None'}

THEORIES ({len(theories)}):
{theories_text or 'None'}

Write a concise JSON summary of the 5 most actionable findings:
{{
  "key_findings": [{{"rank": 1, "severity": "critical|high|medium", "finding": "...",
                     "detail": "...", "recommended_action": "..."}}],
  "discovery_gaps": ["..."],
  "next_actions": ["..."],
  "overall_assessment": "..."
}}
"""
        return self._call_llm_simple(prompt, 'openai', 'gpt-4o', 3000,
                                      'ci:findings_summary')

    # -----------------------------------------------------------------------
    # Phase D3: Paperless write-back
    # -----------------------------------------------------------------------

    def _paperless_writeback(self, run_id: str, run):
        """Apply AI tags to documents in Paperless based on findings."""
        if not self.paperless_client:
            return

        docs_to_tag: Dict[int, set] = {}

        # Tag docs cited in critical contradictions
        for c in [dict(r) for r in get_ci_contradictions(run_id)]:
            for prov_json in [c.get('doc_a_provenance', '[]'),
                               c.get('doc_b_provenance', '[]')]:
                try:
                    for prov in json.loads(prov_json):
                        doc_id = prov.get('paperless_doc_id')
                        if doc_id:
                            docs_to_tag.setdefault(doc_id, set())
                            if c['severity'] in ('high', 'critical'):
                                docs_to_tag[doc_id].add('AI:Contradiction')
                except Exception:
                    pass

        # Tag docs cited in high-confidence theories
        for t in [dict(r) for r in get_ci_theories(run_id)]:
            if t['status'] == 'supported' and (t['confidence'] or 0) >= 0.7:
                try:
                    for ev in json.loads(t.get('supporting_evidence', '[]') or '[]'):
                        doc_id = ev.get('paperless_doc_id')
                        if doc_id:
                            docs_to_tag.setdefault(doc_id, set()).add('AI:KeyExhibit')
                except Exception:
                    pass

        for doc_id, tags in docs_to_tag.items():
            try:
                if hasattr(self.paperless_client, 'update_document_tags'):
                    self.paperless_client.update_document_tags(doc_id, list(tags), add_only=True)
                    logger.debug(f"Write-back: tagged doc {doc_id} with {tags}")
            except Exception as e:
                logger.warning(f"Write-back failed for doc {doc_id}: {e}")

    # -----------------------------------------------------------------------
    # RAG embedding of CI findings
    # -----------------------------------------------------------------------

    def _embed_ci_run_findings(self, run_id: str, run):
        """Embed CI findings into the project's Chroma collection for AI Chat + Director awareness."""
        if not run:
            return
        try:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=run.get('project_slug', 'default'))
            if not vs.enabled:
                logger.info(f"CI embedding skipped — vector store not enabled")
                return
        except Exception as e:
            logger.warning(f"CI embedding: VectorStore init failed: {e}")
            return

        embedded = 0
        ci_meta = {'document_type': 'ci_finding', 'ci_run_id': run_id}

        # entities
        for row in get_ci_entities(run_id):
            try:
                content = (f"{row['name']} ({row['entity_type']}): "
                           f"{row.get('role_in_case') or ''}.")
                doc_id = f"ci:{run_id}:entities:{row['id']}"
                vs.embed_document(doc_id, f"CI Entity: {row['name']}", content,
                                  {**ci_meta, 'ci_domain': 'entities'})
                embedded += 1
            except Exception:
                pass

        # timeline events
        for row in get_ci_timeline(run_id):
            try:
                content = (f"{row['event_date'] or 'unknown date'}: "
                           f"{row['description']}. "
                           f"Significance: {row['significance'] or 'medium'}.")
                doc_id = f"ci:{run_id}:timeline:{row['id']}"
                vs.embed_document(doc_id, f"CI Timeline: {row['event_date'] or '?'}",
                                  content, {**ci_meta, 'ci_domain': 'timeline'})
                embedded += 1
            except Exception:
                pass

        # contradictions
        for row in get_ci_contradictions(run_id):
            try:
                content = (f"Contradiction [{row['severity']}]: {row['description']}. "
                           f"{row.get('explanation') or ''}")
                doc_id = f"ci:{run_id}:contradictions:{row['id']}"
                vs.embed_document(doc_id, f"CI Contradiction", content,
                                  {**ci_meta, 'ci_domain': 'contradictions'})
                embedded += 1
            except Exception:
                pass

        # theories
        for row in get_ci_theories(run_id):
            try:
                content = (f"Theory [{row['theory_type']}]: {row['theory_text']}. "
                           f"Status: {row['status']}. "
                           f"Confidence: {int((row.get('confidence') or 0) * 100)}%.")
                doc_id = f"ci:{run_id}:theories:{row['id']}"
                vs.embed_document(doc_id, f"CI Theory: {row['theory_text'][:60]}",
                                  content, {**ci_meta, 'ci_domain': 'theories'})
                embedded += 1
            except Exception:
                pass

        # authorities
        for row in get_ci_authorities(run_id):
            try:
                content = (f"{row['citation']} ({row['jurisdiction'] or 'unknown'}, "
                           f"{row['authority_type']}): "
                           f"{row['relevance_note'] or ''}. "
                           f"Reliability: {row['reliability'] or 'official'}.")
                doc_id = f"ci:{run_id}:authorities:{row['id']}"
                vs.embed_document(doc_id, f"CI Authority: {row['citation'][:60]}",
                                  content, {**ci_meta, 'ci_domain': 'authorities'})
                embedded += 1
            except Exception:
                pass

        # disputed facts
        for row in get_ci_disputed_facts(run_id):
            try:
                content = (f"Disputed: {row['fact_description']}. "
                           f"{row.get('position_a_label') or 'Party A'}: "
                           f"{row.get('position_a_text') or ''}. "
                           f"{row.get('position_b_label') or 'Party B'}: "
                           f"{row.get('position_b_text') or ''}.")
                doc_id = f"ci:{run_id}:disputed:{row['id']}"
                vs.embed_document(doc_id, f"CI Disputed Fact", content,
                                  {**ci_meta, 'ci_domain': 'disputed_facts'})
                embedded += 1
            except Exception:
                pass

        logger.info(f"CI run {run_id}: embedded {embedded} findings into vector store")

    # -----------------------------------------------------------------------
    # Budget checkpoint
    # -----------------------------------------------------------------------

    # Checkpoints at which budget notifications are sent.
    # 80% and 90% are flagged urgent.
    _BUDGET_CHECKPOINTS = (50, 70, 80, 90)
    _URGENT_CHECKPOINTS = (80, 90)

    def _check_budget_checkpoint(self, run_id: str, current_pct: float):
        """Fire budget checkpoints at 50/70/80/90%. Enforce budget ceiling per overage policy."""
        try:
            run = get_ci_run(run_id)
            if not run:
                return
            last_checkpoint = run.get('last_budget_checkpoint_pct') or 0

            # Find the highest checkpoint we've crossed that we haven't notified yet
            next_checkpoint = None
            for cp in self._BUDGET_CHECKPOINTS:
                if current_pct >= cp > last_checkpoint:
                    next_checkpoint = cp
            if next_checkpoint is None:
                return

            cost_so_far = run.get('cost_so_far_usd') or 0
            budget = run.get('budget_per_run_usd') or 1.0
            # allow_overage_pct: 0 = hard block at 100%, 20 = block at 120%, -1 = never block
            allow_overage_pct = run.get('allow_overage_pct') or 0
            hard_limit = budget if allow_overage_pct == 0 else (
                None if allow_overage_pct < 0 else budget * (1 + allow_overage_pct / 100)
            )

            pct_fraction = next_checkpoint / 100
            projected = cost_so_far / pct_fraction if pct_fraction > 0 else cost_so_far
            ratio = projected / budget if budget > 0 else 1.0

            if ratio < 0.9:
                status = 'under_budget'
            elif ratio < 1.1:
                status = 'on_track'
            else:
                status = 'over_budget'

            is_urgent = next_checkpoint in self._URGENT_CHECKPOINTS
            logger.info(f"CI run {run_id}: budget checkpoint {next_checkpoint}%"
                        f"{' [URGENT]' if is_urgent else ''} — "
                        f"cost ${cost_so_far:.4f}, projected ${projected:.4f}, "
                        f"budget ${budget:.4f} — {status}")

            update_ci_run(run_id, last_budget_checkpoint_pct=float(next_checkpoint))

            # Hard enforcement: block when actual cost exceeds the hard limit (if any)
            if hard_limit is not None and cost_so_far > hard_limit:
                overage_label = (
                    f"${budget:.2f} limit"
                    if allow_overage_pct == 0
                    else f"${hard_limit:.2f} limit ({allow_overage_pct}% overage allowed)"
                )
                note = (f"Budget exceeded: spent ${cost_so_far:.4f} of "
                        f"{overage_label} at {next_checkpoint}% progress")
                logger.warning(f"CI run {run_id}: {note} — halting run")
                update_ci_run(run_id, status='budget_blocked',
                              budget_blocked=1, budget_blocked_note=note)
                try:
                    from analyzer.case_intelligence.job_manager import get_job_manager
                    event = get_job_manager().get_cancel_event(run_id)
                    if event:
                        event.set()
                except Exception as ev_err:
                    logger.warning(f"Could not signal cancel event: {ev_err}")
                if self.budget_notification_cb:
                    try:
                        self.budget_notification_cb(
                            run_id, next_checkpoint, cost_so_far, projected, budget,
                            'blocked', is_urgent=True
                        )
                    except Exception:
                        pass
                return

            if self.budget_notification_cb:
                try:
                    self.budget_notification_cb(
                        run_id, next_checkpoint, cost_so_far, projected, budget,
                        status, is_urgent=is_urgent
                    )
                except Exception as e:
                    logger.warning(f"Budget notification callback error: {e}")

        except Exception as e:
            logger.warning(f"Budget checkpoint error: {e}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve_manager_count(self, run, doc_count: int) -> int:
        """Determine manager count from run config (auto = scale with docs)."""
        count = run.get('manager_count')
        if count and count > 0:
            return min(int(count), 10)
        return min(6, math.ceil(doc_count / 20) + 1)

    def _resolve_workers_per_manager(self, run, manager_count: int) -> int:
        """Determine workers per manager from run config (auto = budget-based)."""
        wpm = run.get('workers_per_manager')
        if wpm and wpm > 0:
            return min(int(wpm), 20)
        # Default: 3 workers per manager, bounded by budget
        budget = run.get('budget_per_run_usd') or 1.0
        est_cost_per_worker = 0.005
        max_by_budget = max(1, int(budget / est_cost_per_worker / max(manager_count, 1)))
        return min(max(3, max_by_budget), 20)

    def _fetch_case_documents(self, run) -> List[Dict[str, Any]]:
        """Fetch all documents for the run's project from Paperless."""
        if not self.paperless_client:
            logger.warning("No Paperless client — cannot fetch documents")
            return []
        try:
            project_slug = run['project_slug']

            # For the default project, skip tag filtering — all Paperless docs belong here.
            # For named projects, filter by project: tag so CI only sees that project's docs.
            if project_slug and project_slug != 'default':
                result = self.paperless_client.get_documents_by_project(
                    project_slug, page_size=10000
                )
                docs = result.get('results', []) if isinstance(result, dict) else result
                # If very few docs found via tag (tag may not be applied yet), fall back to all
                if len(docs) < 5:
                    logger.warning(
                        f"Only {len(docs)} tagged docs for project '{project_slug}', "
                        f"falling back to all documents"
                    )
                    result = self.paperless_client.get_documents(page_size=10000)
                    docs = result.get('results', []) if isinstance(result, dict) else result
            else:
                # Default project: get all documents in Paperless
                result = self.paperless_client.get_documents(page_size=10000)
                docs = result.get('results', []) if isinstance(result, dict) else result

            logger.info(f"CI run {run['id']}: fetched {len(docs)} documents")

            # Enrich each doc with pre-computed AI analysis from the vector store.
            # The standard analysis pipeline stores brief_summary, full_summary, and
            # document_type for every processed document. CI workers use this to give
            # extractors the full-document context even when the raw OCR is very long.
            if docs:
                try:
                    from analyzer.vector_store import VectorStore
                    vs = VectorStore(project_slug=run.get('project_slug', 'default'))
                    if vs.enabled:
                        doc_ids = [d.get('id') for d in docs if d.get('id') is not None]
                        enrichment = vs.get_documents_metadata(doc_ids)
                        for doc in docs:
                            meta = enrichment.get(doc.get('id'), {})
                            doc['vs_brief_summary'] = meta.get('brief_summary', '') or ''
                            doc['vs_full_summary']  = meta.get('full_summary', '') or ''
                            doc['vs_document_type'] = meta.get('document_type', '') or ''
                        enriched = sum(1 for d in docs if d.get('vs_brief_summary'))
                        logger.info(
                            f"CI run {run['id']}: enriched {enriched}/{len(docs)} docs "
                            f"with vector store AI analysis"
                        )
                except Exception as vs_err:
                    logger.warning(f"CI vector store enrichment failed (non-fatal): {vs_err}")

            return docs
        except Exception as e:
            logger.error(f"Failed to fetch case documents: {e}")
            return []

    def _estimate_doc_count(self, run) -> int:
        """Quick doc count estimate for questions phase."""
        try:
            if self.paperless_client:
                docs = self._fetch_case_documents(run)
                return len(docs)
        except Exception:
            pass
        return 0

    def _load_jurisdiction(self, run) -> Optional[JurisdictionProfile]:
        """Load jurisdiction profile from the run config."""
        try:
            jd_data = json.loads(run.get('jurisdiction_json') or '{}')
            # Only construct if required fields are present (user may store a partial
            # dict like {"state": "New York"} when no formal profile is selected)
            if jd_data and jd_data.get('jurisdiction_id'):
                return JurisdictionProfile.from_dict(jd_data)
        except Exception as e:
            logger.warning(f"Jurisdiction load error: {e}")
        return None

    def _finalize_empty(self, run_id: str):
        """Mark run complete with no-documents note."""
        update_ci_run(run_id, status='completed',
                      findings_summary='No documents found for this project.',
                      completed_at=datetime.now(timezone.utc).isoformat(),
                      current_stage='Completed (no documents)',
                      progress_pct=100)

    def _set_status(self, run_id: str, status: str, stage: str = None,
                    progress: float = None, docs_processed: int = None,
                    tokens_in: int = None, tokens_out: int = None,
                    active_managers: int = None, active_workers: int = None):
        """Update run status in the database."""
        kwargs = {'status': status}
        if stage:
            kwargs['current_stage'] = stage
        if progress is not None:
            kwargs['progress_pct'] = progress
        if docs_processed is not None:
            kwargs['docs_processed'] = docs_processed
        if tokens_in is not None:
            kwargs['tokens_in'] = tokens_in
        if tokens_out is not None:
            kwargs['tokens_out'] = tokens_out
        if active_managers is not None:
            kwargs['active_managers'] = active_managers
        if active_workers is not None:
            kwargs['active_workers'] = active_workers
        update_ci_run(run_id, **kwargs)

    def _is_cancelled(self, cancel_event: Optional[threading.Event],
                       run_id: str) -> bool:
        """Check if the run has been cancelled."""
        if cancel_event and cancel_event.is_set():
            update_ci_run(run_id, status='cancelled', current_stage='Cancelled by user')
            return True
        run = get_ci_run(run_id)
        if run and run['status'] in ('cancelled', 'budget_blocked'):
            return True
        return False

    def _call_llm_simple(self, prompt: str, provider: str, model: str,
                          max_tokens: int, operation: str) -> Optional[str]:
        """Simple LLM call returning raw text."""
        client = self.llm_clients.get(provider)
        if not client:
            logger.warning(f"No LLM client for provider '{provider}'")
            return None
        try:
            if provider == 'anthropic':
                response = client.client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.content[0].text
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model, operation=operation,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    )
                with self._token_lock:
                    self._tokens_in += response.usage.input_tokens
                    self._tokens_out += response.usage.output_tokens
                return text
            elif provider == 'openai':
                response = client.client.chat.completions.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.choices[0].message.content
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model, operation=operation,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    )
                with self._token_lock:
                    self._tokens_in += response.usage.prompt_tokens
                    self._tokens_out += response.usage.completion_tokens
                return text
        except Exception as e:
            logger.error(f"LLM call failed [{provider}/{model}]: {e}")
            return None
