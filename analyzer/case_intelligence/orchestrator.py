"""
CI Orchestrator — Director/Manager/Worker hierarchical run controller.

Architecture:
  Director D1: reads run config + doc list → produces manager_plan JSON
  Manager (N parallel, one per domain): splits docs → spawns Workers → aggregates
  Worker  (K per manager, parallel): fetches doc, runs LLM extraction, returns findings
  Director D2: synthesizes all manager_reports → scientific paper report
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
    upsert_ci_entity, add_ci_event, add_ci_contradiction,
    add_ci_disputed_fact, add_ci_theory, update_ci_theory,
    add_ci_authority, increment_ci_run_cost, increment_ci_run_docs,
    upsert_manager_report, get_manager_reports,
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

        # Extractors (used by workers)
        self.entity_extractor = EntityExtractor(llm_clients, usage_tracker)
        self.timeline_builder = TimelineBuilder(llm_clients, usage_tracker)
        self.financial_extractor = FinancialExtractor(llm_clients, usage_tracker)
        self.contradiction_engine = ContradictionEngine(llm_clients, usage_tracker)
        self.theory_planner = TheoryPlanner(llm_clients, usage_tracker)
        self.adversarial_tester = AdversarialTester(llm_clients, usage_tracker)

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def execute_run(self, run_id: str, cancel_event: threading.Event = None):
        """Main entry — called from CIJobManager background thread."""
        if cancel_event is None:
            cancel_event = threading.Event()

        try:
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

            # ── Phase D2: Director synthesizes report ──────────────────────
            if self._is_cancelled(cancel_event, run_id):
                return

            self._set_status(run_id, 'running', stage='Director synthesizing report', progress=90)
            self._director_d2_synthesize(run_id, run, manager_reports,
                                          [d.get('id', 0) for d in documents])

            # ── Phase D3: Write-back + finalize ────────────────────────────
            if not self._is_cancelled(cancel_event, run_id):
                self._set_status(run_id, 'running', stage='Writing back to Paperless', progress=96)
                try:
                    self._paperless_writeback(run_id, run)
                except Exception as e:
                    logger.warning(f"Paperless write-back failed: {e}")

                update_ci_run(run_id, status='completed',
                              completed_at=datetime.now(timezone.utc).isoformat(),
                              current_stage='Completed',
                              progress_pct=100)
                logger.info(f"CI run {run_id} completed")

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

        # Try LLM planning (best-effort)
        prompt = DIRECTOR_D1_PROMPT.format(
            case_name=run.get('case_name') or run.get('goal_text', 'Case')[:40],
            role=run['role'],
            goal_text=run['goal_text'] or 'General analysis',
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
                    instructions, doc_map, workers_per_mgr, cancel_event
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
                             progress=20)
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
                            self._set_status(
                                run_id, 'running',
                                stage=(f'Phase 1: {completed_p1[0]}/{total_p1} '
                                       f'extraction managers done'),
                                progress=pct,
                            )
                            self._check_budget_checkpoint(run_id, pct)

        if cancel_event.is_set():
            return manager_reports

        # ── Phase 2: cross-document analysis (parallel, after extraction) ──
        if phase2_specs:
            entities_count = len(get_ci_entities(run_id))
            events_count = len(get_ci_timeline(run_id))
            logger.info(
                f"CI run {run_id}: Phase 1 complete — "
                f"{entities_count} entities, {events_count} events in DB. "
                f"Starting Phase 2 (contradictions/theories/authorities)."
            )
            self._set_status(run_id, 'running',
                             stage='Phase 2: Contradictions, theories, authorities',
                             progress=62)
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
                            self._set_status(
                                run_id, 'running',
                                stage=(f'Phase 2: {completed_p2[0]}/{total_p2} '
                                       f'analysis managers done'),
                                progress=pct,
                            )
                            self._check_budget_checkpoint(run_id, pct)

        return manager_reports

    def _run_manager(self, run_id: str, run, domain: str,
                      doc_ids: List[int], instructions: str,
                      doc_map: Dict[int, Dict], workers_per_mgr: int,
                      cancel_event: threading.Event) -> Dict:
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
                run_id, run, doc_ids, doc_map, cancel_event
            )
            worker_count = max(1, len(doc_ids))

        elif domain == 'theories':
            findings, total_cost = self._manager_theories(
                run_id, run, doc_ids, doc_map, cancel_event
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
                                  cancel_event: threading.Event) -> tuple:
        """Contradiction and disputed-facts detection across docs."""
        findings: List[Dict] = []
        cost = 0.0
        documents = [
            {'doc_id': doc_map[did].get('id', did),
             'title': doc_map[did].get('title', f'Doc {did}'),
             'content': doc_map[did].get('content', '') or '',
             'entities': [], 'events': [], 'financial_facts': []}
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
                            cancel_event: threading.Event) -> tuple:
        """Theory generation and adversarial testing."""
        findings: List[Dict] = []
        cost = 0.0

        entities = get_ci_entities(run_id)
        events = get_ci_timeline(run_id)
        contradictions = get_ci_contradictions(run_id)
        authorities_list = get_ci_authorities(run_id)

        entities_summary = '\n'.join(
            f"- [{e['entity_type']}] {e['name']}: {e['role_in_case'] or ''}"
            for e in entities[:20]
        )
        timeline_summary = '\n'.join(
            f"- {ev['event_date'] or 'unknown'} [{ev['significance']}]: {ev['description']}"
            for ev in events[:30]
        )
        contradictions_summary = '\n'.join(
            f"- [{c['severity']}] {c['description']}" for c in contradictions[:10]
        )
        authorities_summary = '\n'.join(
            f"- {a['citation']}: {a.get('relevance_note', '')}"
            for a in authorities_list[:10]
        )

        documents = [doc_map[did] for did in doc_ids if did in doc_map]
        financial_lines = []
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

        return findings, cost

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
        for c in get_ci_contradictions(run_id):
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
        for t in get_ci_theories(run_id):
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
    # Budget checkpoint
    # -----------------------------------------------------------------------

    def _check_budget_checkpoint(self, run_id: str, current_pct: float):
        """Fire a budget checkpoint every 10% increment."""
        try:
            run = get_ci_run(run_id)
            if not run:
                return
            last_checkpoint = run.get('last_budget_checkpoint_pct') or 0
            next_checkpoint = int(current_pct // 10) * 10
            if next_checkpoint <= last_checkpoint or next_checkpoint == 0:
                return

            cost_so_far = run.get('cost_so_far_usd') or 0
            budget = run.get('budget_per_run_usd') or 1.0
            pct_fraction = next_checkpoint / 100
            projected = cost_so_far / pct_fraction if pct_fraction > 0 else cost_so_far
            ratio = projected / budget if budget > 0 else 1.0

            if ratio < 0.9:
                status = 'under_budget'
            elif ratio < 1.1:
                status = 'on_track'
            else:
                status = 'over_budget'

            logger.info(f"CI run {run_id}: budget checkpoint {next_checkpoint}% — "
                        f"cost ${cost_so_far:.4f}, projected ${projected:.4f}, "
                        f"budget ${budget:.4f} — {status}")

            update_ci_run(run_id, last_budget_checkpoint_pct=float(next_checkpoint))

            if self.budget_notification_cb:
                try:
                    self.budget_notification_cb(
                        run_id, next_checkpoint, cost_so_far, projected, budget, status
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
            if jd_data:
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
                    progress: float = None, docs_processed: int = None):
        """Update run status in the database."""
        kwargs = {'status': status}
        if stage:
            kwargs['current_stage'] = stage
        if progress is not None:
            kwargs['progress_pct'] = progress
        if docs_processed is not None:
            kwargs['docs_processed'] = docs_processed
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
                return text
        except Exception as e:
            logger.error(f"LLM call failed [{provider}/{model}]: {e}")
            return None
