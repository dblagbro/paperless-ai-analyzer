"""CI orchestrator utilities — budget checkpoints, status updates, document fetching

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


class OrchestratorUtilsMixin:
    """CI orchestrator utilities — budget checkpoints, status updates, document fetching"""

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
        """Simple LLM call returning raw text. Routes through proxy pool with
        direct-provider fallback to the given (provider, model)."""
        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        client = self.llm_clients.get(provider)
        try:
            result = call_llm(
                messages=[{'role': 'user', 'content': prompt}],
                task='analysis',
                max_tokens=max_tokens,
                operation=operation,
                direct_provider=provider,
                direct_api_key=getattr(client, 'api_key', None) if client else None,
                direct_model=model,
                usage_tracker=self.usage_tracker,
            )
            text = result.get('content') or ''
            with self._token_lock:
                self._tokens_in += int(result.get('input_tokens') or 0)
                self._tokens_out += int(result.get('output_tokens') or 0)
            return text
        except LLMUnavailableError as e:
            logger.error(f"LLM unavailable [{provider}/{model}]: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM call failed [{provider}/{model}]: {e}")
            return None
