"""CI terminal phases — Paperless write-back and finding embedding

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


class WritebackPhasesMixin:
    """CI terminal phases — Paperless write-back and finding embedding"""

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

