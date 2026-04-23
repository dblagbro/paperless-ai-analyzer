"""CI Manager phase — parallel domain analysis (entities, timeline, financial, contradictions, theories, authorities)

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


class ManagerPhasesMixin:
    """CI Manager phase — parallel domain analysis (entities, timeline, financial, contradictions, theories, authorities)"""

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

                # Store patterns of conduct (same behavioral pattern repeated 3+ times)
                for p in result.get('patterns_of_conduct', []):
                    instances = p.get('instances', [])
                    doc_ids = [inst.get('paperless_doc_id') for inst in instances if inst.get('paperless_doc_id')]
                    prov = json.dumps([{'paperless_doc_id': d} for d in doc_ids[:3]])
                    add_ci_contradiction(
                        run_id=run_id,
                        description=f"[Pattern] {p.get('party', '')} — {p.get('pattern', '')}",
                        severity='high',
                        doc_a_provenance=prov,
                        doc_b_provenance='[]',
                        contradiction_type='pattern_of_conduct',
                        explanation=p.get('significance'),
                        suggested_action=None,
                    )
                    findings.append({
                        'content': p.get('pattern', ''),
                        'source': 'pattern_of_conduct',
                        'confidence': 'high',
                    })

                # Store behavioral tells (hedging language, formality shifts, CC drops, etc.)
                for t in result.get('behavioral_tells', []):
                    doc_id = t.get('doc_id')
                    prov = json.dumps([{'paperless_doc_id': doc_id, 'excerpt': t.get('excerpt', '')}]) if doc_id else '[]'
                    add_ci_contradiction(
                        run_id=run_id,
                        description=f"[Behavioral Tell: {t.get('type', 'unknown')}] {t.get('tell', '')}",
                        severity='medium',
                        doc_a_provenance=prov,
                        doc_b_provenance='[]',
                        contradiction_type='behavioral_tell',
                        explanation=t.get('significance'),
                        suggested_action=None,
                    )

                # Store communication gaps
                for g in result.get('communication_gaps', []):
                    add_ci_contradiction(
                        run_id=run_id,
                        description=f"[Communication Gap] {g.get('description', '')} ({g.get('date_range', '')})",
                        severity='medium',
                        doc_a_provenance='[]',
                        doc_b_provenance='[]',
                        contradiction_type='communication_gap',
                        explanation=g.get('significance'),
                        suggested_action=f"Parties involved: {', '.join(g.get('parties_involved', []))}",
                    )

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

