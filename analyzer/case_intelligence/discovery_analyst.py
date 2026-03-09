"""
Discovery Analyst — Tier 3+ CI Pipeline Phase 2D.

Identifies discovery gaps: what documents should exist based on the
evidence but are absent from the case file.

Senior partners know that:
  - Every meeting → calendar invite + email chain + follow-up
  - Every wire → authorization + bank confirmation + statement
  - Every agreement → drafts, redlines, negotiation correspondence
  - Every termination → HR records, performance reviews, separation agreement
  - Abruptly ending email threads = deleted next message
  - Zero docs in a key date range = spoliation red flag
  - Signatories who disappear from later docs = were cut out (depose why)

Output stored in ci_discovery_gaps.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

DISCOVERY_GAP_PROMPT = """You are a senior litigator and discovery expert preparing the discovery strategy for this case.

CASE ROLE: {role}
CASE GOAL: {goal_text}
JURISDICTION: {jurisdiction}

DOCUMENTS WE HAVE (summary):
{docs_summary}

ENTITIES (parties, witnesses, organizations):
{entities_data}

TIMELINE (key events):
{timeline_data}

FINANCIAL TRANSACTIONS (summary):
{financial_data}

Based on the documents and facts in evidence, identify what is MISSING.

Think like a senior partner who has done 100 of these cases:
- Every meeting generates a calendar invite, email chain, and follow-up memo
- Every wire transfer generates an authorization, confirmation, and bank statement
- Every signed agreement generates drafts, redlines, and negotiation correspondence
- Every termination generates HR records, performance reviews, and a separation agreement
- Every lawsuit has pre-litigation demand letters and settlement negotiations
- When an email thread ends abruptly, the next message was deleted
- When a key date range has zero documents, that is a spoliation red flag
- When a signatory disappears from later documents, they were cut out — depose why

Respond in JSON:
{{
  "missing_document_types": [
    {{
      "description": "What is missing (specific document type or category)",
      "why_expected": "Why this document should exist based on evidence we have",
      "based_on_docs": [101, 102],
      "based_on_fact": "The specific fact that implies this document exists",
      "priority": "critical|high|medium",
      "who_should_have": "Party or custodian most likely to hold this"
    }}
  ],
  "custodian_gaps": [
    {{
      "person": "Person name",
      "role": "Their role in the matter",
      "expected_documents": "What documents this person should have produced",
      "actual_doc_count": 0,
      "gap_description": "Description of what is missing from this custodian",
      "significance": "Why this gap matters"
    }}
  ],
  "spoliation_indicators": [
    {{
      "indicator": "Description of the spoliation indicator",
      "supporting_evidence": [
        {{"paperless_doc_id": 101, "excerpt": "..."}}
      ],
      "date_range_affected": "Date range with suspicious document absence",
      "severity": "critical|high|medium",
      "recommended_action": "Motion for sanctions / adverse inference instruction / etc."
    }}
  ],
  "rfp_list": [
    {{
      "item": "RFP #N: Produce all [specific document description]",
      "legal_basis": "Relevance basis under applicable discovery rules",
      "priority": 1,
      "expected_source": "Which party or custodian to direct this to",
      "why_critical": "Why this RFP matters to the case theory"
    }}
  ],
  "subpoena_targets": [
    {{
      "entity": "Third-party name (bank, employer, etc.)",
      "reason": "Why they likely have relevant documents",
      "likely_documents": "Specific records to subpoena",
      "priority": "high|medium|low"
    }}
  ],
  "summary": "Discovery strategy memo (3-5 paragraphs: key gaps, spoliation concerns, priority RFPs, and deposition targets to lock in facts before document production)"
}}

RULES:
1. Only flag gaps actually supported by documents/facts in evidence — no speculation.
2. Prioritize critical gaps that directly affect the case theory.
3. RFP items should be specific enough to withstand an objection for overbreadth.
4. Number RFPs sequentially starting at #1.
5. Maximum 20 RFPs, 10 missing_document_types, 5 spoliation_indicators.
"""


class DiscoveryAnalyst:
    """
    Tier 3+ discovery gap analysis.
    Identifies missing documents, spoliation, and generates RFP list.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('discovery_gap_analysis')

    def analyze(self, run_id: str, role: str, goal_text: str, jurisdiction: str,
                documents: List[Dict], entities: List[Dict],
                timeline: List[Dict], financial: List[Dict]) -> Dict[str, Any]:
        """
        Run discovery gap analysis.

        Returns:
            Discovery gaps dict (mirrors ci_discovery_gaps schema)
        """
        if not documents:
            return self._empty_result()

        docs_summary = self._format_docs(documents)
        entities_str = self._format_entities(entities)
        timeline_str = self._format_timeline(timeline)
        financial_str = self._format_financial(financial)

        prompt = DISCOVERY_GAP_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Analyze case for discovery',
            jurisdiction=jurisdiction or 'Not specified',
            docs_summary=docs_summary[:6000],
            entities_data=entities_str[:2000],
            timeline_data=timeline_str[:2000],
            financial_data=financial_str[:2000],
        )

        result = self._call_llm_with_escalation(prompt, 'ci:discovery_gap_analysis')
        if not result:
            return self._empty_result()

        logger.info(
            f"DiscoveryAnalyst run {run_id}: "
            f"{len(result.get('missing_document_types', []))} gaps, "
            f"{len(result.get('rfp_list', []))} RFPs, "
            f"{len(result.get('spoliation_indicators', []))} spoliation flags"
        )
        return result

    def _format_docs(self, docs: List[Dict]) -> str:
        parts = []
        for doc in docs[:50]:
            doc_id = doc.get('id') or doc.get('doc_id', '?')
            title = doc.get('title', 'Untitled')
            content = (doc.get('content') or '')[:300]
            created = doc.get('created', doc.get('added', '?'))
            parts.append(f"[Doc #{doc_id}] {title} (created: {created})\n  {content}")
        return '\n\n'.join(parts)

    def _format_entities(self, entities: List[Dict]) -> str:
        return '\n'.join(
            f"{e.get('entity_type', '?')}: {e.get('name', '?')} ({e.get('role_in_case', '')})"
            for e in entities[:30]
        )

    def _format_timeline(self, events: List[Dict]) -> str:
        return '\n'.join(
            f"{e.get('event_date', '?')}: {e.get('description', '')} [{e.get('significance', '?')}]"
            for e in events[:40]
        )

    def _format_financial(self, data: List[Dict]) -> str:
        if not data:
            return 'No financial transactions in evidence.'
        parts = []
        for item in data[:30]:
            if isinstance(item, dict):
                parts.append(
                    f"{item.get('date', '?')}: {item.get('description', '')} "
                    f"${item.get('amount', item.get('amount_usd', 0)):,} "
                    f"[Doc #{item.get('doc_id', item.get('paperless_doc_id', '?'))}]"
                )
        return '\n'.join(parts)

    def _call_llm_with_escalation(self, prompt: str, operation: str) -> Optional[Dict]:
        for model_key in ('primary', 'escalate'):
            model = (self.task_def.primary_model if model_key == 'primary'
                     else self.task_def.escalate_model)
            provider = (self.task_def.primary_provider if model_key == 'primary'
                        else self.task_def.escalate_provider)
            client = self.llm_clients.get(provider)
            if not client:
                continue
            try:
                result = self._call_llm(client, model, prompt, provider, operation)
                if result:
                    return result
            except Exception as e:
                logger.error(f"DiscoveryAnalyst {provider}/{model} failed: {e}")
        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, operation: str) -> Optional[Dict]:
        try:
            if provider == 'anthropic':
                response = client.client.messages.create(
                    model=model,
                    max_tokens=self.task_def.max_output_tokens,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.content[0].text
                usage = response.usage
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model, operation=operation,
                        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                    )
            elif provider == 'openai':
                response = client.client.chat.completions.create(
                    model=model,
                    max_tokens=self.task_def.max_output_tokens,
                    response_format={'type': 'json_object'},
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.choices[0].message.content
                usage = response.usage
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model, operation=operation,
                        input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
                    )
            else:
                return None

            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"DiscoveryAnalyst JSON error: {e}")
            return None
        except Exception as e:
            logger.error(f"DiscoveryAnalyst LLM error: {e}")
            return None

    @staticmethod
    def _empty_result() -> Dict:
        return {
            'missing_document_types': [],
            'custodian_gaps': [],
            'spoliation_indicators': [],
            'rfp_list': [],
            'subpoena_targets': [],
            'summary': 'Insufficient document data for discovery gap analysis.',
        }
