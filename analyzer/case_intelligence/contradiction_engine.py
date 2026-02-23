"""
Contradiction Engine — Tier 2 CI Pipeline Stage.

Detects conflicting facts, dates, amounts, or party descriptions across
multiple documents. Also builds the Disputed Facts Matrix.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

CONTRADICTION_DETECTION_PROMPT = """You are a forensic document analyst performing cross-document contradiction analysis for legal proceedings.

Analyze the documents below and identify CONTRADICTIONS — places where two or more documents contain directly conflicting facts. A contradiction requires that both documents are accessible as evidence.

CASE ROLE: {role}
CASE GOAL: {goal_text}

DOCUMENTS TO ANALYZE:
{documents_context}

Respond in JSON:
{{
  "contradictions": [
    {{
      "description": "Clear description of the contradiction",
      "severity": "low|medium|high|critical",
      "contradiction_type": "date_conflict|amount_conflict|party_conflict|factual_conflict|status_conflict",
      "doc_a": {{
        "paperless_doc_id": 101,
        "title": "Document A",
        "excerpt": "exact quote (max 200 chars)",
        "page_number": 1,
        "claimed_fact": "What doc A claims"
      }},
      "doc_b": {{
        "paperless_doc_id": 102,
        "title": "Document B",
        "excerpt": "exact quote (max 200 chars)",
        "page_number": 1,
        "claimed_fact": "What doc B claims (contradicting doc A)"
      }},
      "explanation": "Why this is a meaningful contradiction",
      "suggested_action": "What to do about this contradiction"
    }}
  ],
  "disputed_facts": [
    {{
      "fact_description": "The fact at issue",
      "position_a_label": "Plaintiff's position",
      "position_a_text": "What the plaintiff's documents show",
      "position_b_label": "Defense position",
      "position_b_text": "What the defense documents show",
      "key_docs_a": [101, 103],
      "key_docs_b": [102, 104]
    }}
  ]
}}

SEVERITY GUIDE:
- critical: Directly contradicts a key legal element or sworn statement
- high: Significant factual conflict that affects the case theory
- medium: Notable inconsistency requiring explanation
- low: Minor discrepancy that may have an innocent explanation

RULES:
1. Only identify genuine contradictions — not just different perspectives on the same fact.
2. Cite specific text from BOTH documents for every contradiction.
3. Do NOT fabricate contradictions not supported by the provided text.
4. Critical severity = direct conflict on a legally operative fact (dates, amounts, parties, admissions).
"""

DISPUTED_FACTS_PROMPT = """You are building a disputed facts matrix for legal proceedings.

Based on all documents in this case, identify the key factual disputes — areas where the evidence supports opposing positions.

CASE ROLE: {role}
DOCUMENTS SUMMARY:
{docs_summary}

Build a comprehensive disputed facts matrix. Respond in JSON:
{{
  "disputed_facts": [
    {{
      "fact_description": "The fact at issue (specific and concrete)",
      "position_a_label": "Plaintiff / Prosecution position",
      "position_a_text": "Specific position with supporting facts",
      "position_b_label": "Defense / Respondent position",
      "position_b_text": "Specific position with supporting facts",
      "supporting_evidence_a": [
        {{"paperless_doc_id": 101, "excerpt": "...", "page_number": 1}}
      ],
      "supporting_evidence_b": [
        {{"paperless_doc_id": 102, "excerpt": "...", "page_number": 2}}
      ],
      "resolution_status": "open"
    }}
  ]
}}

Include at minimum:
- Disputed amounts/payments
- Disputed timelines
- Disputed authorization/consent
- Disputed ownership/title
"""


class ContradictionEngine:
    """
    Tier 2 contradiction detection and disputed facts matrix builder.
    Primary: gpt-4o, escalation: claude-sonnet-4-5.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.contradiction_task = get_task('contradiction_detection')
        self.disputed_task = get_task('disputed_facts_matrix')

    def detect_contradictions(self, documents: List[Dict[str, Any]],
                               role: str = 'neutral',
                               goal_text: str = '') -> Dict[str, Any]:
        """
        Analyze a set of documents for contradictions.

        Args:
            documents: List of dicts with keys: doc_id, title, content (truncated), excerpts
            role: defense|plaintiff|prosecution|neutral
            goal_text: Case goal description

        Returns:
            dict with 'contradictions' list and 'disputed_facts' list
        """
        if len(documents) < 2:
            return {'contradictions': [], 'disputed_facts': []}

        docs_context = self._build_docs_context(documents)
        prompt = CONTRADICTION_DETECTION_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Identify all relevant contradictions',
            documents_context=docs_context,
        )

        result = self._call_with_escalation(
            prompt, self.contradiction_task, operation='ci:contradiction_detection'
        )
        if not result:
            return {'contradictions': [], 'disputed_facts': []}

        # Add provenance structure for contradictions
        extraction_ts = datetime.now(timezone.utc).isoformat()
        for contr in result.get('contradictions', []):
            doc_a = contr.get('doc_a', {})
            doc_b = contr.get('doc_b', {})
            contr['doc_a_provenance'] = json.dumps([{
                'paperless_doc_id': doc_a.get('paperless_doc_id'),
                'page_number': doc_a.get('page_number'),
                'excerpt': doc_a.get('excerpt', ''),
                'extraction_version': extraction_ts,
                'prompt_version': 'contradiction_v1',
            }])
            contr['doc_b_provenance'] = json.dumps([{
                'paperless_doc_id': doc_b.get('paperless_doc_id'),
                'page_number': doc_b.get('page_number'),
                'excerpt': doc_b.get('excerpt', ''),
                'extraction_version': extraction_ts,
                'prompt_version': 'contradiction_v1',
            }])

        return result

    def build_disputed_facts_matrix(self, documents: List[Dict[str, Any]],
                                     role: str = 'neutral') -> List[Dict[str, Any]]:
        """Build the full disputed facts matrix from all case documents."""
        docs_summary = self._build_docs_summary(documents)
        prompt = DISPUTED_FACTS_PROMPT.format(
            role=role,
            docs_summary=docs_summary,
        )

        result = self._call_with_escalation(
            prompt, self.disputed_task, operation='ci:disputed_facts_matrix'
        )
        if not result:
            return []

        return result.get('disputed_facts', [])

    def _build_docs_context(self, documents: List[Dict[str, Any]],
                             max_per_doc: int = 1500) -> str:
        """Build a compact multi-document context for contradiction analysis."""
        parts = []
        for doc in documents[:20]:  # Limit to 20 docs per call
            parts.append(
                f"--- DOC #{doc.get('doc_id')} | {doc.get('title', 'Untitled')} ---\n"
                f"{doc.get('content', '')[:max_per_doc]}"
            )
        return '\n\n'.join(parts)

    def _build_docs_summary(self, documents: List[Dict[str, Any]]) -> str:
        """Build a summary context for the disputed facts matrix."""
        parts = []
        for doc in documents:
            entities = doc.get('entities', [])
            events = doc.get('events', [])
            parts.append(
                f"Doc #{doc.get('doc_id')} ({doc.get('title', 'Untitled')}): "
                f"entities={len(entities)}, events={len(events)}"
            )
        return '\n'.join(parts)

    def _call_with_escalation(self, prompt: str, task_def,
                               operation: str) -> Optional[dict]:
        for model_key in ('primary', 'escalate'):
            model = (task_def.primary_model if model_key == 'primary'
                     else task_def.escalate_model)
            provider = (task_def.primary_provider if model_key == 'primary'
                        else task_def.escalate_provider)

            client = self.llm_clients.get(provider)
            if not client:
                continue

            try:
                result = self._call_llm(client, model, prompt, provider, operation)
                if result:
                    return result
                if model_key == 'primary':
                    logger.debug(f"ContradictionEngine: escalating ({operation})")
            except Exception as e:
                logger.error(f"ContradictionEngine: {provider}/{model} failed: {e}")

        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, operation: str) -> Optional[dict]:
        try:
            if provider == 'anthropic':
                response = client.client.messages.create(
                    model=model,
                    max_tokens=self.contradiction_task.max_output_tokens,
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.content[0].text
                usage = response.usage
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model, operation=operation,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
            elif provider == 'openai':
                response = client.client.chat.completions.create(
                    model=model,
                    max_tokens=self.contradiction_task.max_output_tokens,
                    response_format={'type': 'json_object'},
                    messages=[{'role': 'user', 'content': prompt}],
                )
                text = response.choices[0].message.content
                usage = response.usage
                if self.usage_tracker:
                    self.usage_tracker.log_usage(
                        provider=provider, model=model, operation=operation,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                    )
            else:
                return None

            return json.loads(text)
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.error(f"ContradictionEngine LLM call failed: {e}")
            return None
