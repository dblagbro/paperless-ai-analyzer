"""
Witness Analyst — Tier 4+ CI Pipeline Phase 2W.

Builds witness intelligence dossiers for every key person in the case.

Senior partners build these before every deposition:
  - Prior inconsistent statements across all case documents
  - Financial interest in the outcome (payments, loans, equity from either side)
  - Relationship bias (family, romantic, business partner connections to parties)
  - Prior bad acts / convictions admissible for impeachment
  - Areas of likely perjury based on documented contradictions
  - Documents they've signed that they'll have to own on the stand
  - Optimal deposition order (lower-level witnesses first to lock in facts)
  - Deposition question list targeting vulnerabilities

One dossier per key person (top 10 by document frequency).
Output stored in ci_witness_cards.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

WITNESS_INTELLIGENCE_PROMPT = """You are a senior trial attorney preparing for depositions. Build a detailed witness intelligence dossier for the person below.

CASE ROLE (our position): {role}
CASE GOAL: {goal_text}

WITNESS: {witness_name}
KNOWN ROLE: {witness_role}

ALL DOCUMENTS IN WHICH THIS PERSON APPEARS:
{witness_docs}

ALL CONTRADICTIONS FOUND IN THE CASE:
{contradictions_summary}

ALL FINANCIAL TRANSACTIONS:
{financial_summary}

ALL CASE ENTITIES (parties, relationships):
{entities_summary}

Build a comprehensive witness dossier. Respond in JSON:
{{
  "witness_name": "{witness_name}",
  "credibility_score": 0.0,
  "impeachment_points": [
    {{
      "point": "Description of the impeachment point",
      "doc_support": [
        {{"paperless_doc_id": 101, "excerpt": "exact quote", "page_number": 1}}
      ],
      "severity": "critical|high|medium",
      "impeachment_method": "prior_inconsistent_statement|financial_interest|bias|prior_bad_act|other"
    }}
  ],
  "financial_interest": {{
    "amount_usd": 0,
    "nature": "Payment, equity stake, loan, etc.",
    "from_which_party": "Which party paid/loaned this",
    "provenance": [{{"paperless_doc_id": 101, "excerpt": "..."}}]
  }},
  "relationship_bias": [
    {{
      "type": "employer|business_partner|family|romantic|creditor|other",
      "with_party": "Which case party",
      "description": "Nature of relationship",
      "provenance": [{{"paperless_doc_id": 101}}]
    }}
  ],
  "prior_inconsistencies": [
    {{
      "statement_a": "What this person said/signed in Document A",
      "doc_a": {{"paperless_doc_id": 101, "date": "YYYY-MM-DD", "doc_type": "email|contract|etc."}},
      "statement_b": "What contradicts it in Document B",
      "doc_b": {{"paperless_doc_id": 102, "date": "YYYY-MM-DD", "doc_type": "..."}},
      "significance": "Why this contradiction matters for deposition"
    }}
  ],
  "signed_documents": [
    {{
      "paperless_doc_id": 101,
      "doc_type": "Contract, declaration, etc.",
      "date": "YYYY-MM-DD",
      "what_they_affirmed": "What this document commits them to on the stand",
      "vulnerability": "How this document can be used against them"
    }}
  ],
  "public_record_flags": [],
  "recommended_deposition_order": 1,
  "deposition_key_questions": [
    "Q: [Specific question targeting a vulnerability or inconsistency]"
  ],
  "vulnerability_summary": "2-3 sentence assessment of this witness's overall credibility and deposition vulnerabilities"
}}

CREDIBILITY SCORE GUIDE:
  0.9–1.0: Highly credible, consistent, no apparent bias
  0.7–0.9: Generally credible but with some inconsistencies
  0.5–0.7: Mixed credibility, notable inconsistencies or bias
  0.3–0.5: Low credibility, significant inconsistencies or financial interest
  0.0–0.3: Very low credibility, multiple impeachment points

RULES:
1. Only include points actually supported by the document excerpts.
2. Every impeachment_point MUST cite a specific document.
3. Deposition questions should be pointed and specific — designed to lock in facts or expose contradictions.
4. Maximum 10 impeachment points, 5 prior inconsistencies, 15 deposition questions.
5. public_record_flags: check names against mentions of BOP (Bureau of Prisons), OFAC, SEC enforcement, or prior litigation in the documents — only flag if actually mentioned.
"""


class WitnessAnalyst:
    """
    Tier 4+ witness intelligence dossier builder.
    Builds one dossier per key witness (top 10 by doc frequency).
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('witness_intelligence')

    def build_dossiers(self, run_id: str, role: str, goal_text: str,
                       entities: List[Dict], documents: List[Dict],
                       contradictions: List[Dict], financial: List[Dict],
                       max_witnesses: int = 10) -> List[Dict[str, Any]]:
        """
        Build witness dossiers for top N key persons.

        Returns list of witness dossier dicts.
        """
        # Identify key persons (entity_type = 'person' or 'attorney')
        persons = [
            e for e in entities
            if e.get('entity_type') in ('person', 'attorney', 'witness', 'expert')
            and not e.get('merged_into')
        ]

        if not persons:
            logger.info(f"WitnessAnalyst run {run_id}: no persons found")
            return []

        # Rank by provenance count (most-referenced first)
        def prov_count(e):
            try:
                return len(json.loads(e.get('provenance') or '[]'))
            except Exception:
                return 0

        persons.sort(key=prov_count, reverse=True)
        top_persons = persons[:max_witnesses]

        # Build doc map for fast lookup
        doc_map = {d.get('id', d.get('doc_id', 0)): d for d in documents}

        # Build context strings once (shared across all witnesses)
        contradictions_summary = self._format_contradictions(contradictions)
        financial_summary = self._format_financial(financial)
        entities_summary = self._format_entities(entities)

        dossiers = []
        for person in top_persons:
            try:
                dossier = self._build_one_dossier(
                    run_id, role, goal_text, person, doc_map,
                    contradictions_summary, financial_summary, entities_summary
                )
                if dossier:
                    dossiers.append(dossier)
            except Exception as e:
                logger.warning(f"WitnessAnalyst: dossier failed for {person.get('name')}: {e}")

        logger.info(f"WitnessAnalyst run {run_id}: built {len(dossiers)} dossiers")
        return dossiers

    def _build_one_dossier(self, run_id: str, role: str, goal_text: str,
                            person: Dict, doc_map: Dict,
                            contradictions_summary: str, financial_summary: str,
                            entities_summary: str) -> Optional[Dict]:
        """Build a dossier for a single witness."""
        witness_docs = self._get_witness_docs(person, doc_map)

        prompt = WITNESS_INTELLIGENCE_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Build the strongest case',
            witness_name=person.get('name', 'Unknown'),
            witness_role=person.get('role_in_case', 'Unknown role'),
            witness_docs=witness_docs[:5000],
            contradictions_summary=contradictions_summary[:2500],
            financial_summary=financial_summary[:2000],
            entities_summary=entities_summary[:1500],
        )

        result = self._call_llm_with_escalation(prompt, 'ci:witness_intelligence')
        return result

    def _get_witness_docs(self, person: Dict, doc_map: Dict) -> str:
        """Find all documents referencing this person and format for prompt."""
        try:
            provenance = json.loads(person.get('provenance') or '[]')
        except Exception:
            provenance = []

        parts = []
        seen_ids = set()
        for prov in provenance[:15]:
            doc_id = prov.get('paperless_doc_id')
            if not doc_id or doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            doc = doc_map.get(doc_id, {})
            title = doc.get('title', f'Doc #{doc_id}')
            content = (doc.get('content') or '')[:600]
            excerpt = prov.get('excerpt', '')
            parts.append(
                f"[Doc #{doc_id}] {title}\n"
                f"  Excerpt mentioning witness: {excerpt}\n"
                f"  Document context: {content}"
            )

        return '\n\n'.join(parts) if parts else f"No documents found for {person.get('name')}"

    def _format_contradictions(self, contradictions: List[Dict]) -> str:
        if not contradictions:
            return 'No contradictions documented.'
        parts = []
        for c in contradictions[:20]:
            parts.append(
                f"[{c.get('severity', '?').upper()}] {c.get('description', '')}\n"
                f"  Doc A: #{c.get('doc_a_provenance', [{}])[0].get('paperless_doc_id', '?') if c.get('doc_a_provenance') else '?'}\n"
                f"  Doc B: #{c.get('doc_b_provenance', [{}])[0].get('paperless_doc_id', '?') if c.get('doc_b_provenance') else '?'}"
            )
        return '\n'.join(parts)

    def _format_financial(self, data: List[Dict]) -> str:
        if not data:
            return 'No financial transactions documented.'
        parts = []
        for item in data[:25]:
            if isinstance(item, dict):
                parts.append(
                    f"{item.get('date', '?')}: ${item.get('amount', item.get('amount_usd', 0)):,} "
                    f"— {item.get('description', '')} [Doc #{item.get('doc_id', '?')}]"
                )
        return '\n'.join(parts)

    def _format_entities(self, entities: List[Dict]) -> str:
        parts = []
        for e in entities[:25]:
            if not e.get('merged_into'):
                parts.append(f"{e.get('entity_type', '?')}: {e.get('name', '?')} — {e.get('role_in_case', '')}")
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
                logger.error(f"WitnessAnalyst {provider}/{model} failed: {e}")
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
            logger.warning(f"WitnessAnalyst JSON error for witness: {e}")
            return None
        except Exception as e:
            logger.error(f"WitnessAnalyst LLM error: {e}")
            return None
