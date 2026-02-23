"""
Entity Extractor — Tier 1 CI Pipeline Stage.

Extracts people, organizations, accounts, properties, phone numbers,
emails, and other identifiers from a single document using gpt-4o-mini
(escalates to gpt-4o if citations are missing).
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.provenance import (
    Provenance, provenance_list_to_json, missing_required_citations
)
from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """You are a legal document entity extractor.

Extract ALL identifiable entities from the document excerpt below. For each entity, you MUST include the exact text excerpt that supports the extraction (provenance).

Document ID: {doc_id}
Document Title: {title}

DOCUMENT CONTENT:
{content}

Respond in JSON:
{{
  "entities": [
    {{
      "entity_type": "person|org|address|account|property|court|phone|email|identifier",
      "name": "canonical name",
      "aliases": ["alternate name 1", "alternate name 2"],
      "role_in_case": "brief description of role",
      "attributes": {{
        "account_number": "...",
        "address": "...",
        "ssn_last4": "...",
        "company_type": "...",
        "jurisdiction": "..."
      }},
      "notes": "any relevant notes",
      "provenance": [
        {{
          "paperless_doc_id": {doc_id},
          "page_number": 1,
          "excerpt": "exact quote from document (max 200 chars)",
          "model_used": "{model_used}"
        }}
      ]
    }}
  ]
}}

RULES:
1. Include ONLY entities clearly present in the document.
2. Every entity MUST have at least one provenance entry with a real excerpt.
3. Use null for attributes not found in the document.
4. Merge duplicate entities (same person/org with slight name variations).
5. Do NOT invent or infer entities not in the text.
"""


class EntityExtractor:
    """
    Tier 1 entity extractor using gpt-4o-mini with escalation to gpt-4o.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        """
        Args:
            llm_clients: dict mapping provider name → LLMClient instance
                         e.g., {'openai': openai_client, 'anthropic': claude_client}
            usage_tracker: LLMUsageTracker instance
        """
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('entity_extraction')

    def extract(self, doc_id: int, title: str, content: str,
                run_id: str = None) -> List[Dict[str, Any]]:
        """
        Extract entities from a single document.

        Returns a list of entity dicts ready to insert into ci_entities.
        """
        if not content or len(content.strip()) < 50:
            return []

        extraction_ts = datetime.now(timezone.utc).isoformat()
        content_truncated = content[:6000]  # Limit for Tier 1

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            title=title,
            content=content_truncated,
            model_used=self.task_def.primary_model,
        )

        result = self._call_with_escalation(prompt, doc_id, extraction_ts)
        if not result:
            return []

        entities = result.get('entities', [])
        # Post-process: ensure extraction_version in provenance
        for entity in entities:
            for prov in entity.get('provenance', []):
                prov['extraction_version'] = extraction_ts
                prov['prompt_version'] = 'entity_v1'
                if 'paperless_doc_id' not in prov:
                    prov['paperless_doc_id'] = doc_id

        return entities

    def _call_with_escalation(self, prompt: str, doc_id: int,
                               extraction_ts: str) -> Optional[dict]:
        """Call primary model; escalate if citation validation fails."""
        for model_key in ('primary', 'escalate'):
            model = (self.task_def.primary_model if model_key == 'primary'
                     else self.task_def.escalate_model)
            provider = (self.task_def.primary_provider if model_key == 'primary'
                        else self.task_def.escalate_provider)

            client = self.llm_clients.get(provider)
            if not client:
                logger.warning(f"EntityExtractor: no client for provider {provider}")
                continue

            try:
                result = self._call_llm(client, model, prompt, provider, doc_id)
                if result and not missing_required_citations(result):
                    return result
                if model_key == 'primary':
                    logger.debug(f"EntityExtractor: escalating doc {doc_id} (citations missing)")
            except Exception as e:
                logger.error(f"EntityExtractor: {provider}/{model} failed for doc {doc_id}: {e}")

        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, doc_id: int) -> Optional[dict]:
        """Raw LLM call with JSON parsing and usage tracking."""
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
                        provider=provider, model=model,
                        operation='ci:entity_extraction',
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        document_id=doc_id,
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
                        provider=provider, model=model,
                        operation='ci:entity_extraction',
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        document_id=doc_id,
                    )
            else:
                return None

            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"EntityExtractor: JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"EntityExtractor LLM call failed: {e}")
            return None
