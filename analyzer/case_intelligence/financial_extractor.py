"""
Financial Extractor — Tier 1 CI Pipeline Stage.

Extracts monetary amounts, account balances, payment transfers,
and financial discrepancies from documents.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

FINANCIAL_EXTRACTION_PROMPT = """You are a forensic financial analyst for legal cases.

Extract ALL financial information from the document below. Focus on amounts, dates, parties, and any discrepancies or unusual patterns.

Document ID: {doc_id}
Document Title: {title}

DOCUMENT CONTENT:
{content}

Respond in JSON:
{{
  "financial_facts": [
    {{
      "fact_type": "balance|transfer|payment|fee|judgment|lien|debt|income|asset|other",
      "description": "clear factual description",
      "amount_usd": 12345.67,
      "amount_raw": "original text (e.g., '$12,345.67')",
      "currency": "USD",
      "as_of_date": "YYYY-MM-DD or null",
      "from_party": "payor/source name or null",
      "to_party": "payee/destination name or null",
      "account_ref": "account number or identifier or null",
      "is_disputed": false,
      "discrepancy_note": "any anomaly or inconsistency noted",
      "provenance": [
        {{
          "paperless_doc_id": {doc_id},
          "page_number": 1,
          "excerpt": "exact quote containing the amount (max 200 chars)",
          "model_used": "{model_used}"
        }}
      ]
    }}
  ],
  "summary": {{
    "total_amounts_found": 0,
    "largest_amount_usd": null,
    "date_range": "YYYY-MM-DD to YYYY-MM-DD",
    "suspected_discrepancies": 0,
    "notes": "any overall financial observations"
  }}
}}

RULES:
1. Every financial fact MUST have provenance with an exact excerpt.
2. Convert all amounts to USD float (use null if currency unknown).
3. Flag discrepancies (amounts that don't add up, missing payments, etc.).
4. Do NOT fabricate amounts not in the document.
5. Include beginning and ending balances if present.
"""


class FinancialExtractor:
    """
    Tier 1 financial extractor with gpt-4o-mini, escalation to gpt-4o.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('financial_extraction')

    def extract(self, doc_id: int, title: str, content: str,
                run_id: str = None) -> Dict[str, Any]:
        """
        Extract financial facts from a document.

        Returns dict with 'financial_facts' list and 'summary'.
        """
        if not content or len(content.strip()) < 50:
            return {'financial_facts': [], 'summary': {}}

        extraction_ts = datetime.now(timezone.utc).isoformat()
        content_truncated = content[:7000]  # Financial docs can be dense

        prompt = FINANCIAL_EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            title=title,
            content=content_truncated,
            model_used=self.task_def.primary_model,
        )

        result = self._call_with_escalation(prompt, doc_id)
        if not result:
            return {'financial_facts': [], 'summary': {}}

        facts = result.get('financial_facts', [])
        for fact in facts:
            for prov in fact.get('provenance', []):
                prov['extraction_version'] = extraction_ts
                prov['prompt_version'] = 'financial_v1'
                if 'paperless_doc_id' not in prov:
                    prov['paperless_doc_id'] = doc_id

        return {
            'financial_facts': facts,
            'summary': result.get('summary', {}),
        }

    def _call_with_escalation(self, prompt: str, doc_id: int) -> Optional[dict]:
        for model_key in ('primary', 'escalate'):
            model = (self.task_def.primary_model if model_key == 'primary'
                     else self.task_def.escalate_model)
            provider = (self.task_def.primary_provider if model_key == 'primary'
                        else self.task_def.escalate_provider)

            client = self.llm_clients.get(provider)
            if not client:
                continue

            try:
                result = self._call_llm(client, model, prompt, provider, doc_id)
                if result and result.get('financial_facts') is not None:
                    return result
                if model_key == 'primary':
                    logger.debug(f"FinancialExtractor: escalating doc {doc_id}")
            except Exception as e:
                logger.error(f"FinancialExtractor: {provider}/{model} failed: {e}")

        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, doc_id: int) -> Optional[dict]:
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
                        operation='ci:financial_extraction',
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
                        operation='ci:financial_extraction',
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        document_id=doc_id,
                    )
            else:
                return None

            return json.loads(text)
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.error(f"FinancialExtractor LLM call failed: {e}")
            return None
