"""
Timeline Builder — Tier 1 CI Pipeline Stage.

Extracts dated events from documents and classifies them by type and
significance. Used to build the case chronology.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

TIMELINE_EXTRACTION_PROMPT = """You are a legal timeline analyst.

Extract ALL dated events from the document below. Include every event that has a date (or approximate date), describes a factual occurrence, and may be relevant to legal proceedings.

Document ID: {doc_id}
Document Title: {title}

DOCUMENT CONTENT:
{content}

Respond in JSON:
{{
  "events": [
    {{
      "event_date": "YYYY-MM-DD or YYYY-MM or YYYY-QN or 'circa YYYY' for approximate",
      "date_approx": false,
      "event_type": "filing|payment|communication|court_event|transfer|meeting|service|deadline|other",
      "description": "clear factual description of the event (1-2 sentences)",
      "significance": "low|medium|high|critical",
      "significance_reason": "why this event matters legally",
      "parties": ["Party Name 1", "Party Name 2"],
      "provenance": [
        {{
          "paperless_doc_id": {doc_id},
          "page_number": 1,
          "excerpt": "exact quote establishing the date/event (max 200 chars)",
          "model_used": "{model_used}"
        }}
      ]
    }}
  ]
}}

RULES:
1. Every event MUST have at least one provenance entry with a real excerpt.
2. Use "date_approx": true for relative dates (e.g., "last month", "in 2023").
3. Set significance to "critical" for: defaults, judgments, service of process, key transfers.
4. Include ALL parties mentioned in connection with the event.
5. Do NOT invent dates or events not supported by the document text.
"""


class TimelineBuilder:
    """
    Tier 1 timeline extractor using gpt-4o-mini with escalation to gpt-4o.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('timeline_extraction')

    def extract(self, doc_id: int, title: str, content: str,
                run_id: str = None) -> List[Dict[str, Any]]:
        """
        Extract timeline events from a single document.
        Returns a list of event dicts ready for ci_timeline_events.
        """
        if not content or len(content.strip()) < 50:
            return []

        extraction_ts = datetime.now(timezone.utc).isoformat()
        content_truncated = content[:6000]

        prompt = TIMELINE_EXTRACTION_PROMPT.format(
            doc_id=doc_id,
            title=title,
            content=content_truncated,
            model_used=self.task_def.primary_model,
        )

        result = self._call_with_escalation(prompt, doc_id)
        if not result:
            return []

        events = result.get('events', [])
        for event in events:
            for prov in event.get('provenance', []):
                prov['extraction_version'] = extraction_ts
                prov['prompt_version'] = 'timeline_v1'
                if 'paperless_doc_id' not in prov:
                    prov['paperless_doc_id'] = doc_id

        return events

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
                if result and result.get('events'):
                    # Validate at least some provenance exists
                    has_prov = any(
                        ev.get('provenance') for ev in result.get('events', [])
                    )
                    if has_prov:
                        return result
                if model_key == 'primary':
                    logger.debug(f"TimelineBuilder: escalating doc {doc_id}")
            except Exception as e:
                logger.error(f"TimelineBuilder: {provider}/{model} failed: {e}")

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
                        operation='ci:timeline_extraction',
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
                        operation='ci:timeline_extraction',
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
            logger.error(f"TimelineBuilder LLM call failed: {e}")
            return None
