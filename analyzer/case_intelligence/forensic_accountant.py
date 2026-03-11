"""
Forensic Accountant — Tier 3+ CI Pipeline Phase 2F.

Analyzes all financial extractions for:
  - Structuring (transactions just below reporting thresholds)
  - Round-number anomalies ("too clean" amounts)
  - Timing correlations (transfers near key legal dates)
  - Cash-flow reconciliation (do debits = credits per party?)
  - Balance discontinuities (jumps without explaining transactions)
  - Missing counterparts (payment without receipt, etc.)
  - Multi-hop transaction tracing (A→B→C→D chains)
  - Intracompany transaction masking

Output stored in ci_forensic_report.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

FORENSIC_ACCOUNTING_PROMPT = """You are a forensic accountant and financial investigator analyzing documents for litigation.

CASE ROLE: {role}
CASE GOAL: {goal_text}

FINANCIAL DATA EXTRACTED FROM CASE DOCUMENTS:
{financial_data}

TIMELINE (for timing correlation):
{timeline_data}

ENTITIES (parties):
{entities_data}

Your task: Perform a thorough forensic accounting analysis. Look for:

1. STRUCTURING: Transactions just below $10,000 or $5,000 reporting thresholds (e.g., $9,900 wires)
2. ROUND-NUMBER ANOMALIES: Suspiciously "clean" amounts (exactly $50,000, $100,000) that suggest estimates, not real transactions
3. TIMING CORRELATIONS: Transfers within 7 days of court hearings, contract signings, terminations, or other key events
4. CASH-FLOW RECONCILIATION: Sum all documented inflows vs outflows per party — flag when they don't balance
5. BALANCE DISCONTINUITIES: Running balance that jumps without an explaining transaction
6. MISSING COUNTERPARTS: Payment recorded but no receipt; invoice without corresponding payment
7. MULTI-HOP TRACING: Money flowing A→B→C — follow the full chain to identify ultimate beneficial recipient
8. INTRACOMPANY: Loans/transfers between related entities at non-arm's-length terms

Respond in JSON:
{{
  "cash_flow_by_party": [
    {{
      "party": "Party name",
      "total_in_usd": 0,
      "total_out_usd": 0,
      "net_usd": 0,
      "transaction_count": 0,
      "flags": ["Note any anomalies for this party"],
      "provenance": [{{"paperless_doc_id": 101, "excerpt": "..."}}]
    }}
  ],
  "flagged_transactions": [
    {{
      "type": "structuring|round_number|timing_correlation|gap|multi_hop|intracompany",
      "amount_usd": 0,
      "parties": ["Sender", "Recipient"],
      "date": "YYYY-MM-DD",
      "description": "Description of the suspicious pattern",
      "significance": "high|medium|low",
      "threshold_proximity_usd": 0,
      "related_event": "Court hearing on YYYY-MM-DD (timing correlation)",
      "provenance": [{{"paperless_doc_id": 101, "page_number": 1, "excerpt": "..."}}]
    }}
  ],
  "balance_discrepancies": [
    {{
      "account_or_party": "Account or party name",
      "expected_balance_usd": 0,
      "documented_balance_usd": 0,
      "gap_usd": 0,
      "explanation": "Why this gap matters",
      "provenance": [{{"paperless_doc_id": 101}}]
    }}
  ],
  "missing_transactions": [
    "Description of what should exist based on documents but is absent"
  ],
  "transaction_chains": [
    {{
      "from_party": "A",
      "to_party": "D (ultimate recipient)",
      "via": ["B", "C"],
      "total_usd": 0,
      "description": "Money flow narrative"
    }}
  ],
  "summary": "Forensic accounting memo (3-5 paragraphs summarizing key findings, implications, and recommended follow-up)",
  "total_documented_exposure_usd": 0
}}

RULES:
1. Only flag patterns actually supported by the document excerpts provided.
2. Cite specific document IDs for every flagged item.
3. Be precise about dollar amounts — never fabricate.
4. If data is insufficient to flag a pattern, omit it rather than speculate.
5. The summary should read like a memo to lead counsel — professional, specific, actionable.
"""


class ForensicAccountant:
    """
    Tier 3+ forensic accounting analysis.
    Analyzes all financial data from Phase 1 for suspicious patterns.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('forensic_accounting')

    def analyze(self, run_id: str, role: str, goal_text: str,
                financial_data: List[Dict], timeline_data: List[Dict],
                entities_data: List[Dict]) -> Dict[str, Any]:
        """
        Run forensic accounting analysis.

        Args:
            run_id: CI run ID (for logging)
            role: plaintiff/defense/neutral
            goal_text: Case goal description
            financial_data: List of financial extraction dicts from Phase 1
            timeline_data: List of timeline events from Phase 1
            entities_data: List of entities from Phase 1

        Returns:
            Forensic report dict (mirrors ci_forensic_report schema)
        """
        if not financial_data and not timeline_data:
            logger.info(f"ForensicAccountant run {run_id}: no financial data to analyze")
            return self._empty_report()

        financial_str = self._format_financial(financial_data)
        timeline_str = self._format_timeline(timeline_data)
        entities_str = self._format_entities(entities_data)

        prompt = FORENSIC_ACCOUNTING_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Analyze all financial transactions',
            financial_data=financial_str[:8000],
            timeline_data=timeline_str[:3000],
            entities_data=entities_str[:2000],
        )

        result = self._call_llm_with_escalation(prompt, 'ci:forensic_accounting')
        if not result:
            return self._empty_report()

        logger.info(
            f"ForensicAccountant run {run_id}: "
            f"{len(result.get('flagged_transactions', []))} flagged transactions, "
            f"exposure=${result.get('total_documented_exposure_usd', 0):,.0f}"
        )
        return result

    def _format_financial(self, data: List[Dict]) -> str:
        if not data:
            return 'No financial data extracted.'
        parts = []
        for item in data[:100]:  # limit
            if isinstance(item, dict):
                doc_id = item.get('doc_id') or item.get('paperless_doc_id', '?')
                parts.append(
                    f"[Doc #{doc_id}] {item.get('description', '')} — "
                    f"Amount: {item.get('amount', item.get('amount_usd', '?'))} "
                    f"Date: {item.get('date', 'unknown')} "
                    f"Parties: {item.get('parties', item.get('from_party', ''))} → {item.get('to_party', '')}"
                )
        return '\n'.join(parts) if parts else str(data)[:4000]

    def _format_timeline(self, events: List[Dict]) -> str:
        if not events:
            return 'No timeline data.'
        parts = []
        for ev in events[:50]:
            if isinstance(ev, dict):
                parts.append(
                    f"{ev.get('event_date', '?')}: {ev.get('description', '')} "
                    f"[significance: {ev.get('significance', '?')}]"
                )
        return '\n'.join(parts)

    def _format_entities(self, entities: List[Dict]) -> str:
        if not entities:
            return 'No entities.'
        parts = []
        for e in entities[:30]:
            if isinstance(e, dict):
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
                logger.error(f"ForensicAccountant {provider}/{model} failed: {e}")
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
            logger.warning(f"ForensicAccountant JSON error: {e} — text_len={len(text)}, preview={text[:300]!r}")
            return None
        except Exception as e:
            logger.error(f"ForensicAccountant LLM error: {e}")
            return None

    @staticmethod
    def _empty_report() -> Dict:
        return {
            'cash_flow_by_party': [],
            'flagged_transactions': [],
            'balance_discrepancies': [],
            'missing_transactions': [],
            'transaction_chains': [],
            'summary': 'Insufficient financial data for forensic accounting analysis.',
            'total_documented_exposure_usd': 0,
        }
