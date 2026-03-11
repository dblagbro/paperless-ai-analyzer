"""
Deep Financial Forensics — Tier 5 (White Glove) CI Pipeline Phase 2F+.

Extends basic forensic accounting with:
  - Benford's Law first-digit analysis (detects fabricated transaction amounts)
  - Beneficial ownership tracing (who ultimately controls each entity)
  - Round-trip transaction detection (A→B→C→A money circles)
  - Shell entity identification (no business purpose, offshore, suspicious patterns)
  - Advanced layering analysis (structured hops to obscure money source)
  - Suspicious coincidence clustering (same-date multi-party transactions)

Output stored in ci_deep_forensics.
"""

import json
import logging
import math
from collections import Counter
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

# Expected Benford's Law first-digit probabilities
BENFORDS_EXPECTED = {
    1: 0.30103, 2: 0.17609, 3: 0.12494, 4: 0.09691,
    5: 0.07918, 6: 0.06695, 7: 0.05799, 8: 0.05115, 9: 0.04576,
}

DEEP_FORENSICS_PROMPT = """You are a forensic accountant and financial crime specialist performing White Glove Tier 5 deep financial analysis for litigation.

CASE ROLE: {role}
CASE GOAL: {goal_text}

FINANCIAL DATA:
{financial_data}

ENTITIES:
{entities_data}

TIMELINE:
{timeline_data}

BENFORD'S LAW ANALYSIS (deterministic, pre-computed):
{benfords_summary}

Your task: Perform deep financial forensics analysis. Focus on:

1. BENEFICIAL OWNERSHIP: For each organization/company, trace who ultimately controls/benefits from it. Look for hidden ownership chains, nominee directors, related-party relationships, and undisclosed beneficiaries. Who is behind the entities?

2. ROUND-TRIP TRANSACTIONS: Identify money that flows out and comes back — A pays B, B pays C, C pays A (in any length chain). These are often used to create false revenue, hide assets, or disguise loans as income.

3. SHELL ENTITY FLAGS: Identify entities with characteristics of shell companies — no described business purpose, offshore jurisdictions (BVI, Cayman, Delaware LLCs with no operations), naming patterns suggesting special purpose vehicles (Holdings, Ventures, Investments LLC), or entities that only appear as pass-through intermediaries.

4. ADVANCED STRUCTURING: Beyond simple threshold avoidance — look for patterns of exactly N transactions at just-under threshold amounts, especially across multiple time periods or multiple payors to the same payee.

5. LAYERING: Multi-step fund movement designed to obscure origin. Unlike simple A→B→C tracing (Phase 2F Tier 3), identify patterns suggesting deliberate obscuring of beneficial ownership: rapid same-day transfers between related entities, back-and-forth transfers, use of professional nominees.

6. SUSPICIOUS CLUSTERS: Multiple significant transactions on the exact same date across different parties — may indicate coordinated execution of a scheme.

7. BENFORD'S INTERPRETATION: If the deterministic analysis above flagged a significant deviation, explain what types of fraud or manipulation this pattern is consistent with given the specific transaction context.

Respond in JSON:
{{
  "beneficial_ownership": [
    {{
      "entity_name": "XYZ Holdings LLC",
      "entity_type": "organization",
      "known_ownership": "John Smith (50%), disclosed",
      "suspected_beneficial_owner": "John Smith through nominee — see Doc #45 CC pattern",
      "ownership_chain": ["XYZ Holdings LLC", "ABC Ventures Ltd (BVI)", "John Smith"],
      "red_flags": ["BVI jurisdiction", "no described business purpose", "only appears as payee"],
      "confidence": "high|medium|low",
      "provenance": [{{"paperless_doc_id": 45, "excerpt": "..."}}]
    }}
  ],
  "round_trip_transactions": [
    {{
      "description": "Describe the round trip",
      "chain": ["Party A", "Party B", "Party C", "Party A"],
      "amounts_usd": [10000, 9500, 9000],
      "dates": ["2023-01-15", "2023-01-20", "2023-02-01"],
      "total_usd": 10000,
      "purpose_claimed": "What the parties say it was for",
      "actual_effect": "What this actually accomplished financially",
      "significance": "high|medium|low",
      "provenance": [{{"paperless_doc_id": 101, "excerpt": "..."}}]
    }}
  ],
  "shell_entity_flags": [
    {{
      "entity_name": "XYZ Holdings LLC",
      "shell_indicators": ["No disclosed business purpose", "Offshore jurisdiction", "Only appears as payment recipient"],
      "jurisdiction": "British Virgin Islands",
      "transaction_count": 5,
      "total_received_usd": 250000,
      "total_paid_usd": 0,
      "assessment": "Appears to be a pass-through vehicle — recommend subpoena of formation documents and bank records",
      "provenance": [{{"paperless_doc_id": 33}}]
    }}
  ],
  "advanced_structuring": [
    {{
      "description": "Pattern of structuring behavior",
      "threshold_targeted_usd": 10000,
      "transactions": [
        {{"amount_usd": 9800, "date": "2023-01-05", "from": "A", "to": "B"}},
        {{"amount_usd": 9900, "date": "2023-01-12", "from": "A", "to": "B"}}
      ],
      "total_usd": 19700,
      "pattern_span_days": 30,
      "significance": "high",
      "provenance": [{{"paperless_doc_id": 55, "excerpt": "..."}}]
    }}
  ],
  "layering_schemes": [
    {{
      "description": "Multi-step obscuring of fund origin",
      "steps": [
        {{"step": 1, "from": "A", "to": "B", "amount_usd": 100000, "date": "2023-01-10", "doc_id": 10}},
        {{"step": 2, "from": "B", "to": "C", "amount_usd": 95000, "date": "2023-01-11", "doc_id": 11}},
        {{"step": 3, "from": "C", "to": "A nominee", "amount_usd": 90000, "date": "2023-01-15", "doc_id": 12}}
      ],
      "ultimate_beneficiary": "Person or entity that ultimately received the funds",
      "amount_laundered_usd": 90000,
      "significance": "high",
      "provenance": [{{"paperless_doc_id": 10}}]
    }}
  ],
  "suspicious_clusters": [
    {{
      "date": "2023-01-15",
      "transactions": [
        {{"parties": ["A", "B"], "amount_usd": 50000}},
        {{"parties": ["C", "D"], "amount_usd": 49000}}
      ],
      "why_suspicious": "Four unrelated parties all transacted on same date for similar amounts — may indicate coordinated scheme",
      "provenance": [{{"paperless_doc_id": 20}}]
    }}
  ],
  "benford_interpretation": "Narrative interpretation of the Benford's Law deviation (if any) given the specific transaction context",
  "summary": "Deep financial forensics memo (4-6 paragraphs). Open with the single most important finding. Describe what the money flows suggest about the overall scheme. Be specific, actionable, and written for senior litigation counsel.",
  "risk_score": 0,
  "highest_priority_investigation": "The single most important financial forensics lead for investigators to pursue next"
}}

RULES:
1. Only flag patterns actually supported by the documents provided.
2. Cite specific document IDs for every finding.
3. risk_score: 0-100 (0=no financial crime indicators, 100=clear evidence of fraud/money laundering).
4. If a category has no findings, return an empty array for that field.
5. Never fabricate amounts or dates.
"""


class DeepFinancialForensics:
    """
    Tier 5 deep financial forensics.
    Extends Tier 3 forensic accounting with Benford's law, beneficial ownership, and layering.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('deep_financial_forensics')

    def analyze(self, run_id: str, role: str, goal_text: str,
                financial_data: List[Dict], timeline_data: List[Dict],
                entities_data: List[Dict]) -> Dict[str, Any]:
        """
        Run deep financial forensics analysis.

        Args:
            run_id: CI run ID (for logging)
            role: plaintiff/defense/neutral
            goal_text: Case goal description
            financial_data: Financial extractions from Phase 1
            timeline_data: Timeline events from Phase 1
            entities_data: Entities from Phase 1

        Returns:
            Deep forensics report dict (mirrors ci_deep_forensics schema)
        """
        if not financial_data and not timeline_data:
            logger.info(f"DeepFinancialForensics run {run_id}: no financial data")
            return self._empty_report()

        # Deterministic Benford's Law analysis (no LLM needed)
        benfords_result = self._run_benfords_analysis(financial_data)
        benfords_summary = self._format_benfords_summary(benfords_result)

        financial_str = self._format_financial(financial_data)
        timeline_str = self._format_timeline(timeline_data)
        entities_str = self._format_entities(entities_data)

        prompt = DEEP_FORENSICS_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Analyze all financial transactions',
            financial_data=financial_str[:10000],
            timeline_data=timeline_str[:3000],
            entities_data=entities_str[:2000],
            benfords_summary=benfords_summary,
        )

        result = self._call_llm_with_escalation(prompt, 'ci:deep_financial_forensics')
        if not result:
            return self._empty_report()

        # Attach deterministic Benford's result
        result['benford_analysis'] = benfords_result

        logger.info(
            f"DeepFinancialForensics run {run_id}: "
            f"{len(result.get('round_trip_transactions', []))} round trips, "
            f"{len(result.get('shell_entity_flags', []))} shell entities, "
            f"risk_score={result.get('risk_score', 0)}"
        )
        return result

    # ── Benford's Law (deterministic) ────────────────────────────────────────

    def _run_benfords_analysis(self, financial_data: List[Dict]) -> Dict:
        """Compute first-digit distribution and chi-squared test."""
        amounts = []
        for item in financial_data:
            if not isinstance(item, dict):
                continue
            raw = item.get('amount') or item.get('amount_usd') or 0
            try:
                val = float(str(raw).replace(',', '').replace('$', '').strip())
                if val > 0:
                    amounts.append(val)
            except (ValueError, TypeError):
                pass

        if len(amounts) < 10:
            return {
                'sample_size': len(amounts),
                'sufficient_data': False,
                'note': 'Insufficient sample size for Benford\'s analysis (need ≥10 transactions)',
            }

        # Count first digits
        first_digits = Counter()
        for a in amounts:
            fd = int(str(a).lstrip('0').replace('.', '')[0])
            if 1 <= fd <= 9:
                first_digits[fd] += 1

        total = sum(first_digits.values())
        observed = {d: first_digits.get(d, 0) / total for d in range(1, 10)}

        # Chi-squared test
        chi2 = sum(
            total * (observed.get(d, 0) - BENFORDS_EXPECTED[d]) ** 2 / BENFORDS_EXPECTED[d]
            for d in range(1, 10)
        )

        # p-value approximation (8 degrees of freedom, chi2 critical values)
        # chi2 > 15.507 → p < 0.05 (significant); > 20.090 → p < 0.01 (highly significant)
        if chi2 > 20.090:
            significance = 'highly_significant'
            interpretation = 'Strong deviation — consistent with fabricated, rounded, or manipulated amounts'
        elif chi2 > 15.507:
            significance = 'significant'
            interpretation = 'Moderate deviation — warrants further investigation'
        elif chi2 > 11.070:
            significance = 'borderline'
            interpretation = 'Slight deviation — monitor but not conclusive'
        else:
            significance = 'normal'
            interpretation = 'Distribution consistent with naturally occurring financial data'

        deviations = []
        for d in range(1, 10):
            obs_pct = observed.get(d, 0) * 100
            exp_pct = BENFORDS_EXPECTED[d] * 100
            diff = obs_pct - exp_pct
            if abs(diff) > 3:
                deviations.append({
                    'digit': d,
                    'observed_pct': round(obs_pct, 1),
                    'expected_pct': round(exp_pct, 1),
                    'deviation': round(diff, 1),
                    'direction': 'over' if diff > 0 else 'under',
                })

        return {
            'sample_size': len(amounts),
            'sufficient_data': True,
            'chi2_statistic': round(chi2, 3),
            'significance': significance,
            'interpretation': interpretation,
            'digit_distribution': [
                {
                    'digit': d,
                    'observed_pct': round(observed.get(d, 0) * 100, 1),
                    'expected_pct': round(BENFORDS_EXPECTED[d] * 100, 1),
                    'count': first_digits.get(d, 0),
                }
                for d in range(1, 10)
            ],
            'notable_deviations': deviations,
        }

    def _format_benfords_summary(self, result: Dict) -> str:
        if not result.get('sufficient_data'):
            return result.get('note', 'Insufficient data for Benford\'s analysis.')
        lines = [
            f"Sample: {result['sample_size']} transactions",
            f"Chi-squared: {result['chi2_statistic']} — {result['significance'].replace('_', ' ').upper()}",
            f"Interpretation: {result['interpretation']}",
        ]
        if result.get('notable_deviations'):
            devs = ', '.join(
                f"digit {d['digit']}: {d['observed_pct']}% observed vs {d['expected_pct']}% expected"
                for d in result['notable_deviations'][:4]
            )
            lines.append(f"Notable deviations: {devs}")
        return '\n'.join(lines)

    # ── Formatting helpers ────────────────────────────────────────────────────

    def _format_financial(self, data: List[Dict]) -> str:
        if not data:
            return 'No financial data extracted.'
        parts = []
        for item in data[:150]:
            if isinstance(item, dict):
                doc_id = item.get('doc_id') or item.get('paperless_doc_id', '?')
                parts.append(
                    f"[Doc #{doc_id}] {item.get('description', '')} — "
                    f"Amount: {item.get('amount', item.get('amount_usd', '?'))} "
                    f"Date: {item.get('date', 'unknown')} "
                    f"Parties: {item.get('parties', item.get('from_party', ''))} "
                    f"→ {item.get('to_party', '')} "
                    f"Type: {item.get('transaction_type', '')}"
                )
        return '\n'.join(parts) if parts else str(data)[:5000]

    def _format_timeline(self, events: List[Dict]) -> str:
        if not events:
            return 'No timeline data.'
        parts = []
        for ev in events[:50]:
            if isinstance(ev, dict):
                parts.append(
                    f"{ev.get('event_date', '?')}: {ev.get('description', '')} "
                    f"[{ev.get('significance', '?')}]"
                )
        return '\n'.join(parts)

    def _format_entities(self, entities: List[Dict]) -> str:
        if not entities:
            return 'No entities.'
        parts = []
        for e in entities[:40]:
            if isinstance(e, dict):
                parts.append(
                    f"{e.get('entity_type', '?')}: {e.get('name', '?')} — "
                    f"{e.get('role_in_case', '')} "
                    f"[aliases: {e.get('aliases', '')}]"
                )
        return '\n'.join(parts)

    # ── LLM call ─────────────────────────────────────────────────────────────

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
                logger.error(f"DeepFinancialForensics {provider}/{model} failed: {e}")
        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, operation: str) -> Optional[Dict]:
        text = ''
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
            logger.warning(f"DeepFinancialForensics JSON error: {e} — preview={text[:300]!r}")
            return None
        except Exception as e:
            logger.error(f"DeepFinancialForensics LLM error: {e}")
            return None

    @staticmethod
    def _empty_report() -> Dict:
        return {
            'beneficial_ownership': [],
            'round_trip_transactions': [],
            'shell_entity_flags': [],
            'advanced_structuring': [],
            'layering_schemes': [],
            'suspicious_clusters': [],
            'benford_analysis': {
                'sufficient_data': False,
                'note': 'No financial data available.',
            },
            'benford_interpretation': '',
            'summary': 'Insufficient financial data for deep forensics analysis.',
            'risk_score': 0,
            'highest_priority_investigation': '',
        }
