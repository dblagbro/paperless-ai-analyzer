"""
War Room — Tier 4+ CI Pipeline Phase 2R.

Simulates the opposing counsel's entire case strategy. Replaces the basic
opposing_theory_generation with a full war room simulation.

Output includes:
  - Opposing case summary (2-3 paragraphs as lead opposing counsel)
  - Top 3 most dangerous arguments + pre-drafted responses
  - Top 5 client vulnerabilities with mitigation strategies
  - Documents most dangerous to the client's position
  - Settlement valuation: range, leverage, walk-away threshold
  - Likelihood of success assessment

Phase 3A (Senior Partner Review) runs after D2 and challenges the entire
analysis, stored as senior_partner_notes in ci_war_room.

Output stored in ci_war_room.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

WAR_ROOM_PROMPT = """You are the lead opposing counsel preparing to defeat our client's position. This is a war room exercise — be realistic and thorough. Do not pull punches.

OUR CLIENT'S ROLE: {role}
OUR CASE GOAL: {goal_text}
JURISDICTION: {jurisdiction}

THE EVIDENCE (documents, entities, timeline, financials, contradictions, theories):
{case_summary}

YOUR TASK:
1. Build the opposing side's best possible case
2. Identify the 3 most dangerous arguments opposing counsel will make
3. Identify our top 5 client vulnerabilities
4. Find the documents most dangerous to our client's position
5. Provide a settlement valuation with range and walk-away threshold
6. Assess our likelihood of success

Respond in JSON:
{{
  "opposing_case_summary": "2-3 paragraph narrative of the opposing case theory, written as opposing lead counsel would present it to a jury",
  "top_3_dangerous_arguments": [
    {{
      "argument": "The dangerous argument, stated as opposing counsel would make it",
      "supporting_evidence": [
        {{"paperless_doc_id": 101, "excerpt": "...", "why_damaging": "..."}}
      ],
      "our_response": "How we rebut this argument",
      "response_strength": "strong|moderate|weak",
      "response_weaknesses": "What makes our rebuttal vulnerable"
    }}
  ],
  "client_vulnerabilities": [
    {{
      "vulnerability": "Description of vulnerability",
      "severity": "critical|high|medium",
      "evidence_supporting_vulnerability": [
        {{"paperless_doc_id": 101, "excerpt": "..."}}
      ],
      "mitigation": "How to minimize this vulnerability",
      "mitigation_feasibility": "easy|moderate|difficult"
    }}
  ],
  "smoking_guns_against_client": [
    {{
      "paperless_doc_id": 101,
      "doc_title": "Document title",
      "why_dangerous": "Specific explanation of why this document is bad for our client",
      "how_opposing_will_use": "How opposing counsel will likely use this in argument or examination",
      "our_best_response": "Our best response to this document"
    }}
  ],
  "settlement_analysis": {{
    "range_low_usd": 0,
    "range_high_usd": 0,
    "most_likely_usd": 0,
    "walk_away_usd": 0,
    "leverage_points": ["List of leverage points favoring our client"],
    "adverse_leverage_points": ["List of leverage points favoring opposing side"],
    "rationale": "2-3 sentence settlement valuation rationale citing specific evidence",
    "comparable_verdicts_notes": "Notes on comparable case outcomes if any data available"
  }},
  "likelihood_of_success_pct": 0,
  "likelihood_rationale": "2-3 sentence explanation of the success percentage",
  "immediate_action_items": [
    "Top 3-5 most urgent things our team should do right now"
  ],
  "war_room_memo": "Full narrative war room memo (5-8 paragraphs): opposing theory, our vulnerabilities, recommended strategy, and key next steps"
}}

RULES:
1. Be brutally honest — the purpose is to identify real risks before trial.
2. Every dangerous argument and vulnerability MUST cite specific documents.
3. Settlement range should be grounded in the documented evidence.
4. Likelihood of success should be realistic, not optimistic.
5. Immediate action items should be specific and actionable.
"""

SENIOR_PARTNER_REVIEW_PROMPT = """You are a managing partner reviewing a junior partner's case analysis. Your job is to find what they missed.

ORIGINAL ANALYSIS SUMMARY:
{analysis_summary}

CASE CONTEXT:
Role: {role}
Goal: {goal_text}

YOUR TASK: Be the senior voice in the room. Find the 5 most significant things the analysis missed, understated, or got wrong.

Respond in JSON:
{{
  "missed_issues": [
    {{
      "issue": "Description of what was missed or understated",
      "why_significant": "Why this matters to the case outcome",
      "recommended_action": "What should be done about it"
    }}
  ],
  "single_most_important_finding": "The one finding from the entire analysis that counsel should focus on above all else",
  "logical_leaps": [
    "Description of an unsupported logical leap or speculative conclusion in the analysis"
  ],
  "theories_that_wont_survive_cross": [
    {{
      "theory": "Theory that is vulnerable",
      "weakness": "Why this theory won't survive cross-examination"
    }}
  ],
  "senior_partner_notes": "2-3 paragraph senior partner memo: what I would do differently, what I'd add to the deposition prep, and what I'm most worried about at trial"
}}
"""


class WarRoom:
    """
    Tier 4+ war room simulation.
    Builds opposing counsel's case and identifies client vulnerabilities.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.war_room_task = get_task('war_room_strategy')
        self.review_task = get_task('senior_partner_review')

    def run_war_room(self, run_id: str, role: str, goal_text: str, jurisdiction: str,
                     entities: List[Dict], timeline: List[Dict], financial: List[Dict],
                     contradictions: List[Dict], theories: List[Dict],
                     documents: List[Dict]) -> Dict[str, Any]:
        """
        Run the full war room simulation.

        Returns war room dict (mirrors ci_war_room schema).
        """
        case_summary = self._build_case_summary(
            entities, timeline, financial, contradictions, theories, documents
        )

        prompt = WAR_ROOM_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Build the strongest case',
            jurisdiction=jurisdiction or 'Not specified',
            case_summary=case_summary[:10000],
        )

        result = self._call_llm_with_escalation(prompt, 'ci:war_room_strategy',
                                                 self.war_room_task)
        if not result:
            return self._empty_result()

        logger.info(
            f"WarRoom run {run_id}: "
            f"{len(result.get('top_3_dangerous_arguments', []))} dangerous args, "
            f"likelihood={result.get('likelihood_of_success_pct', 0)}%"
        )
        return result

    def run_senior_partner_review(self, run_id: str, role: str, goal_text: str,
                                   analysis_summary: str) -> Optional[Dict]:
        """
        Phase 3A: Senior partner review of the full D2 analysis.

        Returns senior partner review dict, or None on failure.
        """
        prompt = SENIOR_PARTNER_REVIEW_PROMPT.format(
            analysis_summary=analysis_summary[:8000],
            role=role,
            goal_text=goal_text or 'Build the strongest case',
        )

        result = self._call_llm_with_escalation(prompt, 'ci:senior_partner_review',
                                                 self.review_task)
        if result:
            logger.info(f"SeniorPartnerReview run {run_id}: "
                        f"{len(result.get('missed_issues', []))} missed issues")
        return result

    def _build_case_summary(self, entities, timeline, financial, contradictions,
                             theories, documents) -> str:
        """Build a compact case summary for the war room prompt."""
        parts = []

        # Entities
        entity_lines = [
            f"  {e.get('entity_type', '?')}: {e.get('name', '?')} ({e.get('role_in_case', '')})"
            for e in entities[:20] if not e.get('merged_into')
        ]
        if entity_lines:
            parts.append("ENTITIES:\n" + '\n'.join(entity_lines))

        # Timeline (significant events)
        timeline_lines = [
            f"  {ev.get('event_date', '?')}: {ev.get('description', '')} [{ev.get('significance', '?')}]"
            for ev in timeline[:30]
        ]
        if timeline_lines:
            parts.append("TIMELINE:\n" + '\n'.join(timeline_lines))

        # Contradictions
        contr_lines = [
            f"  [{c.get('severity', '?').upper()}] {c.get('description', '')}"
            for c in contradictions[:10]
        ]
        if contr_lines:
            parts.append("CONTRADICTIONS:\n" + '\n'.join(contr_lines))

        # Theories
        theory_lines = [
            f"  [{t.get('theory_type', '?')}] {t.get('theory_text', '')[:150]} "
            f"(confidence: {t.get('confidence', 0):.0%}, status: {t.get('status', '?')})"
            for t in theories[:8]
        ]
        if theory_lines:
            parts.append("THEORIES:\n" + '\n'.join(theory_lines))

        # Documents
        doc_lines = [
            f"  [Doc #{d.get('id', d.get('doc_id', '?'))}] {d.get('title', 'Untitled')}"
            for d in documents[:30]
        ]
        if doc_lines:
            parts.append("DOCUMENTS:\n" + '\n'.join(doc_lines))

        # Financial
        fin_lines = [
            f"  {f.get('date', '?')}: ${f.get('amount', f.get('amount_usd', 0)):,} — {f.get('description', '')}"
            for f in financial[:15] if isinstance(f, dict)
        ]
        if fin_lines:
            parts.append("FINANCIAL TRANSACTIONS:\n" + '\n'.join(fin_lines))

        return '\n\n'.join(parts)

    def _call_llm_with_escalation(self, prompt: str, operation: str,
                                   task_def=None) -> Optional[Dict]:
        td = task_def or self.war_room_task
        for model_key in ('primary', 'escalate'):
            model = td.primary_model if model_key == 'primary' else td.escalate_model
            provider = td.primary_provider if model_key == 'primary' else td.escalate_provider
            client = self.llm_clients.get(provider)
            if not client:
                continue
            try:
                result = self._call_llm(client, model, prompt, provider, operation, td)
                if result:
                    return result
            except Exception as e:
                logger.error(f"WarRoom {provider}/{model} failed: {e}")
        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, operation: str, task_def) -> Optional[Dict]:
        try:
            if provider == 'anthropic':
                response = client.client.messages.create(
                    model=model,
                    max_tokens=task_def.max_output_tokens,
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
                    max_tokens=task_def.max_output_tokens,
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
            logger.warning(f"WarRoom JSON error: {e}")
            return None
        except Exception as e:
            logger.error(f"WarRoom LLM error: {e}")
            return None

    @staticmethod
    def _empty_result() -> Dict:
        return {
            'opposing_case_summary': 'War room analysis could not be completed.',
            'top_3_dangerous_arguments': [],
            'client_vulnerabilities': [],
            'smoking_guns_against_client': [],
            'settlement_analysis': {
                'range_low_usd': 0,
                'range_high_usd': 0,
                'most_likely_usd': 0,
                'walk_away_usd': 0,
                'leverage_points': [],
                'adverse_leverage_points': [],
                'rationale': '',
            },
            'likelihood_of_success_pct': 50,
            'likelihood_rationale': '',
            'immediate_action_items': [],
            'war_room_memo': '',
        }
