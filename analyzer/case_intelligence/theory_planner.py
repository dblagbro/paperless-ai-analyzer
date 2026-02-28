"""
Theory Planner + Adversarial Tester — Tier 3 CI Pipeline Stage.

Generates role-aware factual and legal theories, then adversarially
tests each theory to determine its strength before adding to the ledger.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

THEORY_GENERATION_PROMPT = """You are a senior litigator preparing for trial.

Based on all the evidence below, generate LEGAL AND FACTUAL THEORIES that support the {role} in this case.

CASE GOAL: {goal_text}
JURISDICTION: {jurisdiction}

ENTITIES IDENTIFIED:
{entities_summary}

TIMELINE (KEY EVENTS):
{timeline_summary}

FINANCIAL FACTS:
{financial_summary}

CONTRADICTIONS FOUND:
{contradictions_summary}

RELEVANT AUTHORITIES:
{authorities_summary}

Generate theories in JSON:
{{
  "theories": [
    {{
      "theory_text": "Complete theory statement (2-4 sentences). Must be specific and evidence-grounded.",
      "theory_type": "factual|legal|financial|behavioral",
      "role_perspective": "{role}",
      "confidence": 0.75,
      "supporting_evidence": [
        {{
          "paperless_doc_id": 101,
          "excerpt": "supporting quote",
          "page_number": 1,
          "how_it_supports": "explanation"
        }}
      ],
      "legal_basis": "Statute or rule that makes this legally actionable (or null)",
      "what_would_change": "Evidence that would undermine this theory",
      "discovery_implications": "What additional discovery this theory requires"
    }}
  ]
}}

RULES:
1. Every theory MUST cite specific documents as supporting evidence.
2. Each supporting_evidence item MUST include the numeric paperless_doc_id shown as [Doc #NNN] in the evidence above. NEVER set paperless_doc_id to null.
3. Be honest about confidence — use 0.4 if evidence is thin, 0.9 if overwhelming.
4. Generate theories that the {role} can actually USE in proceedings.
5. Include both strong and developing theories (the latter with lower confidence).
6. Maximum 8 theories per run. Keep each theory_text concise (1-2 sentences).
"""

ADVERSARIAL_TESTING_PROMPT = """You are opposing counsel tasked with defeating the following theory.

THEORY: {theory_text}
CONFIDENCE CLAIMED: {confidence}
SUPPORTING EVIDENCE: {supporting_evidence}

AVAILABLE COUNTER-DOCUMENTS:
{counter_docs}

Attempt to FALSIFY or WEAKEN this theory. Respond in JSON:
{{
  "falsification_successful": false,
  "revised_confidence": 0.65,
  "attack_type": "missing_element|counter_evidence|alternative_explanation|hearsay|chain_of_custody|statute_of_limitations|other",
  "falsification_report": "Detailed analysis of weaknesses in the theory (2-4 sentences)",
  "counter_evidence": [
    {{
      "paperless_doc_id": 102,
      "excerpt": "quote that contradicts or weakens the theory",
      "page_number": 2,
      "how_it_undermines": "explanation"
    }}
  ],
  "what_would_change": "Updated description of what evidence would rehabilitate this theory",
  "recommended_status": "supported|refuted|uncertain|needs_more_evidence"
}}

Be rigorous. If the theory is genuinely strong, say so (falsification_successful: false, revised_confidence >= original).
If there are real weaknesses, identify them specifically with document citations.
IMPORTANT: Each counter_evidence item MUST include the numeric paperless_doc_id from the counter-documents listed above. NEVER set paperless_doc_id to null.
"""


class TheoryPlanner:
    """
    Tier 3 theory generator. Uses gpt-4o with Claude escalation.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('theory_generation')

    def generate_theories(self, role: str, goal_text: str, jurisdiction: str,
                           entities_summary: str, timeline_summary: str,
                           financial_summary: str, contradictions_summary: str,
                           authorities_summary: str,
                           run_id: str = None) -> List[Dict[str, Any]]:
        """
        Generate role-aware theories from all evidence gathered.

        Returns list of theory dicts.
        """
        prompt = THEORY_GENERATION_PROMPT.format(
            role=role,
            goal_text=goal_text or f'Build strongest case for {role}',
            jurisdiction=jurisdiction,
            entities_summary=entities_summary[:3500],
            timeline_summary=timeline_summary[:3500],
            financial_summary=financial_summary[:2500],
            contradictions_summary=contradictions_summary[:2500],
            authorities_summary=authorities_summary[:2000],
        )

        result = self._call_llm_with_escalation(
            prompt, self.task_def, operation='ci:theory_generation'
        )
        if not result:
            return []

        theories = result.get('theories', [])
        extraction_ts = datetime.now(timezone.utc).isoformat()
        for theory in theories:
            # Ensure supporting_evidence has extraction_version
            for ev in theory.get('supporting_evidence', []):
                ev['extraction_version'] = extraction_ts
                ev['prompt_version'] = 'theory_v1'

        return theories

    def _call_llm_with_escalation(self, prompt: str, task_def,
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
                    logger.debug(f"TheoryPlanner: escalating ({operation})")
            except Exception as e:
                logger.error(f"TheoryPlanner: {provider}/{model} failed: {e}")

        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, operation: str) -> Optional[dict]:
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
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
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
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                    )
            else:
                return None

            # Strip markdown fences before parsing (Anthropic wraps JSON in ```json...```)
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            parsed = json.loads(text)
            logger.info(f"TheoryPlanner: {provider}/{model} returned "
                        f"{len(parsed.get('theories', []))} theories")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"TheoryPlanner: JSON parse error ({provider}/{model}): {e} | "
                           f"text[:200]={text[:200]!r}")
            return None
        except Exception as e:
            logger.error(f"TheoryPlanner LLM call failed: {e}")
            return None


class AdversarialTester:
    """
    Tier 3 adversarial tester. Attempts to falsify each theory.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('adversarial_testing')

    def test_theory(self, theory_text: str, confidence: float,
                    supporting_evidence: List[dict],
                    counter_docs: List[Dict[str, Any]],
                    theory_id: int = None) -> Dict[str, Any]:
        """
        Test a theory by attempting to falsify it.

        Returns an adversarial test result dict.
        """
        counter_docs_text = '\n\n'.join([
            f"Doc #{d.get('doc_id')} ({d.get('title', 'Untitled')}):\n{d.get('content', '')[:800]}"
            for d in counter_docs[:5]
        ])

        prompt = ADVERSARIAL_TESTING_PROMPT.format(
            theory_text=theory_text,
            confidence=confidence,
            supporting_evidence=json.dumps(supporting_evidence[:5], indent=2),
            counter_docs=counter_docs_text or 'No counter-documents available.',
        )

        result = self._call_llm_with_escalation(
            prompt, self.task_def, operation='ci:adversarial_testing'
        )

        if not result:
            return {
                'falsification_successful': False,
                'revised_confidence': confidence,
                'falsification_report': 'Adversarial testing could not be completed.',
                'recommended_status': 'uncertain',
                'counter_evidence': [],
                'what_would_change': '',
            }

        extraction_ts = datetime.now(timezone.utc).isoformat()
        for ev in result.get('counter_evidence', []):
            ev['extraction_version'] = extraction_ts
            ev['prompt_version'] = 'adversarial_v1'

        return result

    def _call_llm_with_escalation(self, prompt: str, task_def,
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
            except Exception as e:
                logger.error(f"AdversarialTester: {provider}/{model} failed: {e}")

        return None

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, operation: str) -> Optional[dict]:
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
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
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
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                    )
            else:
                return None

            # Strip markdown fences before parsing (Anthropic wraps JSON in ```json...```)
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.error(f"AdversarialTester LLM call failed: {e}")
            return None
