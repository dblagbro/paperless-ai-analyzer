"""
Multi-Model Synthesis — Tier 5 (White Glove) CI Pipeline Phase 3B.

Runs theory generation and key-findings synthesis with BOTH:
  - Anthropic claude-opus-4-6
  - OpenAI gpt-4o

Simultaneously using ThreadPoolExecutor, then:
  - Identifies findings both models agree on (higher confidence)
  - Surfaces findings only one model identified (may be novel or hallucinated)
  - Flags direct disagreements (one model says X, other says not-X)
  - Produces a merged theory set with model_source attribution

Output stored in ci_multi_model_comparison.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

from analyzer.case_intelligence.task_registry import get_task

logger = logging.getLogger(__name__)

MULTI_MODEL_THEORY_PROMPT = """You are a senior legal analyst performing case theory generation for litigation.

CASE ROLE: {role}
CASE GOAL: {goal_text}

CASE SUMMARY (extracted from documents):
{case_summary}

EXISTING THEORIES (from earlier analysis — do not simply repeat, critically evaluate and expand):
{existing_theories}

CONTRADICTIONS IDENTIFIED:
{contradictions}

KEY FINANCIAL DATA:
{financial_summary}

Your task: Generate the most important legal and factual theories for this case, including theories the prior analysis may have missed or understated.

For each theory:
- State the theory clearly
- Identify the legal theory type (fraud, breach of contract, conspiracy, tortious interference, etc.)
- Rate confidence 0.0–1.0 based on documentary support
- Identify what SPECIFIC document evidence supports it
- Identify what evidence would DISPROVE it (adversarial)
- Note if this is a theory the prior analysis missed or understated

Respond in JSON:
{{
  "theories": [
    {{
      "theory_text": "Clear statement of the theory",
      "theory_type": "factual|legal|financial|behavioral",
      "legal_category": "fraud|breach_contract|conspiracy|negligence|etc",
      "confidence": 0.85,
      "key_evidence": ["Evidence item 1", "Evidence item 2"],
      "doc_support": [101, 102],
      "disproving_evidence": ["What would undermine this theory"],
      "missed_by_prior_analysis": true,
      "significance": "critical|high|medium|low"
    }}
  ],
  "key_findings": [
    {{
      "finding": "Important factual or legal finding",
      "significance": "critical|high|medium|low",
      "doc_support": [101],
      "missed_by_prior_analysis": true
    }}
  ],
  "case_assessment": "2-3 sentence overall assessment of the case from this model's perspective",
  "prior_analysis_gaps": ["List of significant gaps in the prior analysis"]
}}
"""

SYNTHESIS_PROMPT = """You are a managing partner reviewing competing legal analyses of the same case.

CASE ROLE: {role}
CASE GOAL: {goal_text}

TWO INDEPENDENT MODELS analyzed this case and produced the following:

── ANALYSIS A (Anthropic claude-opus) ──
{analysis_a}

── ANALYSIS B (OpenAI gpt-4o) ──
{analysis_b}

Your task: Synthesize these two analyses into a unified, authoritative assessment.

1. AGREED FINDINGS: Where both models identified the same theory/finding (list only once, note "both models agree")
2. MODEL A ONLY: Findings unique to Analysis A (may be novel insight or hallucination — flag accordingly)
3. MODEL B ONLY: Findings unique to Analysis B (same)
4. DISAGREEMENTS: Where models directly contradict each other — these are high-uncertainty areas, flag as "disputed"
5. MERGED ASSESSMENT: Unified view incorporating the strongest insights from both

Respond in JSON:
{{
  "agreed_theories": [
    {{
      "theory_text": "...",
      "theory_type": "...",
      "confidence_a": 0.8,
      "confidence_b": 0.75,
      "merged_confidence": 0.82,
      "significance": "critical|high|medium|low"
    }}
  ],
  "model_a_only": [
    {{
      "theory_text": "...",
      "confidence": 0.7,
      "assessment": "Novel insight worth pursuing|Possible hallucination — verify",
      "significance": "..."
    }}
  ],
  "model_b_only": [
    {{
      "theory_text": "...",
      "confidence": 0.65,
      "assessment": "...",
      "significance": "..."
    }}
  ],
  "disagreements": [
    {{
      "topic": "Topic of disagreement",
      "model_a_position": "Model A says...",
      "model_b_position": "Model B says...",
      "recommendation": "How attorney should resolve this uncertainty",
      "uncertainty_level": "high|medium|low"
    }}
  ],
  "merged_summary": "Authoritative 3-4 paragraph synthesis incorporating the strongest insights from both models",
  "confidence_in_analysis": 0.0,
  "models_agreement_rate": 0.0
}}
"""


class MultiModelSynthesis:
    """
    Tier 5 multi-model synthesis.
    Runs theory generation with both Anthropic and OpenAI in parallel, then merges.
    """

    def __init__(self, llm_clients: dict, usage_tracker=None):
        self.llm_clients = llm_clients
        self.usage_tracker = usage_tracker
        self.task_def = get_task('multi_model_synthesis')

    def synthesize(self, run_id: str, role: str, goal_text: str,
                   case_summary: str, existing_theories: List[Dict],
                   contradictions: List[Dict], financial_summary: str) -> Dict[str, Any]:
        """
        Run multi-model synthesis.

        Calls both Anthropic and OpenAI in parallel with the same theory-generation
        prompt, then uses a synthesis pass to merge, compare, and flag disagreements.

        Returns:
            Multi-model comparison dict (mirrors ci_multi_model_comparison schema)
        """
        theories_str = self._format_theories(existing_theories)
        contradictions_str = self._format_contradictions(contradictions)

        prompt = MULTI_MODEL_THEORY_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Build the strongest case',
            case_summary=case_summary[:8000],
            existing_theories=theories_str[:3000],
            contradictions=contradictions_str[:2000],
            financial_summary=financial_summary[:2000],
        )

        # ── Run both models in parallel ──────────────────────────────────────
        analysis_a: Optional[Dict] = None
        analysis_b: Optional[Dict] = None

        def call_anthropic():
            client = self.llm_clients.get('anthropic')
            if not client:
                return None
            return self._call_llm(client, 'claude-opus-4-6', prompt,
                                  'anthropic', 'ci:multi_model_anthropic',
                                  max_tokens=6000)

        def call_openai():
            client = self.llm_clients.get('openai')
            if not client:
                return None
            return self._call_llm(client, 'gpt-4o', prompt,
                                  'openai', 'ci:multi_model_openai',
                                  max_tokens=6000)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ci-multimodel') as ex:
            fut_a = ex.submit(call_anthropic)
            fut_b = ex.submit(call_openai)
            try:
                analysis_a = fut_a.result(timeout=120)
            except Exception as e:
                logger.warning(f"MultiModelSynthesis Anthropic failed: {e}")
            try:
                analysis_b = fut_b.result(timeout=120)
            except Exception as e:
                logger.warning(f"MultiModelSynthesis OpenAI failed: {e}")

        if not analysis_a and not analysis_b:
            logger.error(f"MultiModelSynthesis run {run_id}: both models failed")
            return self._empty_result()

        # If only one succeeded, return a partial result
        if not analysis_a:
            logger.warning(f"MultiModelSynthesis run {run_id}: Anthropic failed, using OpenAI only")
            return self._single_model_result(analysis_b, 'openai')
        if not analysis_b:
            logger.warning(f"MultiModelSynthesis run {run_id}: OpenAI failed, using Anthropic only")
            return self._single_model_result(analysis_a, 'anthropic')

        # ── Synthesis pass ───────────────────────────────────────────────────
        a_str = json.dumps(analysis_a, indent=2)[:5000]
        b_str = json.dumps(analysis_b, indent=2)[:5000]

        synth_prompt = SYNTHESIS_PROMPT.format(
            role=role,
            goal_text=goal_text or 'Build the strongest case',
            analysis_a=a_str,
            analysis_b=b_str,
        )

        # Use Anthropic for the synthesis pass (escalate to OpenAI if unavailable)
        synthesis_client = self.llm_clients.get('anthropic') or self.llm_clients.get('openai')
        synthesis_provider = 'anthropic' if self.llm_clients.get('anthropic') else 'openai'
        synthesis_model = 'claude-opus-4-6' if synthesis_provider == 'anthropic' else 'gpt-4o'

        synthesis_result = self._call_llm(
            synthesis_client, synthesis_model, synth_prompt,
            synthesis_provider, 'ci:multi_model_synthesis',
            max_tokens=self.task_def.max_output_tokens,
        )

        if not synthesis_result:
            # Fall back to simple merge if synthesis LLM call fails
            synthesis_result = self._simple_merge(analysis_a, analysis_b)

        result = {
            'anthropic_analysis': analysis_a,
            'openai_analysis': analysis_b,
            **synthesis_result,
        }

        logger.info(
            f"MultiModelSynthesis run {run_id}: "
            f"{len(result.get('agreed_theories', []))} agreed, "
            f"{len(result.get('disagreements', []))} disagreements, "
            f"agreement_rate={result.get('models_agreement_rate', 0):.0%}"
        )
        return result

    def _single_model_result(self, analysis: Dict, model: str) -> Dict:
        """Wrap a single-model result when the other model failed."""
        theories = analysis.get('theories', [])
        return {
            'anthropic_analysis': analysis if model == 'anthropic' else None,
            'openai_analysis': analysis if model == 'openai' else None,
            'agreed_theories': [],
            'model_a_only': theories if model == 'anthropic' else [],
            'model_b_only': theories if model == 'openai' else [],
            'disagreements': [],
            'merged_summary': (
                f"Note: Only {model} analysis is available for this run. "
                + analysis.get('case_assessment', '')
            ),
            'confidence_in_analysis': 0.5,
            'models_agreement_rate': 0.0,
        }

    def _simple_merge(self, analysis_a: Dict, analysis_b: Dict) -> Dict:
        """Fallback: simple merge without a synthesis LLM call."""
        a_theories = {t.get('theory_text', ''): t for t in analysis_a.get('theories', [])}
        b_theories = {t.get('theory_text', ''): t for t in analysis_b.get('theories', [])}

        agreed, a_only, b_only = [], [], []
        for text, ta in a_theories.items():
            if any(self._theories_similar(text, bt) for bt in b_theories):
                agreed.append({
                    'theory_text': text,
                    'theory_type': ta.get('theory_type', ''),
                    'confidence_a': ta.get('confidence', 0.5),
                    'confidence_b': 0.5,
                    'merged_confidence': ta.get('confidence', 0.5),
                    'significance': ta.get('significance', 'medium'),
                })
            else:
                a_only.append({'theory_text': text, 'confidence': ta.get('confidence', 0.5),
                                'assessment': 'Verify', 'significance': ta.get('significance', 'medium')})

        for text, tb in b_theories.items():
            if not any(self._theories_similar(text, at) for at in a_theories):
                b_only.append({'theory_text': text, 'confidence': tb.get('confidence', 0.5),
                               'assessment': 'Verify', 'significance': tb.get('significance', 'medium')})

        total = len(a_theories) + len(b_theories)
        agreement_rate = (2 * len(agreed)) / total if total > 0 else 0.0

        return {
            'agreed_theories': agreed,
            'model_a_only': a_only,
            'model_b_only': b_only,
            'disagreements': [],
            'merged_summary': (
                f"Multi-model synthesis: {len(agreed)} agreed theories, "
                f"{len(a_only)} Anthropic-only, {len(b_only)} OpenAI-only. "
                f"Agreement rate: {agreement_rate:.0%}."
            ),
            'confidence_in_analysis': min(0.9, 0.5 + agreement_rate * 0.4),
            'models_agreement_rate': round(agreement_rate, 3),
        }

    @staticmethod
    def _theories_similar(text_a: str, text_b: str) -> bool:
        """Rough similarity: >30% word overlap."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        return overlap / union > 0.30

    # ── Formatting helpers ────────────────────────────────────────────────────

    def _format_theories(self, theories: List[Dict]) -> str:
        if not theories:
            return 'No prior theories.'
        return '\n'.join(
            f"- [{t.get('theory_type', '?')}] {t.get('theory_text', '')[:150]} "
            f"(conf: {t.get('confidence', 0):.0%}, status: {t.get('status', '?')})"
            for t in theories[:12]
        )

    def _format_contradictions(self, contradictions: List[Dict]) -> str:
        if not contradictions:
            return 'No contradictions identified.'
        return '\n'.join(
            f"- [{c.get('severity', '?').upper()}] {c.get('description', '')[:150]}"
            for c in contradictions[:12]
        )

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, client, model: str, prompt: str,
                  provider: str, operation: str, max_tokens: int = 6000) -> Optional[Dict]:
        """Route through proxy pool with provider PINNED (for parallel multi-model
        synthesis each thread must use a single specified provider). Uses LMRH
        fallback-chain=<provider> and direct-provider fallback to same provider."""
        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        import json as _json
        try:
            result = call_llm(
                messages=[{'role': 'user', 'content': prompt}],
                task='analysis',
                fallback_chain=provider,  # pin proxy to this provider only
                model_pref=model,
                max_tokens=max_tokens,
                operation=operation,
                direct_provider=provider,
                direct_api_key=getattr(client, 'api_key', None),
                direct_model=model,
                usage_tracker=self.usage_tracker,
                response_format={'type': 'json_object'} if provider == 'openai' else None,
            )
            text = result.get('content') or ''
            if '```json' in text:
                text = text.split('```json', 1)[1].split('```', 1)[0].strip()
            elif '```' in text:
                text = text.split('```', 1)[1].split('```', 1)[0].strip()
            return _json.loads(text)
        except LLMUnavailableError as e:
            logger.error(f"MultiModelSynthesis LLM unavailable ({provider}): {e}")
            return None
        except _json.JSONDecodeError as e:
            logger.warning(f"MultiModelSynthesis JSON error ({provider}): {e}")
            return None
        except Exception as e:
            logger.error(f"MultiModelSynthesis LLM error ({provider}): {e}")
            return None

    @staticmethod
    def _empty_result() -> Dict:
        return {
            'anthropic_analysis': None,
            'openai_analysis': None,
            'agreed_theories': [],
            'model_a_only': [],
            'model_b_only': [],
            'disagreements': [],
            'merged_summary': 'Multi-model synthesis could not be completed.',
            'confidence_in_analysis': 0.0,
            'models_agreement_rate': 0.0,
        }
