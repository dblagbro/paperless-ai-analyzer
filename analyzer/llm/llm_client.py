"""
LLM Client

Optional AI-assisted anomaly analysis using Claude or OpenAI.
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for AI-assisted anomaly analysis."""

    def __init__(self,
                 provider: str = 'anthropic',
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            provider: 'anthropic' or 'openai'
            api_key: API key for the provider
            model: Model name (optional, uses defaults)
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._get_default_model()
        self.client = None

        if api_key:
            self._initialize_client()
        else:
            logger.warning("No LLM API key provided, AI analysis disabled")

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            'anthropic': 'claude-3-5-sonnet-20241022',
            'openai': 'gpt-4o'
        }
        return defaults.get(self.provider, 'claude-3-5-sonnet-20241022')

    def _initialize_client(self) -> None:
        """Initialize the LLM client."""
        try:
            if self.provider == 'anthropic':
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif self.provider == 'openai':
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                logger.error(f"Unknown LLM provider: {self.provider}")
        except ImportError as e:
            logger.error(f"Failed to import {self.provider} library: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")

    def analyze_anomalies(self,
                         document_info: Dict[str, Any],
                         deterministic_results: Dict[str, Any],
                         extracted_data: Dict[str, Any],
                         forensics_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI-assisted analysis of detected anomalies.

        Args:
            document_info: Basic document metadata
            deterministic_results: Results from deterministic checks
            extracted_data: Extracted transaction data
            forensics_results: Forensics analysis results

        Returns:
            Dict with AI analysis, narrative, and suggested tags
        """
        if not self.client:
            return {
                'enabled': False,
                'narrative': '',
                'suggested_tags': [],
                'error': 'llm_not_initialized'
            }

        try:
            # Build prompt that includes only facts, not raw data
            prompt = self._build_analysis_prompt(
                document_info,
                deterministic_results,
                extracted_data,
                forensics_results
            )

            # Call LLM
            response = self._call_llm(prompt)

            # Parse response
            parsed = self._parse_response(response)

            return {
                'enabled': True,
                'narrative': parsed.get('narrative', ''),
                'suggested_tags': parsed.get('suggested_tags', []),
                'recommended_actions': parsed.get('recommended_actions', []),
                'confidence': parsed.get('confidence', 'medium')
            }

        except Exception as e:
            logger.error(f"AI analysis failed: {e}", exc_info=True)
            return {
                'enabled': True,
                'narrative': '',
                'suggested_tags': [],
                'error': str(e)
            }

    def _build_analysis_prompt(self,
                              doc_info: Dict,
                              determ_results: Dict,
                              extracted: Dict,
                              forensics: Dict) -> str:
        """Build prompt for LLM analysis."""

        # Summarize findings (no raw transactions, only counts and evidence)
        anomalies_found = determ_results.get('anomalies_found', [])
        evidence = determ_results.get('evidence', {})
        risk_score = forensics.get('risk_score_percent', 0)
        forensic_signals = forensics.get('signals', [])

        prompt = f"""You are analyzing a financial document for anomalies. Based on the following FACTUAL findings from automated checks, provide a brief narrative summary and suggest additional anomaly tags if appropriate.

Document: {doc_info.get('title', 'Unknown')}
Document Type: Financial statement with transactions
Total Transactions: {extracted.get('total_transactions', 0)}

DETERMINISTIC FINDINGS:
"""

        # Add specific anomaly evidence
        for anomaly_type in anomalies_found:
            if anomaly_type == 'balance_mismatch':
                mismatches = evidence.get('balance_mismatch', {}).get('mismatches', [])
                prompt += f"\n- Balance Mismatches: {len(mismatches)} found\n"
                if mismatches:
                    for m in mismatches[:3]:  # Show first 3
                        prompt += f"  * Row {m['row_index']}: Expected {m['expected_balance']}, Found {m['actual_balance']}, Diff: ${m['difference']:.2f}\n"

            elif anomaly_type == 'duplicate_transactions':
                dups = evidence.get('duplicate_transactions', {}).get('duplicates', [])
                prompt += f"\n- Duplicate Transactions: {len(dups)} found\n"

            elif anomaly_type == 'date_order_violation':
                violations = evidence.get('date_order_violation', {}).get('violations', [])
                prompt += f"\n- Date Order Violations: {len(violations)} found\n"

        prompt += f"\nFORENSIC ANALYSIS:\n"
        prompt += f"- Risk Score: {risk_score}% (0-100 scale)\n"
        if forensic_signals:
            prompt += f"- Signals Detected: {len(forensic_signals)}\n"
            for sig in forensic_signals[:3]:
                prompt += f"  * {sig['type']}: {sig['description']} (weight: {sig['weight']:.1f})\n"

        prompt += """
IMPORTANT CONSTRAINTS:
1. Do NOT invent or guess transaction amounts or dates
2. Base your analysis ONLY on the provided findings above
3. Suggest additional anomaly tags ONLY if the evidence strongly supports them

Please provide your analysis in this JSON format:
{
  "narrative": "Brief 2-3 sentence summary of findings and their implications",
  "suggested_tags": ["aianomaly:tag1", "aianomaly:tag2"],
  "recommended_actions": ["Action 1", "Action 2"],
  "confidence": "high|medium|low"
}
"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with multi-provider fallback."""
        # Load AI configuration for document analysis
        from pathlib import Path
        import json

        config_path = Path('/app/data/ai_config.json')
        ai_config = None

        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    ai_config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load AI config, using defaults: {e}")

        # If config exists, use it for multi-provider fallback
        if ai_config and 'document_analysis' in ai_config:
            providers = ai_config['document_analysis'].get('providers', [])

            last_error = None
            attempted = []

            for provider_config in providers:
                if not provider_config.get('enabled', False):
                    continue

                provider_name = provider_config.get('name')
                api_key = provider_config.get('api_key', '').strip()
                models = provider_config.get('models', [])

                if not api_key:
                    continue

                # Try this provider
                try:
                    if provider_name == 'openai':
                        import openai
                        client = openai.OpenAI(api_key=api_key)

                        for model in models:
                            try:
                                logger.info(f"Trying document analysis: OpenAI {model}")
                                attempted.append(f"OpenAI {model}")

                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=1024
                                )
                                result = response.choices[0].message.content
                                logger.info(f"✓ Document analysis using: OpenAI {model}")
                                return result
                            except Exception as e:
                                if '404' in str(e) or 'model_not_found' in str(e):
                                    logger.warning(f"Model {model} not available, trying next...")
                                    last_error = e
                                    continue
                                logger.warning(f"Error with OpenAI {model}: {e}")
                                last_error = e
                                continue

                    elif provider_name == 'anthropic':
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)

                        for model in models:
                            try:
                                logger.info(f"Trying document analysis: Anthropic {model}")
                                attempted.append(f"Anthropic {model}")

                                response = client.messages.create(
                                    model=model,
                                    max_tokens=1024,
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                result = response.content[0].text
                                logger.info(f"✓ Document analysis using: Anthropic {model}")
                                return result
                            except Exception as e:
                                if '404' in str(e) or 'not_found' in str(e):
                                    logger.warning(f"Model {model} not available, trying next...")
                                    last_error = e
                                    continue
                                logger.warning(f"Error with Anthropic {model}: {e}")
                                last_error = e
                                continue

                except Exception as e:
                    logger.error(f"Failed to initialize {provider_name}: {e}")
                    last_error = e
                    continue

            # All configured providers failed
            attempted_str = ", ".join(attempted) if attempted else "no models"
            raise Exception(f"No available models. Tried: {attempted_str}. Last error: {last_error}")

        # Fallback to original single-provider logic if no config
        if self.provider == 'anthropic':
            # Try the configured model first, then fallback models
            models_to_try = [self.model]
            if self.model not in ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']:
                models_to_try.extend(['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'])

            last_error = None
            for model in models_to_try:
                try:
                    response = self.client.messages.create(
                        model=model,
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    # Update self.model to the working one for future calls
                    if model != self.model:
                        logger.info(f"Using fallback model: {model}")
                        self.model = model
                    return response.content[0].text
                except Exception as e:
                    last_error = e
                    if '404' not in str(e) and 'not_found' not in str(e):
                        raise
                    logger.warning(f"Model {model} not available, trying next...")
                    continue

            # If all models failed, raise the last error
            raise last_error or Exception("No models available")

        elif self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=1024
            )
            return response.choices[0].message.content

        return ""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response."""
        try:
            # Try to extract JSON from response
            # Look for JSON block
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                # Validate suggested tags
                tags = parsed.get('suggested_tags', [])
                valid_tags = [t for t in tags if isinstance(t, str) and t.startswith('aianomaly:')]
                parsed['suggested_tags'] = valid_tags

                return parsed
            else:
                # Fallback: treat entire response as narrative
                return {
                    'narrative': response,
                    'suggested_tags': [],
                    'recommended_actions': [],
                    'confidence': 'medium'
                }

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {
                'narrative': response,
                'suggested_tags': [],
                'recommended_actions': [],
                'confidence': 'low'
            }
