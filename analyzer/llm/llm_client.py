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
                 model: Optional[str] = None,
                 usage_tracker=None):
        """
        Initialize LLM client.

        Args:
            provider: 'anthropic' or 'openai'
            api_key: API key for the provider
            model: Model name (optional, uses defaults)
            usage_tracker: LLMUsageTracker instance for token tracking
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._get_default_model()
        self.client = None
        self.usage_tracker = usage_tracker

        if api_key:
            self._initialize_client()
        else:
            logger.warning("No LLM API key provided, AI analysis disabled")

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            'anthropic': 'claude-sonnet-4-5-20250929',
            'openai': 'gpt-4o'
        }
        return defaults.get(self.provider, 'claude-sonnet-4-5-20250929')

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
            response = self._call_llm(
                prompt,
                operation='anomaly_detection',
                document_id=document_info.get('id')
            )

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

    def generate_document_summary(self, document_info: Dict[str, Any], content_preview: str = "") -> Dict[str, str]:
        """
        Generate brief and full summaries of a document.

        Args:
            document_info: Document metadata (title, type, etc.)
            content_preview: First ~500 chars of document content for context

        Returns:
            Dict with 'brief' (1 sentence) and 'full' (3-4 sentences) summaries
        """
        if not self.client:
            # Fallback to title-based summary
            title = document_info.get('title', 'Unknown Document')
            return {
                'brief': f"Financial document: {title}",
                'full': f"This is a financial document titled '{title}'. No AI summary available - LLM client not initialized."
            }

        try:
            doc_title = document_info.get('title', 'Unknown')
            doc_type = document_info.get('document_type', 'financial document')

            # Build prompt for summarization
            prompt = f"""Analyze this document and provide two summaries:

Document Title: {doc_title}
Document Type: {doc_type}
{f"Content Preview: {content_preview[:500]}..." if content_preview else ""}

Provide:
1. BRIEF (1 sentence): A concise description of what this document is
2. FULL (3-4 sentences): A detailed summary including purpose, key details, and context

Respond in JSON format:
{{
  "brief": "One sentence description",
  "full": "Three to four sentence detailed summary"
}}

Important: Base your summary on the title and any content provided. If content is limited, describe what the document type and title suggest about its purpose."""

            # Call LLM
            response = self._call_llm(
                prompt,
                operation='summary_generation',
                document_id=document_info.get('id')
            )

            # Parse JSON response
            import json
            try:
                result = json.loads(response)
                return {
                    'brief': result.get('brief', f"{doc_type}: {doc_title}"),
                    'full': result.get('full', f"Financial document titled '{doc_title}'.")
                }
            except json.JSONDecodeError:
                # If not valid JSON, try to extract summaries from text
                lines = response.strip().split('\n')
                brief = next((l.split(':', 1)[1].strip() for l in lines if 'brief' in l.lower()), f"{doc_type}: {doc_title}")
                full = next((l.split(':', 1)[1].strip() for l in lines if 'full' in l.lower()), brief)
                return {'brief': brief, 'full': full}

        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            title = document_info.get('title', 'Unknown Document')
            return {
                'brief': f"Financial document: {title}",
                'full': f"Document titled '{title}'. Summary generation failed."
            }

    def generate_comparative_summary(self, document_info: Dict[str, Any],
                                    content_preview: str = "",
                                    similar_documents: list = None) -> Dict[str, str]:
        """
        Generate enhanced summaries that compare the document with others in the system.

        Args:
            document_info: Document metadata (title, type, date, etc.)
            content_preview: First ~500 chars of document content for context
            similar_documents: List of similar documents from vector search

        Returns:
            Dict with 'brief' (1 sentence) and 'full' (detailed comparison) summaries
        """
        if not self.client or not similar_documents:
            # Fall back to basic summary if no LLM or no similar docs
            return self.generate_document_summary(document_info, content_preview)

        try:
            doc_title = document_info.get('title', 'Unknown')
            doc_type = document_info.get('document_type', 'financial document')
            doc_date = document_info.get('created', 'unknown date')
            doc_id = document_info.get('id', 'unknown')

            # Build context about similar documents
            similar_docs_context = "\n".join([
                f"- Doc #{doc.get('id', 'N/A')}: {doc.get('title', 'Untitled')} "
                f"(Date: {doc.get('created', 'unknown')}, Similarity: {doc.get('similarity', 0):.2f})"
                for doc in similar_documents[:5]  # Top 5 most similar
            ])

            # Enhanced prompt with comparison instructions
            prompt = f"""Analyze this document and compare it with similar documents in the system.

Current Document:
- ID: {doc_id}
- Title: {doc_title}
- Type: {doc_type}
- Date: {doc_date}
{f"- Content Preview: {content_preview[:500]}..." if content_preview else ""}

Similar Documents in System:
{similar_docs_context if similar_docs_context else "No similar documents found"}

Provide:
1. BRIEF (1 sentence): What this document is
2. FULL (4-6 sentences): Detailed analysis including:
   - What this document contains
   - How it relates to similar documents (if any)
   - Whether this supersedes or is superseded by another document (check dates)
   - Any notable patterns or relationships with other documents

Respond in JSON format:
{{
  "brief": "One sentence description",
  "full": "Four to six sentence detailed summary with comparisons"
}}

Important:
- Compare dates to determine document supersession
- Note if this is an updated version of an earlier document
- Identify relationships between documents"""

            # Call LLM
            response = self._call_llm(
                prompt,
                operation='comparative_summary',
                document_id=doc_id
            )

            # Parse JSON response
            import json
            try:
                result = json.loads(response)
                return {
                    'brief': result.get('brief', f"{doc_type}: {doc_title}"),
                    'full': result.get('full', f"Financial document titled '{doc_title}'.")
                }
            except json.JSONDecodeError:
                # Fallback parsing
                lines = response.strip().split('\n')
                brief = next((l.split(':', 1)[1].strip() for l in lines if 'brief' in l.lower()), f"{doc_type}: {doc_title}")
                full = next((l.split(':', 1)[1].strip() for l in lines if 'full' in l.lower()), brief)
                return {'brief': brief, 'full': full}

        except Exception as e:
            logger.warning(f"Failed to generate comparative summary: {e}")
            # Fall back to basic summary
            return self.generate_document_summary(document_info, content_preview)

    def extract_rich_metadata(self, document_info: Dict[str, Any], content_preview: str = "") -> Dict[str, Any]:
        """
        Extract comprehensive metadata from document for optimal RAG performance.

        This extracts structured data once, stores it in vector DB, enabling:
        - Faster queries (pre-computed answers)
        - Lower token usage (no need to re-analyze)
        - Better search (rich metadata)
        - Smaller context windows

        Args:
            document_info: Document metadata (title, type, etc.)
            content_preview: First ~1000 chars of document content

        Returns:
            Dict with rich structured metadata
        """
        if not self.client:
            return self._fallback_metadata(document_info)

        try:
            doc_title = document_info.get('title', 'Unknown')
            doc_type = document_info.get('document_type', 'financial document')

            # Build comprehensive extraction prompt
            prompt = f"""Analyze this document and extract ALL useful metadata in structured JSON format.

Document Title: {doc_title}
Document Type: {doc_type}
Content Preview (first 1000 chars):
{content_preview[:1000]}

Extract the following metadata (return "null" or empty array if not applicable):

{{
  "classification": {{
    "primary_category": "legal|financial|correspondence|administrative|other",
    "sub_type": "specific type like 'bank_statement', 'court_order', 'invoice', etc.",
    "confidence": "high|medium|low",
    "industry_context": "brief context"
  }},
  "entities": {{
    "people": ["list of person names with roles if clear"],
    "organizations": ["companies, courts, banks, etc."],
    "locations": ["addresses, jurisdictions"],
    "identifiers": ["account numbers, case numbers, etc."],
    "financial_figures": ["key dollar amounts with context"]
  }},
  "temporal": {{
    "document_date": "YYYY-MM-DD or null",
    "period_start": "YYYY-MM-DD or null",
    "period_end": "YYYY-MM-DD or null",
    "deadlines": ["important dates with descriptions"],
    "time_context": "brief temporal context"
  }},
  "financial_summary": {{
    "beginning_balance": "amount or null",
    "ending_balance": "amount or null",
    "total_amount": "amount or null",
    "key_figures": ["list of important amounts with labels"],
    "account_summary": "brief financial summary"
  }},
  "content_analysis": {{
    "main_topics": ["3-5 key topics"],
    "keywords": ["8-12 searchable keywords"],
    "purpose": "what this document is for",
    "importance_level": "critical|high|medium|low",
    "sentiment": "neutral|positive|negative|urgent|formal"
  }},
  "actionable_intelligence": {{
    "action_items": ["things that need to be done"],
    "deadlines_urgent": ["any urgent deadlines"],
    "red_flags": ["concerns or issues"],
    "follow_up": ["follow-up actions needed"]
  }},
  "relationships": {{
    "references_documents": ["mentioned document names/numbers"],
    "related_parties": ["connected people/orgs"],
    "part_of_series": "yes/no - is this part of a sequence",
    "context": "how this relates to other documents"
  }},
  "qa_pairs": [
    {{"question": "What is this document?", "answer": "concise answer"}},
    {{"question": "Who is involved?", "answer": "key parties"}},
    {{"question": "What are the key amounts?", "answer": "financial figures"}},
    {{"question": "Any important dates?", "answer": "key dates"}},
    {{"question": "What action is required?", "answer": "actions needed"}}
  ],
  "search_tags": ["8-12 tags optimized for search - include topics, entities, types"],
  "one_line_summary": "Ultra-concise 1-line summary for quick scanning"
}}

IMPORTANT:
- Extract only what's evident from the content
- Use null/empty arrays for missing data
- Be precise with dates and amounts
- Focus on searchable, reusable information
- The goal is to make future queries fast by pre-extracting everything useful

Return ONLY valid JSON."""

            # Call LLM with longer max tokens for rich extraction
            response_text = self._call_llm(
                prompt,
                operation='metadata_extraction',
                document_id=document_info.get('id')
            )

            # Parse JSON response with safe parser
            metadata = self._safe_json_parse(response_text)
            if metadata:
                logger.info(f"Extracted rich metadata: {len(str(metadata))} chars, {len(metadata.get('keywords', []))} keywords")
                return metadata
            else:
                logger.error("Could not extract valid JSON from response")
                return self._fallback_metadata(document_info)

        except Exception as e:
            logger.warning(f"Rich metadata extraction failed: {e}")
            return self._fallback_metadata(document_info)

    def analyze_document_integrity(self, document_info: Dict[str, Any], content_preview: str = "",
                                   related_docs: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze document for conflicts, inconsistencies, and quality issues.
        Critical for legal document review - finds problems that could affect case.

        Args:
            document_info: Document metadata
            content_preview: First ~1500 chars of document
            related_docs: Optional list of related documents for cross-doc checking

        Returns:
            Dict with integrity findings, each with detailed evidence
        """
        if not self.client:
            return {'enabled': False, 'findings': []}

        try:
            doc_title = document_info.get('title', 'Unknown')
            doc_type = document_info.get('document_type', 'document')

            # Build related documents context section
            related_context = ""
            if related_docs and len(related_docs) > 0:
                related_context = "\n\n**RELATED DOCUMENTS IN THIS PROJECT/CASE:**\n"
                related_context += "Consider these related documents when assessing if issues are truly missing or contradictory:\n\n"
                for i, rdoc in enumerate(related_docs, 1):
                    related_context += f"{i}. Document #{rdoc.get('document_id')}: {rdoc.get('title')}\n"
                    related_context += f"   Content snippet: {rdoc.get('content_snippet', '')[:300]}...\n"
                    related_context += f"   Relevance: {int(rdoc.get('relevance_score', 0) * 100)}%\n\n"
                related_context += "IMPORTANT: If an issue appears to be missing information or contradictory, CHECK if it's explained in one of these related documents. If so, reference that document instead of flagging it as an issue.\n"

            prompt = f"""Analyze this document for integrity issues, conflicts, and quality problems.
This is for LEGAL REVIEW - be thorough and precise.

Document: {doc_title}
Type: {doc_type}
Content Preview (first 1500 chars):
{content_preview[:1500]}
{related_context}
Analyze for these issues:

1. SELF-CONFLICTS (internal contradictions)
   - Conflicting statements or facts
   - Same item with different values
   - Contradictory dates or timelines
   - Math that doesn't add up
   - Logic inconsistencies

2. SENSE CHECKING (things that don't make sense)
   - Impossible timelines (effect before cause)
   - Unreasonable amounts or values
   - Missing required information
   - Illogical sequences
   - Unexplained gaps

3. QUALITY ISSUES
   - Missing signatures or dates
   - Incomplete information
   - Formatting problems
   - Unclear or ambiguous language
   - Potential redaction needs (PII, SSN, etc.)

4. LEGAL COMPLIANCE
   - Improper citations (if any)
   - Missing exhibit references
   - Date stamp issues
   - Party identification problems

5. CROSS-DOCUMENT CONFLICTS (if related docs provided)
   - Contradictions with other documents in the case
   - Conflicting facts across documents
   - Timeline inconsistencies between documents
   - BUT: If related docs EXPLAIN or CLARIFY something, note that instead of flagging as an issue

**CRITICAL INSTRUCTION:**
If related documents are provided above, cross-reference them BEFORE flagging issues. For example:
- If this document seems to be missing property details, but Document #1234 contains those details, mention that instead
- If this document mentions venue concerns, but another document explains the legal basis, reference that
- If something seems incomplete but is covered in a related document, note "Addressed in Document #1234" instead of flagging as an issue

For EACH issue found, provide:
- Severity: critical|high|medium|low
- Category: conflict|logic_error|missing_info|quality|legal_compliance
- Description: What's wrong
- Evidence: EXACT quote or values that show the problem
- Location: Page number or section if identifiable
- Impact: Why this matters
- Suggested_action: What to do about it

Return JSON:
{{
  "has_issues": true/false,
  "issue_count": number,
  "critical_count": number,
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "category": "conflict|logic_error|missing_info|quality|legal_compliance",
      "issue_type": "specific type like 'date_conflict', 'math_error', etc.",
      "description": "Clear description of the problem",
      "evidence": {{
        "quotes": ["exact quote 1", "exact quote 2"],
        "values": ["conflicting value 1", "conflicting value 2"],
        "context": "surrounding context",
        "location": "page/section reference"
      }},
      "impact": "Why this matters for legal review",
      "suggested_action": "What to do",
      "confidence": "high|medium|low",
      "related_doc_reference": "Document #ID that explains/contradicts this, or null if none"
    }}
  ],
  "summary": "Brief summary of integrity analysis (mention if related docs provide context)",
  "cross_document_notes": "Optional notes about how this document relates to others in the project"
}}

IMPORTANT:
- Only flag real issues, not minor stylistic choices
- Provide EXACT evidence with quotes
- Be specific about location when possible
- Consider legal context and implications
- If no issues found, return empty findings array"""

            response = self._call_llm(
                prompt,
                operation='integrity_check',
                document_id=document_info.get('id')
            )

            # Parse response with safe JSON parser
            integrity_result = self._safe_json_parse(response)
            if integrity_result:
                logger.info(f"Integrity analysis: {integrity_result.get('issue_count', 0)} issues found, {integrity_result.get('critical_count', 0)} critical")
                return integrity_result
            else:
                logger.warning("Could not parse integrity analysis response")
                return {
                    'has_issues': False,
                    'issue_count': 0,
                    'critical_count': 0,
                    'findings': [],
                    'summary': 'Analysis failed to parse'
                }

        except Exception as e:
            logger.warning(f"Integrity analysis failed: {e}")
            return {
                'has_issues': False,
                'issue_count': 0,
                'critical_count': 0,
                'findings': [],
                'summary': f'Analysis error: {str(e)}'
            }

    def _fallback_metadata(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate minimal metadata when LLM extraction fails."""
        title = document_info.get('title', 'Unknown Document')
        doc_type = document_info.get('document_type', 'unknown')

        return {
            "classification": {
                "primary_category": "financial",
                "sub_type": doc_type,
                "confidence": "low",
                "industry_context": "financial document"
            },
            "entities": {"people": [], "organizations": [], "locations": [], "identifiers": [], "financial_figures": []},
            "temporal": {"document_date": None, "period_start": None, "period_end": None, "deadlines": [], "time_context": ""},
            "financial_summary": {"beginning_balance": None, "ending_balance": None, "total_amount": None, "key_figures": [], "account_summary": ""},
            "content_analysis": {
                "main_topics": [title],
                "keywords": [doc_type, "financial"],
                "purpose": "Financial document",
                "importance_level": "medium",
                "sentiment": "neutral"
            },
            "actionable_intelligence": {"action_items": [], "deadlines_urgent": [], "red_flags": [], "follow_up": []},
            "relationships": {"references_documents": [], "related_parties": [], "part_of_series": "no", "context": ""},
            "qa_pairs": [
                {"question": "What is this document?", "answer": title},
                {"question": "What type?", "answer": doc_type}
            ],
            "search_tags": [doc_type, "financial", "document"],
            "one_line_summary": f"{doc_type}: {title}"
        }

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON issues from LLM responses.

        Args:
            json_str: Potentially malformed JSON string

        Returns:
            Repaired JSON string
        """
        import re

        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Remove comments (sometimes LLMs add them)
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        # Fix common quote issues
        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes

        # Remove control characters that break JSON
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

        return json_str.strip()

    def _safe_json_parse(self, response: str) -> Optional[Dict]:
        """
        Safely parse JSON with multiple fallback strategies.

        Args:
            response: LLM response that should contain JSON

        Returns:
            Parsed dict or None if all attempts fail
        """
        import re

        # Strategy 1: Try parsing response directly
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Try repairing and parsing
        try:
            repaired = self._repair_json(response)
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract JSON object from response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Strategy 4: Try repairing extracted JSON
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                repaired = self._repair_json(json_match.group())
                return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        logger.error(f"All JSON parse strategies failed for response: {response[:200]}...")
        return None

    def _call_llm(self, prompt: str, operation: str = 'analysis', document_id: int = None) -> str:
        """
        Call the LLM API with multi-provider fallback.

        Args:
            prompt: The prompt to send to the LLM
            operation: Operation type for usage tracking (e.g., 'metadata_extraction', 'integrity_check')
            document_id: Optional document ID for tracking

        Returns:
            LLM response text
        """
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

                                # Log usage
                                if self.usage_tracker and response.usage:
                                    self.usage_tracker.log_usage(
                                        provider='openai',
                                        model=model,
                                        operation=operation,
                                        input_tokens=response.usage.prompt_tokens,
                                        output_tokens=response.usage.completion_tokens,
                                        document_id=document_id,
                                        success=True
                                    )

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

                                # Log usage
                                if self.usage_tracker and response.usage:
                                    self.usage_tracker.log_usage(
                                        provider='anthropic',
                                        model=model,
                                        operation=operation,
                                        input_tokens=response.usage.input_tokens,
                                        output_tokens=response.usage.output_tokens,
                                        document_id=document_id,
                                        success=True
                                    )

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
            # Try the configured model first, then fallback models (Claude 4.5/4.6 and Claude 3.x)
            models_to_try = [self.model]
            if self.model not in ['claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001', 'claude-3-haiku-20240307']:
                models_to_try.extend(['claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001', 'claude-3-haiku-20240307'])

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

                    # Log usage
                    if self.usage_tracker and response.usage:
                        self.usage_tracker.log_usage(
                            provider='anthropic',
                            model=model,
                            operation=operation,
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            document_id=document_id,
                            success=True
                        )

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

            # Log usage
            if self.usage_tracker and response.usage:
                self.usage_tracker.log_usage(
                    provider='openai',
                    model=self.model,
                    operation=operation,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    document_id=document_id,
                    success=True
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
