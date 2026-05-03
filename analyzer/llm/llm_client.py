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
2. FULL (minimum 3 sentences, target 4-5): A detailed summary covering purpose, key details, dates or amounts if present, and relevant context. Must be at least 3 complete sentences.

Respond in JSON format:
{{
  "brief": "One sentence description",
  "full": "Minimum three sentences. Four or five sentences preferred. Include specific details from the document."
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

            # v3.9.21: split the prompt — STABLE template into ``system``
            # (with cache_control: ephemeral so claude-oauth caches it),
            # VARIABLE per-doc payload into ``user``. The legal-review
            # bulk job sends ~17k of these per period and the bulk of every
            # prompt is the same instructions + format spec; caching that
            # prefix collapses input-token cost on every call after the first.
            #
            # v3.9.22: prefix size matters. Anthropic's per-model auto-cache
            # threshold is ~2048 tokens for Haiku (the cheapest tier we want
            # to route to). The v3.9.21 prefix was ~833-1111 tokens — below
            # threshold, so caching silently skipped. Padded with substantive
            # rubric examples, edge-case guidance, and concrete severity-
            # calibration material to land firmly above 2048 tokens. This is
            # not filler — it improves output quality (more grounded
            # severity calls, fewer false positives on minor formatting) AND
            # makes the prefix cacheable.
            system_prompt = """You are a senior legal document reviewer assisting an attorney
on case preparation. Your task is to analyze each document for integrity
issues, internal conflicts, and quality problems that could affect
the case. Be thorough and precise. Findings are read by an attorney
and may be cited in motions or exhibits — only flag real issues
backed by exact evidence from the document. Speculation and stylistic
preferences are not findings.

# WHAT TO ANALYZE

## 1. SELF-CONFLICTS (internal contradictions inside this single document)
   - Conflicting statements or facts
     Example: "Defendant denies all allegations" appears in para 3,
     "Defendant admits paragraph 7" appears in para 14, but the same
     facts are at issue in both — flag as `category=conflict`,
     `issue_type=admit_deny_inconsistency`.
   - Same item with different values
     Example: "Total: $12,450" on page 2, "Total: $12,540" on page 5.
     Always cite both quotes verbatim in evidence.quotes.
   - Contradictory dates or timelines
     Example: "Filed: 2024-03-15" but "Effective: 2024-02-10" with no
     explanation of retroactive effect.
   - Math that doesn't add up
     Example: line items sum to $9,800 but total reads $9,000. Show
     the actual sum in evidence.values.
   - Logic inconsistencies
     Example: party named "John Smith" in caption but "Jon Smith" in
     signature block on the same instrument.

## 2. SENSE CHECKING (things that don't make sense in context)
   - Impossible timelines — effect before cause, dates outside the
     document's stated period, deadlines with no triggering event.
   - Unreasonable amounts or values that don't fit the document type.
     Example: a residential lease with $1.2M annual rent absent
     special-property explanation.
   - Missing required information for the document type (a 1099 with
     no payer TIN, a deed with no legal description, etc.).
   - Illogical sequences — exhibit cross-references to documents that
     don't exist in this filing.
   - Unexplained gaps in continuity — bank statement skipping months,
     receipts referenced but not attached.

## 3. QUALITY ISSUES (clarity / completeness problems)
   - Missing signatures or dates on instruments that legally require them.
   - Incomplete information — TBD, blank fields, [REDACTED] on what
     should be public-record content, "see attached" with nothing attached.
   - Formatting problems severe enough to change interpretation
     (mid-sentence cutoffs, OCR errors that swap key digits).
   - Unclear or ambiguous language on operative provisions only — do
     NOT flag general legalese style.
   - Potential redaction needs — PII, SSN, account numbers, minor names,
     medical data, that should be redacted before filing.

## 4. LEGAL COMPLIANCE (procedural / formal defects)
   - Improper citations — wrong reporter, missing pin cite, broken
     subsequent history (overruled / superseded without note).
   - Missing exhibit references — "Exhibit A" cited in body, no Exhibit
     A in document, or vice versa.
   - Date stamp issues — file-stamp date earlier than signature date,
     notarization date inconsistent with party signature date.
   - Party identification problems — caption parties don't match parties
     named in the body, capacity (individual / trustee / officer) not
     consistent throughout.

## 5. CROSS-DOCUMENT CONFLICTS (only if related docs are provided)
   - Contradictions with other documents in the case.
   - Conflicting facts across documents (different addresses for the
     same party, different dates for the same event).
   - Timeline inconsistencies between documents.
   - **BUT** if a related doc EXPLAINS or CLARIFIES something that
     would otherwise look like an issue, note that explanation rather
     than flagging it. Example: a bank statement appears to skip a
     month, but a related closing-statement document explains the
     account was held in escrow that month — flag this as
     `category=conflict` only if the explanation actually contradicts
     it; otherwise note "Addressed in Document #N" and move on.

# CRITICAL CROSS-REFERENCE INSTRUCTION
If related documents are provided in the user prompt, cross-reference
them BEFORE flagging issues:
- If this document seems to be missing property details, but Doc #1234
  contains those details, mention that instead.
- If this document mentions venue concerns, but another document
  explains the legal basis, reference that.
- If something seems incomplete but is covered in a related document,
  note "Addressed in Document #1234" instead of flagging it.
This is the most common reason for false-positive findings on legal
documents reviewed in isolation.

# SEVERITY CALIBRATION (use these definitions consistently)
- critical: would void the document, change the case outcome, or
  require immediate refiling. Example: signature missing on the
  instrument creating the legal obligation.
- high: would weaken the document's evidentiary value or invite
  successful objection. Example: party-identification inconsistency
  on a contract.
- medium: would require explanation or supplementation but doesn't
  affect validity. Example: math error in a non-operative summary.
- low: cosmetic or process notes. Example: unusual but not improper
  citation format.
Default to medium when uncertain. Don't inflate severity to look
thorough.

# OUTPUT
For EACH issue found, provide:
- severity: critical|high|medium|low (per calibration above)
- category: conflict|logic_error|missing_info|quality|legal_compliance
- issue_type: specific snake_case type (date_conflict, math_error,
  missing_signature, citation_error, party_mismatch, etc.)
- description: 1-2 sentence plain-language description of the problem
- evidence:
  - quotes: array of EXACT verbatim quotes from the document
  - values: array of conflicting values where applicable
  - context: surrounding context that helps the attorney locate it
  - location: page or section reference if identifiable
- impact: why this matters for legal review specifically
- suggested_action: what the attorney should do (e.g., "request
  amended filing", "address in deposition", "verify with client")
- confidence: high|medium|low — your certainty this is a real issue
- related_doc_reference: "Document #ID that explains/contradicts this",
  or null if none

Return strictly this JSON shape:
{{
  "has_issues": true/false,
  "issue_count": number,
  "critical_count": number,
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "category": "conflict|logic_error|missing_info|quality|legal_compliance",
      "issue_type": "specific snake_case type",
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

# REVIEW DISCIPLINE — read carefully before producing findings
- Only flag real issues, not minor stylistic choices.
- Provide EXACT evidence with quotes — paraphrasing is not evidence.
- Be specific about location when possible — page numbers and section
  identifiers help the attorney verify quickly.
- Consider legal context and implications, not just textual mistakes.
- If no issues found, return an empty findings array with has_issues=false.
- If the document is too short, redacted, or unreadable to assess, set
  has_issues=false and put the reason in summary — do not invent
  findings to look thorough.
- If related docs were provided and resolved a candidate issue, do
  NOT include it in findings; mention it in cross_document_notes only."""

            # v3.9.21: variable per-document payload goes in ``user``. This
            # is the only part that changes between calls — the system_prompt
            # above is identical across the entire bulk legal-review job
            # and gets cached on the proxy after the first call.
            user_prompt = f"""Document: {doc_title}
Type: {doc_type}
Content Preview (first 1500 chars):
{content_preview[:1500]}
{related_context}"""

            response = self._call_llm_cached(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                operation='integrity_check',
                document_id=document_info.get('id'),
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
        Call the LLM API with proxy-first routing and direct-provider fallback.

        Routes through analyzer.llm.proxy_call.call_llm() which:
          1. Tries each healthy llm_proxy_endpoint in priority order.
          2. Falls through to direct Anthropic/OpenAI SDK if pool is exhausted.
          3. Raises LLMUnavailableError if both paths fail.

        Args:
            prompt: The prompt to send to the LLM
            operation: Operation type for usage tracking (e.g., 'metadata_extraction', 'integrity_check')
            document_id: Optional document ID for tracking

        Returns:
            LLM response text
        """
        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError

        try:
            result = call_llm(
                messages=[{"role": "user", "content": prompt}],
                task="analysis",
                max_tokens=1024,
                operation=operation,
                document_id=document_id,
                usage_tracker=self.usage_tracker,
                direct_provider=self.provider,
                direct_api_key=self.api_key,
                direct_model=self.model,
            )
            return result["content"]
        except LLMUnavailableError as e:
            logger.error(f"LLM unavailable for operation={operation}: {e} (attempted={e.attempted})")
            raise Exception(str(e)) from e

    def _call_llm_cached(self, system_prompt: str, user_prompt: str,
                        operation: str = 'analysis', document_id: int = None,
                        max_tokens: int = 1024) -> str:
        """Call the LLM with the stable instruction template in ``system`` and
        the per-call variable content in ``user``, with ``cache_control:
        ephemeral`` on the system block.

        v3.9.21: introduced after the LLM-Proxy team's Round 4 reply explained
        why our v3.9.20 cache-token logging was always 0 — claude-oauth's
        per-model auto-cache thresholds are ~1024 (Sonnet) / ~2048 (Haiku) /
        ~4096 (Opus), and the proxy's auto-injection only wraps ``system`` +
        last ``tool`` blocks. Our legal-review template lived in the user
        message, below threshold, so neither auto-wrap nor caller-side
        cache_control was active. This helper fixes both: caller-side wrap
        guarantees cache regardless of threshold, and putting the stable
        prefix in ``system`` matches the proxy's auto-inject path too.

        High-volume callers should migrate to this method when the call
        carries a stable instruction template + a variable per-call payload.
        ``analyze_document_integrity`` is the canonical example.
        """
        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError

        try:
            result = call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                task="analysis",
                max_tokens=max_tokens,
                operation=operation,
                document_id=document_id,
                usage_tracker=self.usage_tracker,
                direct_provider=self.provider,
                direct_api_key=self.api_key,
                direct_model=self.model,
                cache_system=True,
            )
            return result["content"]
        except LLMUnavailableError as e:
            logger.error(f"LLM unavailable for operation={operation}: {e} (attempted={e.attempted})")
            raise Exception(str(e)) from e

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
