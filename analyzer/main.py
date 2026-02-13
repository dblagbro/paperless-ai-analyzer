"""
Paperless AI Analyzer - Main Entry Point

Orchestrates document analysis pipeline.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from analyzer.paperless_client import PaperlessClient
from analyzer.state import StateManager
from analyzer.profile_loader import ProfileLoader
# NOTE: Deterministic checks and forensics handled by paperless-anomaly-detector
# from analyzer.extract.unstructured_extract import UnstructuredExtractor
# from analyzer.checks.deterministic import DeterministicChecker
# from analyzer.forensics.risk_score import ForensicsAnalyzer
from analyzer.vector_store import VectorStore
from analyzer.llm.llm_client import LLMClient
from analyzer.web_ui import start_web_server_thread, update_ui_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Main analyzer orchestrator."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize components
        self.paperless = PaperlessClient(
            base_url=config['paperless_api_base_url'],
            api_token=config['paperless_api_token']
        )

        self.state_manager = StateManager(
            state_path=config.get('state_path', '/app/data/state.json')
        )

        self.profile_loader = ProfileLoader(
            profiles_dir=config.get('profiles_dir', '/app/profiles')
        )

        # Initialize vector store for RAG
        self.vector_store = VectorStore()

        # NOTE: Deterministic checks and forensics are handled by paperless-anomaly-detector
        # This container ONLY does AI/LLM analysis
        # self.extractor = UnstructuredExtractor()  # Not needed for AI-only analysis
        # self.deterministic_checker = DeterministicChecker()  # Handled by anomaly-detector
        # self.forensics_analyzer = ForensicsAnalyzer()  # Handled by anomaly-detector

        # Optional LLM client
        self.llm_enabled = config.get('llm_enabled', False)
        if self.llm_enabled:
            self.llm_client = LLMClient(
                provider=config.get('llm_provider', 'anthropic'),
                api_key=config.get('llm_api_key'),
                model=config.get('llm_model')
            )
        else:
            self.llm_client = None

        self.archive_path = config.get('archive_path', '/paperless/media/documents/archive')

    def is_poor_quality_ocr(self, content: str, document: Dict[str, Any]) -> bool:
        """
        Detect if Paperless OCR content is poor quality and needs Vision AI fallback.

        Args:
            content: OCR text from Paperless
            document: Document metadata

        Returns:
            True if OCR is poor quality, False otherwise
        """
        if not content:
            return True

        content_clean = content.strip()

        # Check 1: Very short content (less than 200 chars for multi-page docs)
        page_count = document.get('page_count', 1)
        if page_count > 1 and len(content_clean) < 200:
            logger.info(f"Poor OCR: {page_count} pages but only {len(content_clean)} chars")
            return True

        # Check 2: No financial data (no dollar amounts or decimal numbers)
        import re
        has_amounts = bool(re.search(r'\$\s*[\d,]+\.\d{2}|\b[\d,]+\.\d{2}\b', content_clean))
        has_dates = bool(re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', content_clean))

        # If it's a financial document but has no amounts, OCR likely failed
        doc_type = document.get('document_type', '')
        title = document.get('title', '').lower()
        if any(kw in title for kw in ['statement', 'invoice', 'receipt', 'report', 'mor']):
            if not has_amounts:
                logger.info(f"Poor OCR: Financial document but no dollar amounts found")
                return True

        # Check 3: Repetitive content (same line repeated many times)
        lines = content_clean.split('\n')
        if len(lines) > 10:
            unique_lines = set(line.strip() for line in lines if len(line.strip()) > 20)
            repetition_ratio = len(lines) / max(len(unique_lines), 1)
            if repetition_ratio > 5:  # Same lines repeated 5+ times
                logger.info(f"Poor OCR: High repetition ratio {repetition_ratio:.1f}")
                return True

        # Check 4: Content is mostly just case numbers or headers
        header_patterns = [
            r'^[\d-]+\s+Doc\s+\d+\s+Filed',  # Court filing headers
            r'^Page\s+\d+\s+of\s+\d+',        # Page headers
            r'^Pg\s+\d+\s+of\s+\d+'           # Page headers variant
        ]

        header_lines = sum(1 for line in lines if any(re.match(p, line.strip(), re.I) for p in header_patterns))
        if len(lines) > 0 and header_lines / len(lines) > 0.7:  # 70%+ header lines
            logger.info(f"Poor OCR: {header_lines}/{len(lines)} lines are headers")
            return True

        return False

    def extract_with_vision_ai(self, document: Dict[str, Any]) -> Optional[str]:
        """
        Extract text from document using Claude Vision API as OCR fallback.

        Args:
            document: Document metadata from Paperless

        Returns:
            Extracted text content, or None if extraction failed
        """
        doc_id = document['id']
        doc_title = document.get('title', f'Document {doc_id}')

        try:
            logger.info(f"Using Vision AI fallback for document {doc_id}: {doc_title}")

            # Download PDF from Paperless
            pdf_bytes = self.paperless.download_document(doc_id)
            if not pdf_bytes:
                logger.error(f"Failed to download document {doc_id} for Vision AI")
                return None

            # Convert PDF to images (first 10 pages max to control cost)
            try:
                from pdf2image import convert_from_bytes
                import io
                import base64

                # Convert PDF pages to images
                images = convert_from_bytes(pdf_bytes, first_page=1, last_page=10)
                logger.info(f"Converted {len(images)} pages to images for Vision AI")

                # Process each page with Claude Vision
                extracted_content = []

                for page_num, image in enumerate(images, 1):
                    # Convert image to base64
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                    # Use Claude Vision to extract text/tables
                    if self.llm_client:
                        vision_prompt = """Extract all text, numbers, tables, and financial data from this document image.

Focus on:
1. Financial amounts (dollar values, totals, subtotals)
2. Dates and transaction details
3. Tables with rows and columns
4. Account numbers, check numbers
5. Any text content

Format the output as structured text that preserves the layout and relationships between data."""

                        # Call Claude Vision API
                        response = self.llm_client.client.messages.create(
                            model="claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet with vision
                            max_tokens=2000,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": img_base64
                                            }
                                        },
                                        {
                                            "type": "text",
                                            "text": vision_prompt
                                        }
                                    ]
                                }
                            ]
                        )

                        page_content = response.content[0].text
                        extracted_content.append(f"\n--- Page {page_num} ---\n{page_content}")
                        logger.info(f"Vision AI extracted {len(page_content)} chars from page {page_num}")

                # Combine all pages
                full_content = "\n".join(extracted_content)
                logger.info(f"Vision AI total extraction: {len(full_content)} chars from {len(images)} pages")

                return full_content

            except ImportError as e:
                logger.error(f"pdf2image not available: {e}")
                return None
            except Exception as e:
                logger.error(f"Vision AI extraction failed: {e}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Failed to extract with Vision AI for document {doc_id}: {e}", exc_info=True)
            return None

    def run_polling_loop(self) -> None:
        """Main polling loop."""
        poll_interval = self.config.get('poll_interval_seconds', 30)
        logger.info(f"Starting polling loop (interval={poll_interval}s)")

        # Start web UI if enabled
        if self.config.get('web_ui_enabled', True):
            web_host = self.config.get('web_host', '0.0.0.0')
            web_port = self.config.get('web_port', 8051)
            start_web_server_thread(
                self.state_manager,
                self.profile_loader,
                self.paperless,
                host=web_host,
                port=web_port
            )

        # Health check
        if not self.paperless.health_check():
            logger.error("Paperless API health check failed, exiting")
            sys.exit(1)

        while True:
            try:
                self.poll_and_analyze()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in polling loop: {e}", exc_info=True)

            time.sleep(poll_interval)

    def poll_and_analyze(self) -> None:
        """Poll for new documents and analyze them."""
        logger.info("Polling for new documents...")

        # Get documents modified after last seen (or all documents if in reprocess mode)
        state_stats = self.state_manager.get_stats()
        modified_after = state_stats.get('last_seen_modified')

        # In reprocess_all_mode, don't filter by modified date
        reprocess_mode = self.state_manager.state.reprocess_all_mode
        if reprocess_mode:
            modified_after = None
            logger.info("Reprocess all mode: fetching all documents")

        try:
            # Fetch all documents with pagination
            documents = []
            page = 1
            while True:
                logger.info(f"Fetching page {page}...")
                response = self.paperless.get_documents(
                    ordering='-modified',
                    page_size=100,
                    page=page,
                    modified_after=modified_after
                )

                page_results = response.get('results', [])
                documents.extend(page_results)
                logger.info(f"Page {page}: fetched {len(page_results)} documents, total so far: {len(documents)}")

                # Check if there are more pages
                has_next = response.get('next')
                logger.info(f"Page {page}: has_next={bool(has_next)}")
                if not has_next:
                    break

                page += 1

            logger.info(f"Found {len(documents)} documents to check (across {page} pages)")

            processed_count = 0
            skipped_count = 0

            for doc in documents:
                doc_id = doc['id']
                doc_modified = doc['modified']

                # Check if we should process this document
                if not self.state_manager.should_process_document(doc_modified, doc_id):
                    logger.debug(f"Skipping document {doc_id} (already processed)")
                    skipped_count += 1
                    continue

                # Analyze document
                logger.info(f"Analyzing document {doc_id}: {doc['title']}")
                self.analyze_document(doc)

                # Update state - in reprocess mode, just track IDs
                if reprocess_mode:
                    with self.state_manager.lock:
                        self.state_manager.state.last_seen_ids.add(doc_id)
                        self.state_manager._save_state()
                else:
                    self.state_manager.update_last_seen(doc_modified, {doc_id})
                processed_count += 1

            if processed_count > 0:
                self.state_manager.mark_processed()
                logger.info(f"Processed {processed_count} new documents")

                # Exit reprocess mode if we've seen all documents
                if reprocess_mode and skipped_count == 0 and len(documents) < 100:
                    with self.state_manager.lock:
                        self.state_manager.state.reprocess_all_mode = False
                        self.state_manager.state.last_seen_modified = None
                        self.state_manager.state.last_seen_ids = set()
                        self.state_manager._save_state()
                    logger.info("Reprocess all mode complete - returning to incremental mode")
            else:
                logger.info("No new documents to process")

        except Exception as e:
            logger.error(f"Failed to poll documents: {e}", exc_info=True)

    def analyze_document(self, document: Dict[str, Any]) -> None:
        """
        Analyze a single document.

        Args:
            document: Document data from Paperless API
        """
        doc_id = document['id']
        doc_title = document['title']

        try:
            # Step 1: Match profile
            profile, match_score = self.profile_loader.match_profile(document)

            if profile:
                logger.info(f"Matched profile: {profile.profile_id} (score={match_score:.2f})")
            else:
                logger.info(f"No profile matched, using generic checks")

                # Generate staging profile suggestion
                try:
                    staging_file = self.profile_loader.generate_staging_profile(document)
                    logger.info(f"Generated staging profile: {staging_file}")

                    # Tag document as needing profile
                    self.paperless.update_document_tags(doc_id, ['needs_profile:unmatched'])
                except Exception as e:
                    logger.error(f"Failed to generate staging profile: {e}")

            # Step 2: Read existing tags from anomaly-detector
            # The anomaly-detector container writes deterministic and forensic tags
            # We read those to provide context to the LLM
            existing_tags = self.paperless.get_document_tags(doc_id)
            logger.info(f"Existing tags: {existing_tags}")

            # Extract anomaly tags written by anomaly-detector
            anomaly_tags = [tag for tag in existing_tags if tag.startswith('anomaly:')]
            deterministic_results = {'anomalies_found': anomaly_tags, 'details': {}}

            # Extract forensic risk from tags
            risk_score = 0
            if 'anomaly:forensic_risk_high' in existing_tags:
                risk_score = 80
            elif 'anomaly:forensic_risk_medium' in existing_tags:
                risk_score = 60
            elif 'anomaly:forensic_risk_low' in existing_tags:
                risk_score = 30

            forensics_results = {'risk_score_percent': risk_score, 'signals': []}
            logger.info(f"Read {len(anomaly_tags)} anomaly tags from anomaly-detector, risk_score: {risk_score}%")

            # Step 3: Optional LLM analysis (ONLY AI analysis)
            # Pass anomaly context to LLM for AI-based analysis
            llm_results = None
            if self.llm_enabled and self.llm_client:
                llm_results = self.llm_client.analyze_anomalies(
                    document,
                    deterministic_results,
                    {},  # No extracted data needed - anomalies come from detector
                    forensics_results
                )
                narrative = llm_results.get('narrative', '')
                logger.info(f"LLM analysis: {narrative[:100]}...")

                # Save AI analysis as document note
                if narrative and llm_results.get('enabled'):
                    ai_note = self._format_ai_note(llm_results, deterministic_results, forensics_results)
                    self.paperless.add_document_note(doc_id, ai_note, append=True)

            # Step 7: Compile tags and write back
            tags_to_add = self._compile_tags(
                profile,
                deterministic_results,
                forensics_results,
                llm_results
            )

            if tags_to_add:
                logger.info(f"Adding tags to document {doc_id}: {tags_to_add}")
                self.paperless.update_document_tags(doc_id, tags_to_add)

            # Update UI stats
            update_ui_stats({
                'doc_id': doc_id,
                'title': doc_title,
                'timestamp': datetime.utcnow().isoformat(),
                'profile_matched': profile.profile_id if profile else None,
                'anomalies_found': deterministic_results.get('anomalies_found', []),
                'risk_score': risk_score,
                'tags_added': tags_to_add
            })

            logger.info(f"Successfully analyzed document {doc_id}")

            # Embed document for RAG (semantic search)
            try:
                if self.vector_store.enabled:
                    # Prepare content for embedding
                    content_parts = []

                    # PRIMARY: Add actual document content from Paperless (OCR'd text)
                    paperless_content = document.get('content', '').strip()

                    # Check OCR quality and use Vision AI fallback if poor
                    final_content = paperless_content
                    used_vision_ai = False

                    if self.is_poor_quality_ocr(paperless_content, document):
                        logger.warning(f"Poor OCR quality detected for document {doc_id}, attempting Vision AI fallback")

                        # Try Vision AI extraction
                        vision_content = self.extract_with_vision_ai(document)

                        if vision_content and len(vision_content) > len(paperless_content):
                            logger.info(f"Vision AI extraction successful: {len(vision_content)} chars vs {len(paperless_content)} from Paperless OCR")
                            final_content = vision_content
                            used_vision_ai = True

                            # Tag document to indicate Vision AI was used
                            try:
                                self.paperless.update_document_tags(doc_id, ['aianomaly:vision_ai_extracted'])
                            except Exception as e:
                                logger.warning(f"Failed to add vision_ai tag: {e}")
                        else:
                            logger.warning(f"Vision AI extraction did not improve content quality")

                    if final_content:
                        # For legal/court documents, include full content (no truncation)
                        # Embedding models can handle large context
                        extraction_method = "Vision AI" if used_vision_ai else "Paperless OCR"
                        content_parts.append(f"Document Content (extracted via {extraction_method}):\n{final_content}")
                    else:
                        # No content available - add note for better searchability
                        content_parts.append("Document Content: (No extractable text - OCR and Vision AI both failed)")

                    # Add document metadata for better searchability
                    metadata_parts = []
                    if document.get('page_count'):
                        metadata_parts.append(f"{document['page_count']} pages")
                    if document.get('correspondent'):
                        metadata_parts.append(f"Correspondent: {document['correspondent']}")
                    if document.get('created'):
                        metadata_parts.append(f"Created: {document['created']}")
                    if metadata_parts:
                        content_parts.append(f"\nDocument Metadata: {', '.join(metadata_parts)}")

                    # Add AI analysis if available
                    if llm_results and llm_results.get('narrative'):
                        content_parts.append(f"\nAI Analysis: {llm_results['narrative']}")

                    # Add anomalies found by detector
                    if deterministic_results.get('anomalies_found'):
                        content_parts.append(f"\nAnomalies Detected: {', '.join(deterministic_results['anomalies_found'])}")

                    # Add forensic info
                    if forensics_results.get('signals'):
                        signals = [s.get('description', s.get('type', '')) for s in forensics_results['signals'][:5]]
                        content_parts.append(f"\nForensic Signals: {'; '.join(signals)}")

                    content_text = "\n\n".join(content_parts)

                    # Determine document type from profile
                    if profile and hasattr(profile, 'profile_id'):
                        document_type = profile.profile_id
                    elif profile and hasattr(profile, 'document_type'):
                        document_type = profile.document_type
                    else:
                        document_type = 'unknown'

                    # Embed and store
                    self.vector_store.embed_document(
                        document_id=doc_id,
                        title=doc_title,
                        content=content_text,
                        metadata={
                            'risk_score': risk_score,
                            'anomalies': deterministic_results.get('anomalies_found', []),
                            'timestamp': datetime.utcnow().isoformat(),
                            'document_type': document_type
                        }
                    )
            except Exception as embed_error:
                logger.warning(f"Failed to embed document {doc_id}: {embed_error}")

        except Exception as e:
            logger.error(f"Failed to analyze document {doc_id}: {e}", exc_info=True)

    def _get_pdf_path(self, document: Dict[str, Any]) -> Optional[str]:
        """Get path to archived PDF on disk."""
        archived_filename = document.get('archived_file_name')
        if not archived_filename:
            return None

        # Archived PDFs are stored as NNNNNNN.pdf where N is the document ID
        doc_id = document['id']
        pdf_filename = f"{doc_id:07d}.pdf"

        pdf_path = Path(self.archive_path) / pdf_filename
        return str(pdf_path) if pdf_path.exists() else None

    def _format_ai_note(self,
                       llm_results: Dict[str, Any],
                       deterministic_results: Dict[str, Any],
                       forensics_results: Dict[str, Any]) -> str:
        """
        Format AI analysis as a readable note.

        Args:
            llm_results: LLM analysis results
            deterministic_results: Deterministic check results
            forensics_results: Forensics analysis results

        Returns:
            Formatted note text
        """
        from datetime import datetime

        note_parts = []
        note_parts.append("🤖 AI ANOMALY ANALYSIS")
        note_parts.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        note_parts.append("")

        # AI Narrative
        narrative = llm_results.get('narrative', '')
        if narrative:
            note_parts.append("📝 SUMMARY")
            note_parts.append(narrative)
            note_parts.append("")

        # Confidence
        confidence = llm_results.get('confidence', 'medium')
        note_parts.append(f"Confidence Level: {confidence.upper()}")
        note_parts.append("")

        # Detected Issues
        anomalies = deterministic_results.get('anomalies_found', [])
        if anomalies:
            note_parts.append("⚠️ DETECTED ANOMALIES")
            anomaly_descriptions = {
                'balance_mismatch': 'Running balance calculations do not match expected values',
                'continuity_mismatch': 'Ending balance of one period does not match starting balance of next',
                'date_order_violation': 'Transactions are not in chronological order',
                'duplicate_transactions': 'Duplicate transactions detected',
                'missing_data': 'Required fields are missing or incomplete'
            }
            for anomaly in anomalies:
                desc = anomaly_descriptions.get(anomaly, anomaly.replace('_', ' ').title())
                note_parts.append(f"• {desc}")
            note_parts.append("")

        # Forensic Analysis
        risk_score = forensics_results.get('risk_score_percent', 0)
        if risk_score > 0:
            risk_level = 'HIGH' if risk_score >= 70 else 'MEDIUM' if risk_score >= 40 else 'LOW'
            pages_analyzed = forensics_results.get('pages_analyzed', 0)
            note_parts.append(f"🔍 FORENSIC RISK: {risk_level} ({risk_score}%)")
            note_parts.append(f"Pages analyzed: {pages_analyzed}")
            note_parts.append("")
            note_parts.append("How this score is calculated:")
            note_parts.append("The system analyzes each page for three types of manipulation indicators:")
            note_parts.append("• Compression inconsistency (unusual JPEG artifacts, max 25 pts)")
            note_parts.append("• Noise inconsistency (uneven noise patterns, max 20 pts)")
            note_parts.append("• Edge anomalies (suspicious edge patterns, variable pts)")
            note_parts.append("Score = highest page score × (0.7 + 0.3 × affected pages ratio)")
            note_parts.append("")

            signals = forensics_results.get('signals', [])
            if signals:
                note_parts.append("Detected indicators:")
                # Group signals by type
                signal_counts = {}
                for signal in signals:
                    sig_type = signal.get('type', 'unknown')
                    if sig_type not in signal_counts:
                        signal_counts[sig_type] = {'count': 0, 'weight': 0, 'desc': signal.get('description', '')}
                    signal_counts[sig_type]['count'] += 1
                    signal_counts[sig_type]['weight'] += signal.get('weight', 0)

                for sig_type, info in signal_counts.items():
                    note_parts.append(f"• {sig_type.replace('_', ' ').title()}: {info['count']} instances, {info['weight']:.1f} pts total")
            note_parts.append("")

        # Recommended Actions
        actions = llm_results.get('recommended_actions', [])
        if actions:
            note_parts.append("✅ RECOMMENDED ACTIONS")
            for action in actions:
                note_parts.append(f"• {action}")
            note_parts.append("")

        # Footer
        note_parts.append("---")
        note_parts.append("Analyzed by Paperless AI Analyzer with Claude AI")

        return '\n'.join(note_parts)

    def _compile_tags(self,
                     profile: Any,
                     deterministic_results: Dict[str, Any],
                     forensics_results: Dict[str, Any],
                     llm_results: Optional[Dict[str, Any]]) -> List[str]:
        """
        Compile list of tags to add to document.

        Args:
            profile: Matched profile
            deterministic_results: Deterministic check results
            forensics_results: Forensics analysis results
            llm_results: Optional LLM analysis results

        Returns:
            List of tag names
        """
        tags = []

        # ONLY write AI-related tags (and only if AI found something)
        # The anomaly-detector container handles deterministic and forensic tags

        # LLM suggested tags (AI-detected anomalies)
        # Only add tags if AI actually found issues - no tags for clean documents
        if llm_results:
            ai_tags = llm_results.get('suggested_tags', [])
            # Ensure they use aianomaly: prefix
            for tag in ai_tags:
                if tag.startswith('aianomaly:'):
                    tags.append(tag)

        return tags

    def analyze_single_document(self, doc_id: int, dry_run: bool = False) -> None:
        """
        Analyze a single document by ID (for testing/debugging).

        Args:
            doc_id: Document ID
            dry_run: If True, don't write tags back
        """
        logger.info(f"Analyzing document {doc_id} (dry_run={dry_run})")

        document = self.paperless.get_document(doc_id)
        self.analyze_document(document)

        if dry_run:
            logger.info("Dry run mode - tags not written to Paperless")


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        'paperless_api_base_url': os.getenv('PAPERLESS_API_BASE_URL', 'http://paperless-web:8000'),
        'paperless_api_token': os.getenv('PAPERLESS_API_TOKEN', ''),
        'poll_interval_seconds': int(os.getenv('POLL_INTERVAL_SECONDS', '30')),
        'state_path': os.getenv('STATE_PATH', '/app/data/state.json'),
        'profiles_dir': os.getenv('PROFILES_DIR', '/app/profiles'),
        # NOTE: archive_path, balance_tolerance, forensics_dpi no longer needed
        # Deterministic checks and forensics handled by paperless-anomaly-detector
        'llm_enabled': os.getenv('LLM_ENABLED', 'false').lower() == 'true',
        'llm_provider': os.getenv('LLM_PROVIDER', 'anthropic'),
        'llm_api_key': os.getenv('LLM_API_KEY'),
        'llm_model': os.getenv('LLM_MODEL'),
        'web_ui_enabled': os.getenv('WEB_UI_ENABLED', 'true').lower() == 'true',
        'web_host': os.getenv('WEB_HOST', '0.0.0.0'),
        'web_port': int(os.getenv('WEB_PORT', '8051')),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Paperless AI Analyzer')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no tag writes)')
    parser.add_argument('--doc-id', type=int, help='Analyze single document by ID')
    args = parser.parse_args()

    config = load_config()

    if not config['paperless_api_token']:
        logger.error("PAPERLESS_API_TOKEN not set")
        sys.exit(1)

    analyzer = DocumentAnalyzer(config)

    if args.doc_id:
        # Single document mode
        analyzer.analyze_single_document(args.doc_id, dry_run=args.dry_run)
    else:
        # Polling mode
        analyzer.run_polling_loop()


if __name__ == '__main__':
    main()
