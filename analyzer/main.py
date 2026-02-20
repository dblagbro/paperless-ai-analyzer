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
from analyzer.project_manager import ProjectManager  # v1.5.0
from analyzer.smart_upload import SmartUploader  # v1.5.0
# NOTE: Deterministic checks and forensics handled by paperless-anomaly-detector
# from analyzer.extract.unstructured_extract import UnstructuredExtractor
# from analyzer.checks.deterministic import DeterministicChecker
# from analyzer.forensics.risk_score import ForensicsAnalyzer
from analyzer.vector_store import VectorStore
from analyzer.llm.llm_client import LLMClient
from analyzer.llm_usage_tracker import LLMUsageTracker
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

        # Use default project for now (full multi-tenancy in later phase)
        project_slug = config.get('project_slug', 'default')
        self.state_manager = StateManager(
            state_dir=config.get('state_dir', '/app/data'),
            project_slug=project_slug
        )

        self.profile_loader = ProfileLoader(
            profiles_dir=config.get('profiles_dir', '/app/profiles')
        )

        # Initialize vector store for RAG (project-aware)
        self.vector_store = VectorStore(project_slug=project_slug)

        # NOTE: Deterministic checks and forensics are handled by paperless-anomaly-detector
        # This container ONLY does AI/LLM analysis
        # self.extractor = UnstructuredExtractor()  # Not needed for AI-only analysis
        # self.deterministic_checker = DeterministicChecker()  # Handled by anomaly-detector
        # self.forensics_analyzer = ForensicsAnalyzer()  # Handled by anomaly-detector

        # Initialize LLM usage tracker
        logger.info("Initializing LLM Usage Tracker...")
        self.usage_tracker = LLMUsageTracker(
            db_path=config.get('usage_db_path', '/app/data/llm_usage.db')
        )
        logger.info("LLM Usage Tracker initialized successfully")

        # Optional LLM client
        self.llm_enabled = config.get('llm_enabled', False)
        if self.llm_enabled:
            self.llm_client = LLMClient(
                provider=config.get('llm_provider', 'anthropic'),
                api_key=config.get('llm_api_key'),
                model=config.get('llm_model'),
                usage_tracker=self.usage_tracker
            )
        else:
            self.llm_client = None

        # v1.5.0: Project management and smart upload
        logger.info("Initializing ProjectManager...")
        self.project_manager = ProjectManager()
        logger.info("ProjectManager initialized successfully")

        logger.info("Initializing SmartUploader...")
        self.smart_uploader = SmartUploader(
            llm_client=self.llm_client,
            paperless_client=self.paperless,
            project_manager=self.project_manager
        ) if self.llm_enabled else None
        logger.info("SmartUploader initialized successfully")

        # v1.5.0: Re-analysis tracking for project-wide context
        self.last_document_time = {}  # project_slug -> timestamp of last processed doc
        self.last_reanalysis_time = {}  # project_slug -> timestamp of last re-analysis
        self.reanalysis_delay_seconds = 300  # 5 minutes after last doc before re-analyzing
        self.documents_processed_this_cycle = 0

        # v2.0.4: Stale embedding detection
        self._stale_check_counter = 0  # Incremented each poll; triggers check every 10 polls

        # v2.0.5: Guard so re_analyze_project never runs concurrently with itself
        self._reanalysis_running = False

        self.archive_path = config.get('archive_path', '/paperless/media/documents/archive')
        logger.info("DocumentAnalyzer initialization complete")

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
                            model="claude-sonnet-4-6",  # Claude Sonnet 4.6 with vision
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

    def check_and_trigger_reanalysis(self) -> None:
        """
        Check if any project needs re-analysis with full context.
        Triggers automatic re-analysis 5 minutes after last document processed.
        """
        import time

        current_time = time.time()
        project_slug = self.config.get('project_slug', 'default')

        # Check if documents were processed and enough time has passed
        if project_slug in self.last_document_time:
            last_doc_time = self.last_document_time[project_slug]
            last_reanalysis = self.last_reanalysis_time.get(project_slug, 0)
            time_since_last_doc = current_time - last_doc_time

            # Trigger if: 5 minutes passed AND we haven't re-analyzed since last doc
            if (time_since_last_doc >= self.reanalysis_delay_seconds and
                last_doc_time > last_reanalysis):

                if self._reanalysis_running:
                    logger.debug("Re-analysis already in progress, skipping trigger")
                    return

                logger.info(f"⏰ Triggering automatic re-analysis for project '{project_slug}' " +
                           f"({int(time_since_last_doc/60)} minutes since last document)")
                self.last_reanalysis_time[project_slug] = current_time
                self._reanalysis_running = True

                def _run_reanalysis():
                    try:
                        self.re_analyze_project(project_slug)
                    finally:
                        self._reanalysis_running = False

                import threading
                threading.Thread(target=_run_reanalysis, daemon=True).start()

    def re_analyze_project(self, project_slug: str) -> None:
        """
        Re-analyze all documents in a project with full project context.
        This ensures every document sees ALL other documents in the project.

        Args:
            project_slug: Project identifier to re-analyze
        """
        if not self.llm_enabled or not self.llm_client:
            logger.warning("Cannot re-analyze: LLM not enabled")
            return

        try:
            logger.info(f"🔄 Starting project-wide re-analysis for '{project_slug}'...")

            # Get all documents in the project
            # For now, get all documents (in v1.5.0 we'd filter by project tag)
            all_docs = []
            page = 1
            while True:
                response = self.paperless.get_documents(
                    ordering='-modified',
                    page_size=100,
                    page=page
                )
                page_results = response.get('results', [])
                all_docs.extend(page_results)

                if not response.get('next'):
                    break
                page += 1

            logger.info(f"📊 Re-analyzing {len(all_docs)} documents with full project context...")

            reanalyzed_count = 0
            for doc in all_docs:
                doc_id = doc['id']
                doc_title = doc.get('title', f'Document {doc_id}')

                try:
                    # Get document content
                    full_doc = self.paperless.get_document(doc_id)
                    content = full_doc.get('content', '')

                    if not content or len(content) < 50:
                        continue

                    content_preview = content[:1500]

                    # Query vector store for related documents (now includes ALL docs in project)
                    related_docs = []
                    if self.vector_store:
                        try:
                            search_results = self.vector_store.collection.query(
                                query_texts=[content_preview[:500]],
                                n_results=5,
                                include=['documents', 'metadatas', 'distances']
                            )

                            if search_results and search_results['documents']:
                                for i, doc_content in enumerate(search_results['documents'][0]):
                                    metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                                    distance = search_results['distances'][0][i] if search_results['distances'] else 0

                                    if distance < 1.0 and metadata.get('document_id') != doc_id:
                                        related_docs.append({
                                            'title': metadata.get('title', 'Unknown'),
                                            'document_id': metadata.get('document_id'),
                                            'content_snippet': doc_content[:500],
                                            'relevance_score': round(1.0 - distance, 2)
                                        })
                        except Exception as e:
                            logger.warning(f"Failed to query related docs for re-analysis: {e}")

                    # Re-run integrity analysis with full context
                    integrity_analysis = self.llm_client.analyze_document_integrity(
                        document_info={
                            'title': doc_title,
                            'document_type': 'document'
                        },
                        content_preview=content_preview,
                        related_docs=related_docs
                    )

                    # Update tags if needed
                    if integrity_analysis.get('has_issues') and integrity_analysis.get('findings'):
                        tags_to_add = []
                        for finding in integrity_analysis['findings']:
                            tag_name = f"issue:{finding['issue_type']}"
                            if tag_name not in tags_to_add:
                                tags_to_add.append(tag_name)

                        if tags_to_add:
                            self.paperless.update_document_tags(doc_id, tags_to_add, add_only=True)

                    reanalyzed_count += 1

                    if reanalyzed_count % 10 == 0:
                        logger.info(f"  ↳ Re-analyzed {reanalyzed_count}/{len(all_docs)} documents...")

                except Exception as e:
                    logger.warning(f"Failed to re-analyze document {doc_id}: {e}")
                    continue

            logger.info(f"✅ Project re-analysis complete: {reanalyzed_count} documents analyzed with full context")

        except Exception as e:
            logger.error(f"Failed to re-analyze project '{project_slug}': {e}", exc_info=True)

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
                port=web_port,
                project_manager=self.project_manager,  # v1.5.0
                llm_client=self.llm_client,  # v1.5.0
                smart_uploader=self.smart_uploader,  # v1.5.0
                document_analyzer=self  # v1.5.0 - for re-analysis
            )

        # Health check
        if not self.paperless.health_check():
            logger.error("Paperless API health check failed, exiting")
            sys.exit(1)

        # v2.0.5: On startup, automatically queue any Paperless docs not yet in processed_documents.
        # Runs once in a background thread 30 s after boot so the web UI is up first.
        def _auto_fill_gap():
            import time as _time
            _time.sleep(30)
            try:
                from analyzer.db import get_analyzed_doc_ids
                analyzed_ids = get_analyzed_doc_ids(project_slug=self.config.get('project_slug', 'default'))
                all_docs = []
                page = 1
                while True:
                    resp = self.paperless.get_documents(ordering='-modified', page_size=100, page=page)
                    all_docs.extend(resp.get('results', []))
                    if not resp.get('next'):
                        break
                    page += 1
                missing = [d for d in all_docs if d['id'] not in analyzed_ids]
                if not missing:
                    logger.info("Startup gap-fill: all documents already analyzed")
                    return
                logger.info(f"Startup gap-fill: {len(missing)} unanalyzed documents found — queuing now")
                ok = 0
                for d in missing:
                    try:
                        full_doc = self.paperless.get_document(d['id'])
                        self.analyze_document(full_doc)
                        ok += 1
                        if ok % 10 == 0:
                            logger.info(f"  ↳ Gap-fill progress: {ok}/{len(missing)} done")
                    except Exception as _e:
                        logger.warning(f"Gap-fill: failed doc {d['id']}: {_e}")
                logger.info(f"Startup gap-fill complete: {ok}/{len(missing)} documents analyzed")
            except Exception as _e:
                logger.error(f"Startup gap-fill failed: {_e}", exc_info=True)

        import threading as _threading
        _threading.Thread(target=_auto_fill_gap, daemon=True).start()

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

                # v1.5.0: Track document processing time for re-analysis trigger
                import time
                project_slug = self.config.get('project_slug', 'default')
                self.last_document_time[project_slug] = time.time()
                self.documents_processed_this_cycle = processed_count

                # Exit reprocess mode when a full pass had no skips (all docs freshly processed)
                if reprocess_mode and skipped_count == 0:
                    with self.state_manager.lock:
                        self.state_manager.state.reprocess_all_mode = False
                        self.state_manager.state.last_seen_modified = None
                        self.state_manager.state.last_seen_ids = set()
                        self.state_manager._save_state()
                    logger.info("Reprocess all mode complete - returning to incremental mode")
            else:
                if reprocess_mode:
                    # All documents already seen in previous passes — exit reprocess mode
                    with self.state_manager.lock:
                        self.state_manager.state.reprocess_all_mode = False
                        self.state_manager.state.last_seen_modified = None
                        self.state_manager.state.last_seen_ids = set()
                        self.state_manager._save_state()
                    logger.info("Reprocess all mode complete - returning to incremental mode")
                else:
                    logger.info("No new documents to process")

            # v1.5.0: Check if we should trigger automatic re-analysis
            self.check_and_trigger_reanalysis()

            # v2.0.4: Periodic stale embedding check (every 10 idle polls, skip when busy or reprocessing)
            # Only runs when there are no new docs to process, to avoid concurrent OpenAI calls.
            if not reprocess_mode and processed_count == 0:
                self._stale_check_counter += 1
                if self._stale_check_counter % 10 == 1:  # 1st, 11th, 21st… idle poll
                    import threading
                    threading.Thread(target=self.check_stale_embeddings, daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to poll documents: {e}", exc_info=True)

    def check_stale_embeddings(self) -> int:
        """
        v2.0.4: Detect and re-analyze documents whose Chroma embeddings are stale.

        An embedding is stale when:
          - Its stored 'paperless_modified' in Chroma is older than the current Paperless
            modified timestamp (OCR completed after the document was first embedded), OR
          - No 'paperless_modified' is stored (embedded before v2.0.4) AND the Chroma
            document text is suspiciously short (< 200 chars — indicates empty OCR at
            embed time).

        To avoid flooding the Paperless API, only checks documents modified within the
        past 7 days (or with short content), and caps at 50 re-analyses per call.

        Returns: number of documents re-analyzed.
        """
        if not self.vector_store or not self.vector_store.enabled:
            return 0

        try:
            from datetime import timezone, timedelta
            all_data = self.vector_store.collection.get(include=['metadatas', 'documents'])
            if not all_data['ids']:
                return 0

            now = datetime.now(timezone.utc)
            recent_cutoff = now - timedelta(days=7)

            candidates = []  # list of (doc_id, stored_modified)
            for i, chroma_id in enumerate(all_data['ids']):
                meta = all_data['metadatas'][i]
                stored_modified = meta.get('paperless_modified', '')
                doc_text = all_data['documents'][i] if all_data.get('documents') else ''
                doc_id = int(chroma_id)

                if not stored_modified:
                    # Embedded before v2.0.4 — only check if content looks empty/stub
                    if len(doc_text.strip()) < 200:
                        candidates.append((doc_id, ''))
                else:
                    # Embedded after v2.0.4 — only check recently modified docs
                    try:
                        stored_dt = datetime.fromisoformat(stored_modified.replace('Z', '+00:00'))
                        if stored_dt.tzinfo is None:
                            stored_dt = stored_dt.replace(tzinfo=timezone.utc)
                        if stored_dt >= recent_cutoff:
                            candidates.append((doc_id, stored_modified))
                    except Exception:
                        if len(doc_text.strip()) < 200:
                            candidates.append((doc_id, stored_modified))

            if not candidates:
                logger.info("Stale embedding check: no candidates found")
                return 0

            # Cap to avoid Paperless API overload
            candidates = candidates[:50]
            logger.info(f"Stale embedding check: verifying {len(candidates)} candidate documents")

            reanalyzed = 0
            for doc_id, stored_modified in candidates:
                try:
                    current_doc = self.paperless.get_document(doc_id)
                    current_modified = current_doc.get('modified', '')
                    current_content_len = len(current_doc.get('content', '').strip())

                    is_stale = False
                    if stored_modified and current_modified > stored_modified:
                        logger.info(f"Stale doc {doc_id}: modified {stored_modified} → {current_modified}")
                        is_stale = True
                    elif not stored_modified and current_content_len >= 200:
                        logger.info(f"Stale doc {doc_id}: was empty OCR, now {current_content_len} chars")
                        is_stale = True

                    if is_stale:
                        logger.info(f"Re-analyzing stale document {doc_id}: {current_doc.get('title', '?')}")
                        self.analyze_document(current_doc)
                        reanalyzed += 1

                except Exception as e:
                    logger.warning(f"Stale check failed for doc {doc_id}: {e}")

            logger.info(f"Stale embedding check complete: {reanalyzed}/{len(candidates)} documents re-analyzed")
            return reanalyzed

        except Exception as e:
            logger.error(f"check_stale_embeddings failed: {e}", exc_info=True)
            return 0

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

            # Determine which project this document belongs to (from project: tag)
            doc_project_slug = self.config.get('project_slug', 'default')
            for _t in existing_tags:
                if _t.startswith('project:'):
                    doc_project_slug = _t.split('project:', 1)[1]
                    break

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

            # Generate document summaries and extract rich metadata for UI & RAG optimization
            doc_summary = {'brief': '', 'full': ''}
            rich_metadata = {}
            if self.llm_enabled and self.llm_client:
                try:
                    content_preview = document.get('content', '')[:1000]  # First 1000 chars for better extraction

                    # Extract rich metadata (single LLM call gets everything)
                    rich_metadata = self.llm_client.extract_rich_metadata(
                        document_info={
                            'title': doc_title,
                            'document_type': profile.profile_id if profile else 'financial document'
                        },
                        content_preview=content_preview
                    )

                    # Use the one-line summary from rich metadata as brief summary
                    doc_summary['brief'] = rich_metadata.get('one_line_summary', f"Financial document: {doc_title}")

                    # Build full summary (5+ sentences) from rich metadata fields — no extra LLM call
                    _cls = rich_metadata.get('classification', {})
                    _ca = rich_metadata.get('content_analysis', {})
                    _fin = rich_metadata.get('financial_summary', {})
                    _tmp = rich_metadata.get('temporal', {})
                    _ent = rich_metadata.get('entities', {})
                    _act = rich_metadata.get('actionable_intelligence', {})
                    _rel = rich_metadata.get('relationships', {})
                    _parts = []

                    def _s(text):
                        """Ensure sentence ends with a period."""
                        t = str(text).strip()
                        return t if t.endswith('.') else t + '.'

                    # Sentence 1: document type + purpose
                    _type_str = _cls.get('sub_type') or _cls.get('primary_category', 'document')
                    _purpose = _ca.get('purpose', '') or _cls.get('industry_context', '')
                    if _purpose:
                        _parts.append(_s(f"This {_type_str} {_purpose}"))
                    else:
                        _parts.append(_s(f"This is a {_type_str}"))

                    # Sentence 2: temporal context — prefer explicit period, fall back to time_context
                    _p_start = _tmp.get('period_start', '')
                    _p_end = _tmp.get('period_end', '')
                    _doc_date = _tmp.get('document_date', '')
                    _time_ctx = _tmp.get('time_context', '')
                    if _p_start and _p_end:
                        _parts.append(f"It covers the period from {_p_start} to {_p_end}.")
                    elif _doc_date:
                        _parts.append(_s(f"Dated {_doc_date}" + (f"; {_time_ctx}" if _time_ctx else "")))
                    elif _time_ctx:
                        _parts.append(_s(_time_ctx))

                    # Sentence 3: financial detail — account_summary + key balances/figures
                    _acct_sum = _fin.get('account_summary', '')
                    _beg = _fin.get('beginning_balance', '')
                    _end = _fin.get('ending_balance', '')
                    _tot = _fin.get('total_amount', '')
                    _kfigs = [f for f in _fin.get('key_figures', []) if f][:3]
                    if _acct_sum:
                        _extra = ''
                        if _beg and _end:
                            _extra = f" Opening balance: {_beg}; closing balance: {_end}."
                        elif _tot:
                            _extra = f" Total amount: {_tot}."
                        _parts.append(_s(_acct_sum) + _extra)
                    elif _beg and _end:
                        _parts.append(f"The account opened at {_beg} and closed at {_end}.")
                    elif _tot:
                        _parts.append(f"The total amount involved is {_tot}.")
                    elif _kfigs:
                        _parts.append(f"Key financial figures: {'; '.join(_kfigs)}.")

                    # Sentence 4: entities — people, organizations, locations
                    _people = [p for p in _ent.get('people', []) if p][:3]
                    _orgs = [o for o in _ent.get('organizations', []) if o][:3]
                    _locs = [l for l in _ent.get('locations', []) if l][:2]
                    _ids = [i for i in _ent.get('identifiers', []) if i][:2]
                    _entity_parts = []
                    if _people:
                        _entity_parts.append(f"Parties involved: {', '.join(_people)}")
                    if _orgs:
                        _entity_parts.append(f"{'Organizations' if _people else 'Parties'}: {', '.join(_orgs)}")
                    if _locs:
                        _entity_parts.append(f"Location: {', '.join(_locs)}")
                    if _ids:
                        _entity_parts.append(f"Reference: {', '.join(_ids)}")
                    if _entity_parts:
                        _parts.append(_s('; '.join(_entity_parts)))

                    # Sentence 5: main topics / key content areas
                    _topics = [t for t in _ca.get('main_topics', []) if t][:5]
                    if _topics:
                        _parts.append(f"Key topics covered: {', '.join(_topics)}.")

                    # Sentence 6: red flags and action items
                    _flags = [f for f in _act.get('red_flags', []) if f][:2]
                    _actions = [a for a in _act.get('action_items', []) if a][:3]
                    _urgent = [d for d in _act.get('deadlines_urgent', []) if d][:2]
                    _followup = [f for f in _act.get('follow_up', []) if f][:2]
                    if _flags:
                        _parts.append(_s(f"Concerns noted: {'; '.join(_flags)}"))
                    if _actions:
                        _parts.append(_s(f"Actions required: {'; '.join(_actions)}"))
                    elif _urgent:
                        _parts.append(_s(f"Urgent deadlines: {'; '.join(_urgent)}"))
                    elif _followup:
                        _parts.append(_s(f"Follow-up needed: {'; '.join(_followup)}"))

                    # Sentence 7: relationships / series context
                    _rel_ctx = _rel.get('context', '')
                    _ref_docs = [d for d in _rel.get('references_documents', []) if d][:2]
                    _rel_parties = [p for p in _rel.get('related_parties', []) if p][:2]
                    if _rel_ctx and _rel_ctx.lower() not in ('', 'none', 'n/a', 'null'):
                        _parts.append(_s(_rel_ctx))
                    elif _ref_docs:
                        _parts.append(_s(f"References related documents: {', '.join(_ref_docs)}"))
                    elif _rel_parties:
                        _parts.append(_s(f"Connected parties: {', '.join(_rel_parties)}"))

                    # Sentence 8: importance / sentiment context (only if still short)
                    _imp = _ca.get('importance_level', '')
                    _sent = _ca.get('sentiment', '')
                    if len(_parts) < 5 and (_imp or _sent):
                        _ctx_parts = []
                        if _imp and _imp not in ('', 'null'):
                            _ctx_parts.append(f"Importance: {_imp}")
                        if _sent and _sent not in ('', 'null', 'neutral'):
                            _ctx_parts.append(f"tone is {_sent}")
                        if _ctx_parts:
                            _parts.append(_s(f"Document assessment — {'; '.join(_ctx_parts)}"))

                    # Pad to minimum 5 sentences using Q&A pairs as fallback
                    _qa = rich_metadata.get('qa_pairs', [])
                    for _qa_item in _qa:
                        if len(_parts) >= 5:
                            break
                        _q = _qa_item.get('question', '')
                        _a = _qa_item.get('answer', '')
                        if _a and _a.lower() not in ('none', 'n/a', 'null', 'not applicable', 'unknown'):
                            _parts.append(_s(f"{_q.rstrip('?')}: {_a}"))

                    # Final fallback: add document title reference
                    if len(_parts) < 5:
                        _industry = _cls.get('industry_context', '')
                        if _industry and _industry.lower() not in ('', 'null', 'n/a'):
                            _parts.append(_s(f"Industry context: {_industry}"))
                    if len(_parts) < 5:
                        _fin_figs = [f for f in _ent.get('financial_figures', []) if f][:2]
                        if _fin_figs:
                            _parts.append(_s(f"Financial figures mentioned: {'; '.join(_fin_figs)}"))
                    if len(_parts) < 5:
                        _parts.append(f"Analyzed from document '{doc_title}'.")

                    doc_summary['full'] = ' '.join(_parts[:8])

                    logger.info(f"Extracted metadata: {len(rich_metadata.get('keywords', []))} keywords, {len(rich_metadata.get('qa_pairs', []))} Q&A pairs")
                except Exception as e:
                    logger.warning(f"Failed to extract metadata for doc {doc_id}: {e}")

            # Analyze document integrity (conflicts, errors, quality issues)
            integrity_analysis = {}
            enhanced_tags = []  # Tags with evidence
            if self.llm_enabled and self.llm_client:
                try:
                    content_preview = document.get('content', '')[:1500]  # Longer preview for integrity check

                    # v1.5.0: Query vector store for related documents in same project
                    related_docs = []
                    if self.vector_store and content_preview:
                        try:
                            # Search for similar documents in the same project/collection
                            search_results = self.vector_store.collection.query(
                                query_texts=[content_preview[:500]],  # Use first 500 chars for similarity search
                                n_results=5,  # Get top 5 most relevant documents
                                include=['documents', 'metadatas', 'distances']
                            )

                            # Build related docs list with context
                            if search_results and search_results['documents']:
                                for i, doc_content in enumerate(search_results['documents'][0]):
                                    metadata = search_results['metadatas'][0][i] if search_results['metadatas'] else {}
                                    distance = search_results['distances'][0][i] if search_results['distances'] else 0

                                    # Only include if reasonably similar (distance < 1.0) and not the same document
                                    if distance < 1.0 and metadata.get('document_id') != doc_id:
                                        related_docs.append({
                                            'title': metadata.get('title', 'Unknown'),
                                            'document_id': metadata.get('document_id'),
                                            'content_snippet': doc_content[:500],  # First 500 chars
                                            'relevance_score': round(1.0 - distance, 2)  # Convert distance to similarity
                                        })

                            if related_docs:
                                logger.info(f"Found {len(related_docs)} related documents in project for context")
                        except Exception as e:
                            logger.warning(f"Failed to query related documents: {e}")

                    integrity_analysis = self.llm_client.analyze_document_integrity(
                        document_info={
                            'title': doc_title,
                            'document_type': profile.profile_id if profile else 'financial document'
                        },
                        content_preview=content_preview,
                        related_docs=related_docs  # v1.5.0: Pass project context
                    )

                    # Generate enhanced tags with evidence for each finding
                    if integrity_analysis.get('has_issues') and integrity_analysis.get('findings'):
                        for finding in integrity_analysis['findings']:
                            # Create tag with embedded evidence
                            tag_name = f"issue:{finding['issue_type']}"
                            tag_evidence = {
                                'tag': tag_name,
                                'severity': finding['severity'],
                                'category': finding['category'],
                                'description': finding['description'],
                                'evidence': finding['evidence'],
                                'impact': finding['impact'],
                                'suggested_action': finding.get('suggested_action', ''),
                                'confidence': finding.get('confidence', 'medium')
                            }
                            enhanced_tags.append(tag_evidence)

                            # Add to tags_to_add for Paperless
                            if tag_name not in tags_to_add:
                                tags_to_add.append(tag_name)

                        logger.info(f"Integrity check: {integrity_analysis['issue_count']} issues ({integrity_analysis.get('critical_count', 0)} critical)")
                except Exception as e:
                    logger.warning(f"Failed integrity analysis for doc {doc_id}: {e}")

            # Update UI stats with enhanced tag evidence
            update_ui_stats({
                'doc_id': doc_id,
                'title': doc_title,
                'timestamp': datetime.utcnow().isoformat(),
                'project_slug': doc_project_slug,
                'profile_matched': profile.profile_id if profile else None,
                'anomalies_found': deterministic_results.get('anomalies_found', []),
                'risk_score': risk_score,
                'tags_added': tags_to_add,
                'brief_summary': doc_summary.get('brief', ''),
                'full_summary': doc_summary.get('full', ''),
                'enhanced_tags': enhanced_tags,  # Tags with evidence and context
                'integrity_summary': integrity_analysis.get('summary', ''),
                'issue_count': integrity_analysis.get('issue_count', 0),
                'critical_count': integrity_analysis.get('critical_count', 0)
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

                    # Add AI-generated summaries for better searchability
                    if doc_summary.get('brief'):
                        content_parts.append(f"\nDocument Summary: {doc_summary['brief']}")
                    if doc_summary.get('full') and doc_summary['full'] != doc_summary.get('brief'):
                        content_parts.append(f"\nDetailed Summary: {doc_summary['full']}")

                    # Add rich metadata for optimal RAG (pre-computed structured data)
                    if rich_metadata:
                        # Add classification
                        classification = rich_metadata.get('classification', {})
                        if classification:
                            content_parts.append(f"\nDocument Type: {classification.get('sub_type', '')}, Category: {classification.get('primary_category', '')}")

                        # Add entities for searchability
                        entities = rich_metadata.get('entities', {})
                        if entities.get('people'):
                            content_parts.append(f"\nPeople: {', '.join(entities['people'][:10])}")
                        if entities.get('organizations'):
                            content_parts.append(f"\nOrganizations: {', '.join(entities['organizations'][:10])}")
                        if entities.get('identifiers'):
                            content_parts.append(f"\nIdentifiers: {', '.join(entities['identifiers'][:5])}")

                        # Add temporal info
                        temporal = rich_metadata.get('temporal', {})
                        if temporal.get('document_date'):
                            content_parts.append(f"\nDocument Date: {temporal['document_date']}")
                        if temporal.get('deadlines'):
                            content_parts.append(f"\nDeadlines: {', '.join(temporal['deadlines'][:3])}")

                        # Add financial summary
                        financial = rich_metadata.get('financial_summary', {})
                        if financial.get('account_summary'):
                            content_parts.append(f"\nFinancial Summary: {financial['account_summary']}")

                        # Add keywords and topics for search optimization
                        content_analysis = rich_metadata.get('content_analysis', {})
                        if content_analysis.get('keywords'):
                            content_parts.append(f"\nKeywords: {', '.join(content_analysis['keywords'])}")
                        if content_analysis.get('main_topics'):
                            content_parts.append(f"\nMain Topics: {', '.join(content_analysis['main_topics'])}")

                        # Add pre-generated Q&A pairs for fast RAG responses
                        qa_pairs = rich_metadata.get('qa_pairs', [])
                        if qa_pairs:
                            qa_text = "\n".join([f"Q: {qa['question']} A: {qa['answer']}" for qa in qa_pairs[:5]])
                            content_parts.append(f"\nCommon Questions:\n{qa_text}")

                        # Add actionable intelligence
                        actionable = rich_metadata.get('actionable_intelligence', {})
                        if actionable.get('action_items'):
                            content_parts.append(f"\nAction Items: {', '.join(actionable['action_items'][:3])}")
                        if actionable.get('red_flags'):
                            content_parts.append(f"\nRed Flags: {', '.join(actionable['red_flags'][:3])}")

                    # Add integrity analysis findings (critical for legal review)
                    if integrity_analysis.get('has_issues') and integrity_analysis.get('findings'):
                        findings_text = []
                        for finding in integrity_analysis['findings'][:5]:  # Top 5 issues
                            issue_desc = f"{finding['severity'].upper()}: {finding['description']}"
                            if finding.get('evidence', {}).get('quotes'):
                                issue_desc += f" | Evidence: {finding['evidence']['quotes'][0][:100]}"
                            findings_text.append(issue_desc)
                        content_parts.append(f"\nDocument Integrity Issues:\n" + "\n".join(findings_text))
                        content_parts.append(f"\nIntegrity Summary: {integrity_analysis.get('summary', '')}")

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

                    # Prepare comprehensive metadata for vector store
                    vector_metadata = {
                        'risk_score': risk_score,
                        'anomalies': deterministic_results.get('anomalies_found', []),
                        'timestamp': datetime.utcnow().isoformat(),
                        'paperless_modified': document.get('modified', ''),  # v2.0.4: track Paperless modified time for stale detection
                        'document_type': document_type,
                        'brief_summary': doc_summary.get('brief', ''),
                        'full_summary': doc_summary.get('full', '')
                    }

                    # Add rich metadata for fast retrieval (no need to re-analyze)
                    if rich_metadata:
                        vector_metadata.update({
                            'keywords': rich_metadata.get('content_analysis', {}).get('keywords', []),
                            'main_topics': rich_metadata.get('content_analysis', {}).get('main_topics', []),
                            'importance_level': rich_metadata.get('content_analysis', {}).get('importance_level', 'medium'),
                            'entities_people': rich_metadata.get('entities', {}).get('people', [])[:10],
                            'entities_orgs': rich_metadata.get('entities', {}).get('organizations', [])[:10],
                            'document_date': rich_metadata.get('temporal', {}).get('document_date'),
                            'has_deadlines': len(rich_metadata.get('temporal', {}).get('deadlines', [])) > 0,
                            'action_items_count': len(rich_metadata.get('actionable_intelligence', {}).get('action_items', [])),
                            'search_tags': rich_metadata.get('search_tags', []),
                            'classification': rich_metadata.get('classification', {}).get('sub_type', ''),
                            'purpose': rich_metadata.get('content_analysis', {}).get('purpose', '')
                        })

                    # Add integrity analysis for legal review (critical quality metadata)
                    if integrity_analysis:
                        vector_metadata.update({
                            'has_integrity_issues': integrity_analysis.get('has_issues', False),
                            'issue_count': integrity_analysis.get('issue_count', 0),
                            'critical_issues': integrity_analysis.get('critical_count', 0),
                            'integrity_summary': integrity_analysis.get('summary', ''),
                            'issue_categories': list(set([f['category'] for f in integrity_analysis.get('findings', [])])),
                            'max_severity': self._get_max_severity(integrity_analysis.get('findings', []))
                        })

                    # Route to the correct project collection (doc_project_slug set above)
                    if doc_project_slug == self.config.get('project_slug', 'default'):
                        embed_store = self.vector_store
                    else:
                        embed_store = VectorStore(project_slug=doc_project_slug)

                    # Embed and store with rich metadata
                    embed_store.embed_document(
                        document_id=doc_id,
                        title=doc_title,
                        content=content_text,
                        metadata=vector_metadata
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

    def _get_max_severity(self, findings: List[Dict]) -> str:
        """
        Determine highest severity level from integrity findings.

        Args:
            findings: List of integrity finding dictionaries

        Returns:
            String representing max severity: 'critical', 'high', 'medium', 'low', or 'none'
        """
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        if not findings:
            return 'none'
        max_finding = max(findings, key=lambda f: severity_order.get(f.get('severity', 'low'), 1))
        return max_finding.get('severity', 'low')

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
