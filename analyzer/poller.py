"""Polling-loop control for DocumentAnalyzer (v3.9.3 mixin split).

Extracted from analyzer/main.py during the v3.9.3 maintainability refactor.
Mixin — methods reference `self.*` state initialised in DocumentAnalyzer.__init__.
Do not instantiate directly.
"""
import os
import sys
import time
import signal
import logging
import threading
from threading import Thread
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone

from analyzer.paperless_client import PaperlessClient
from analyzer.state import StateManager
from analyzer.profile_loader import ProfileLoader
from analyzer.project_manager import ProjectManager
from analyzer.smart_upload import SmartUploader
from analyzer.vector_store import VectorStore
from analyzer.llm.llm_client import LLMClient
from analyzer.llm_usage_tracker import LLMUsageTracker
from analyzer.web_ui import start_web_server_thread, update_ui_stats

logger = logging.getLogger(__name__)


class PollerMixin:
    """Polling-loop control for DocumentAnalyzer (v3.9.3 mixin split)."""

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


    def start_project_pollers(self) -> None:
        """
        v3.6.0: Start dedicated polling threads for each project that has its
        own Paperless-ngx instance configured (paperless_url + token set).

        Projects without per-project config continue to be served by the
        default poll_and_analyze() loop.
        """
        try:
            projects = self.project_manager.list_projects(include_archived=False)
        except Exception as e:
            logger.warning(f"start_project_pollers: could not list projects: {e}")
            return

        poll_interval = self.config.get('poll_interval_seconds', 30)

        for proj in projects:
            slug = proj['slug']
            if slug == 'default':
                continue  # default project uses the main loop

            cfg = self.project_manager.get_paperless_config(slug)
            if not cfg.get('url') or not cfg.get('token'):
                continue  # no dedicated instance configured

            try:
                pc = PaperlessClient(base_url=cfg['url'], api_token=cfg['token'])
                state_mgr = StateManager(
                    state_dir=self.config.get('state_dir', '/app/data'),
                    project_slug=slug
                )
                import threading as _t
                t = _t.Thread(
                    target=_poll_project_loop,
                    args=(slug, pc, state_mgr, self, poll_interval),
                    daemon=True,
                    name=f"poller-{slug}"
                )
                t.start()
                logger.info(f"Started polling thread for project '{slug}' "
                            f"→ {cfg['url']}")
            except Exception as e:
                logger.error(f"Failed to start poller for project '{slug}': {e}")


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

        # v3.6.0: Start per-project polling threads for projects with dedicated instances
        self.start_project_pollers()

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
                # CI findings (v3.1+) are embedded with composite IDs like
                # `ci:<run>:<type>:<n>` that live alongside paperless doc IDs.
                # Skip them — they aren't Paperless documents.
                try:
                    doc_id = int(chroma_id)
                except (ValueError, TypeError):
                    continue

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



# --------------------------------------------------------------------------
# Module-level function: per-project polling daemon thread target
# --------------------------------------------------------------------------

def _poll_project_loop(slug: str, paperless_client, state_manager,
                       analyzer_ref, poll_interval: int) -> None:
    """
    v3.6.0: Daemon thread target for per-project Paperless polling.

    Mirrors poll_and_analyze() but uses a dedicated PaperlessClient for the
    project's own Paperless-ngx instance and the project's own StateManager.
    """
    logger.info(f"[{slug}] Per-project polling thread started (interval={poll_interval}s)")
    while True:
        try:
            state_stats = state_manager.get_stats()
            modified_after = state_stats.get('last_seen_modified')

            documents = []
            page = 1
            while True:
                response = paperless_client.get_documents(
                    ordering='-modified',
                    page_size=100,
                    page=page,
                    modified_after=modified_after
                )
                documents.extend(response.get('results', []))
                if not response.get('next'):
                    break
                page += 1

            new_docs = [d for d in documents
                        if state_manager.should_process_document(d['modified'], d['id'])]

            if new_docs:
                logger.info(f"[{slug}] {len(new_docs)} new document(s) to analyze")
                for doc in new_docs:
                    try:
                        full_doc = paperless_client.get_document(doc['id'])
                        analyzer_ref.analyze_document(full_doc)
                        state_manager.update_last_seen(doc['modified'], {doc['id']})
                    except Exception as _e:
                        logger.warning(f"[{slug}] Failed to analyze doc {doc['id']}: {_e}")

        except Exception as e:
            logger.error(f"[{slug}] Per-project poll error: {e}", exc_info=True)

        time.sleep(poll_interval)
