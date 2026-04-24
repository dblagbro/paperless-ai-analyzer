"""Private helpers used by court routes: credential gate, connector builder,
post-import analysis, import job worker."""
import logging

from flask import session
from flask_login import current_user

from analyzer.app import _get_project_client, safe_json_body

logger = logging.getLogger(__name__)

def _court_gate():
    """Always returns (True, None) — Court Import is always enabled."""
    return True, None


def _get_current_project_slug() -> str:
    """Return the active project slug for the current user session."""
    project_slug = request.args.get('project') or request.json.get('project') \
        if request.is_json else request.form.get('project', '')
    if not project_slug:
        project_slug = getattr(current_user, 'active_project', 'default') or 'default'
    return project_slug


def _build_court_connector(court_system: str, project_slug: str):
    """
    Build the appropriate court connector for the given court system.
    Loads credentials from the DB and decrypts the password.

    Returns connector or raises RuntimeError.
    """
    from analyzer.court_db import load_credentials
    from analyzer.court_connectors.credential_store import decrypt_password

    creds = load_credentials(project_slug, court_system)
    password = ''
    if creds and creds.get('password_encrypted'):
        blob = creds['password_encrypted']
        if isinstance(blob, str):
            blob = blob.encode('latin-1')
        password = decrypt_password(blob) or ''

    if court_system == 'federal':
        from analyzer.court_connectors.federal import FederalConnector
        return FederalConnector(project_slug, creds or {}, pacer_password=password)
    elif court_system == 'nyscef':
        from analyzer.court_connectors.nyscef import NYSCEFConnector
        return NYSCEFConnector(project_slug, creds or {}, password=password)
    else:
        raise RuntimeError(f"Unknown court system: {court_system}")


def _post_import_analyze(job_id: str, project_slug: str,
                         task_ids: list, doc_ids: list):
    """
    Background daemon: resolve Paperless task UUIDs → doc IDs, then run AI analysis.
    Called after _run_court_import upload loop finishes.
    """
    from flask import current_app
    from analyzer.court_db import update_court_doc_task_resolved, update_import_job
    import datetime as _dt

    def _log(msg: str):
        ts = _dt.datetime.utcnow().strftime('%H:%M:%S')
        update_import_job(job_id, log_append=[f"[{ts}] [AI] {msg}"])

    try:
        resolved_doc_ids = list(doc_ids)  # start with any already-known doc IDs

        # Resolve task UUIDs → Paperless doc IDs (waits for OCR to complete)
        _pc = _get_project_client(project_slug)
        for task_id in task_ids:
            try:
                _log(f"Waiting for OCR on task {task_id[:8]}…")
                doc_id = _pc.resolve_task_to_doc_id(task_id, timeout=180)
                if doc_id:
                    update_court_doc_task_resolved(task_id, doc_id)
                    resolved_doc_ids.append(doc_id)
                    _log(f"Task {task_id[:8]} → doc {doc_id} — OCR complete.")
                else:
                    _log(f"Task {task_id[:8]} — OCR timed out or failed, skipping AI.")
            except Exception as e:
                _log(f"Task {task_id[:8]} resolve error: {e}")

        if not resolved_doc_ids:
            _log("No documents to analyze.")
            return

        # Run AI analysis on all resolved docs
        _log(f"Starting AI analysis on {len(resolved_doc_ids)} document(s)…")
        ok = 0
        for doc_id in resolved_doc_ids:
            try:
                full_doc = _pc.get_document(doc_id)
                if not full_doc.get('content', '').strip():
                    _log(f"Doc {doc_id}: no content yet, skipping.")
                    continue
                current_app.document_analyzer.analyze_document(full_doc)
                ok += 1
            except Exception as e:
                _log(f"Doc {doc_id}: analysis error — {e}")

        _log(f"AI analysis complete — {ok}/{len(resolved_doc_ids)} succeeded.")
        update_import_job(job_id, log_append=[])  # flush

    except Exception as e:
        logger.error(f"_post_import_analyze failed for job {job_id}: {e}", exc_info=True)
        try:
            update_import_job(job_id, log_append=[f"[AI ERROR] {str(e)[:200]}"])
        except Exception:
            pass


def _analyze_missing_for_project(project_slug: str) -> int:
    """
    Scan all Paperless docs tagged project:<slug> that are NOT yet in ChromaDB
    and run AI analysis on each.  Returns the number of docs queued.
    """
    from flask import current_app
    try:
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=project_slug)
        if not vs.enabled:
            logger.warning(f"_analyze_missing_for_project: vector store disabled for {project_slug}")
            return 0

        # IDs already in Chroma (as strings)
        existing = set(vs.collection.get(ids=None)['ids'])

        # Page through all Paperless docs for this project using tags__id__all.
        # tags__name is NOT reliable — some Paperless versions ignore it and return
        # all documents regardless of the filter value.
        _pc = _get_project_client(project_slug)
        missing = []
        page = 1
        while True:
            resp = _pc.get_documents_by_project(
                project_slug, ordering='-modified', page_size=100, page=page
            )
            for doc in resp.get('results', []):
                if str(doc['id']) not in existing:
                    missing.append(doc['id'])
            if not resp.get('next'):
                break
            page += 1

        logger.info(f"_analyze_missing_for_project({project_slug}): {len(missing)} missing from Chroma")

        ok = 0
        for doc_id in missing:
            try:
                full_doc = _pc.get_document(doc_id)
                if not full_doc.get('content', '').strip():
                    continue
                current_app.document_analyzer.analyze_document(full_doc)
                ok += 1
            except Exception as e:
                logger.warning(f"_analyze_missing_for_project: doc {doc_id} failed: {e}")

        logger.info(f"_analyze_missing_for_project({project_slug}): {ok}/{len(missing)} analyzed")
        return ok

    except Exception as e:
        logger.error(f"_analyze_missing_for_project({project_slug}) error: {e}", exc_info=True)
        return 0


def _run_court_import(job_id: str, project_slug: str, court_system: str,
                      case_id: str, cancel_event=None):
    """
    Background worker: download all docket entries for a case and upload to Paperless.
    Called from CourtImportJobManager in a daemon thread.
    """
    from flask import current_app
    from analyzer.court_db import (
        update_import_job, log_court_doc, load_credentials
    )
    from analyzer.court_connectors.credential_store import decrypt_password
    from analyzer.court_connectors.deduplicator import CourtDeduplicator
    import datetime as _dt

    def _log(msg: str):
        ts = _dt.datetime.utcnow().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        logger.info(f"Court import {job_id[:8]}: {msg}")
        update_import_job(job_id, log_append=[line])

    try:
        update_import_job(job_id, status='running',
                          started_at=_dt.datetime.utcnow().isoformat())
        _log("Initializing connector…")

        creds = load_credentials(project_slug, court_system) or {}
        password = ''
        if creds.get('password_encrypted'):
            blob = creds['password_encrypted']
            if isinstance(blob, str):
                blob = blob.encode('latin-1')
            password = decrypt_password(blob) or ''

        if court_system == 'federal':
            from analyzer.court_connectors.federal import FederalConnector
            connector = FederalConnector(project_slug, creds, pacer_password=password)
        elif court_system == 'nyscef':
            from analyzer.court_connectors.nyscef import NYSCEFConnector
            connector = NYSCEFConnector(project_slug, creds, password=password)
        else:
            raise RuntimeError(f"Unknown court system: {court_system}")

        _log("Authenticating…")
        connector.authenticate()

        _log(f"Fetching docket for {case_id}…")
        docket = connector.get_docket(case_id)
        total = len(docket)
        update_import_job(job_id, total_docs=total)
        _log(f"Found {total} docket entries.")

        paperless_client = getattr(current_app, 'paperless_client', None)
        dedup = CourtDeduplicator(project_slug, paperless_client)

        # Get or create court tag
        court_tag_id = None
        project_tag_id = None
        try:
            if paperless_client:
                court_tag_id = paperless_client.get_or_create_tag(
                    f"court:{court_system}"
                )
                project_tag_id = paperless_client.get_or_create_tag(
                    f"project:{project_slug}"
                )
        except Exception as e:
            _log(f"Warning: could not create tags: {e}")

        imported = skipped = failed = 0
        collected_task_ids: list = []
        collected_doc_ids: list = []

        for entry in docket:
            if cancel_event and cancel_event.is_set():
                _log("Cancelled by user.")
                update_import_job(job_id, status='cancelled',
                                  imported_docs=imported, skipped_docs=skipped,
                                  failed_docs=failed,
                                  completed_at=_dt.datetime.utcnow().isoformat())
                return

            seq_label = f"seq {entry.seq}"

            # Tier 1: URL dedup
            skip, reason = dedup.check_url(entry.source_url)
            if skip:
                _log(f"{seq_label}: skipped (url_match)")
                log_court_doc(job_id, project_slug, court_system, case_id,
                              status='skipped', doc_sequence=entry.seq,
                              source_url=entry.source_url, skip_reason='url_match')
                skipped += 1
                update_import_job(job_id, imported_docs=imported,
                                  skipped_docs=skipped, failed_docs=failed)
                continue

            # Download
            _log(f"{seq_label}: downloading ({entry.source or 'unknown'})…")
            tmp_path = connector.download_document(entry)
            if not tmp_path:
                _log(f"{seq_label}: no source available, skipping")
                log_court_doc(job_id, project_slug, court_system, case_id,
                              status='failed', doc_sequence=entry.seq,
                              source_url=entry.source_url,
                              error_msg='No download source available')
                failed += 1
                update_import_job(job_id, imported_docs=imported,
                                  skipped_docs=skipped, failed_docs=failed)
                continue

            # Tier 2: hash dedup
            skip, reason, digest = dedup.check_hash(tmp_path)
            if skip:
                _log(f"{seq_label}: skipped (hash_match)")
                tmp_path.unlink(missing_ok=True)
                log_court_doc(job_id, project_slug, court_system, case_id,
                              status='skipped', doc_sequence=entry.seq,
                              source_url=entry.source_url, sha256_hash=digest,
                              skip_reason='hash_match')
                skipped += 1
                update_import_job(job_id, imported_docs=imported,
                                  skipped_docs=skipped, failed_docs=failed)
                continue

            # Build title
            title = f"{case_id} — {entry.seq}: {entry.title[:80]}"

            # Tier 3: title dedup
            skip, reason = dedup.check_title(title)
            if skip:
                _log(f"{seq_label}: skipped (title_match)")
                tmp_path.unlink(missing_ok=True)
                log_court_doc(job_id, project_slug, court_system, case_id,
                              status='skipped', doc_sequence=entry.seq,
                              source_url=entry.source_url, sha256_hash=digest,
                              filename=tmp_path.name, skip_reason='title_match')
                skipped += 1
                update_import_job(job_id, imported_docs=imported,
                                  skipped_docs=skipped, failed_docs=failed)
                continue

            # Upload to Paperless
            tag_ids = [t for t in [court_tag_id, project_tag_id] if t]
            # Convert PACER date (MM/DD/YYYY) to ISO format (YYYY-MM-DD) that
            # Paperless-ngx expects. Dates in other formats are passed through
            # unchanged; missing dates are sent as None.
            _raw_date = entry.date or None
            if _raw_date:
                try:
                    import datetime as _ddt
                    _raw_date = _ddt.datetime.strptime(_raw_date, '%m/%d/%Y').strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    pass  # already ISO or unparseable — leave as-is
            paperless_doc_id = None
            paperless_task_id = ''
            try:
                if paperless_client:
                    result = paperless_client.upload_document(
                        str(tmp_path),
                        title=title,
                        tags=tag_ids,
                        created=_raw_date,
                    )
                    if result and isinstance(result, dict):
                        if result.get('task_id'):
                            paperless_task_id = result['task_id']
                            collected_task_ids.append(paperless_task_id)
                        elif result.get('id'):
                            paperless_doc_id = result['id']
                            collected_doc_ids.append(paperless_doc_id)
                    elif paperless_client:
                        # upload_document returned None — Paperless rejected or
                        # errored; treat as upload failure so the doc is NOT
                        # marked 'imported' (which would block future re-imports
                        # via the URL-dedup check).
                        raise RuntimeError(
                            "Paperless upload returned no result (check Paperless logs)"
                        )
                _log(f"{seq_label}: uploaded as \"{title[:60]}\"")
                log_court_doc(job_id, project_slug, court_system, case_id,
                              status='imported', doc_sequence=entry.seq,
                              source_url=entry.source_url, sha256_hash=digest,
                              filename=tmp_path.name,
                              paperless_doc_id=paperless_doc_id,
                              paperless_task_id=paperless_task_id)
                imported += 1
            except Exception as e:
                _log(f"{seq_label}: upload failed — {e}")
                log_court_doc(job_id, project_slug, court_system, case_id,
                              status='failed', doc_sequence=entry.seq,
                              source_url=entry.source_url, sha256_hash=digest,
                              filename=tmp_path.name, error_msg=str(e)[:300])
                failed += 1
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

            update_import_job(job_id, imported_docs=imported,
                              skipped_docs=skipped, failed_docs=failed)

        _log(f"Complete — {imported} imported, {skipped} skipped, {failed} failed.")
        update_import_job(job_id, status='completed',
                          imported_docs=imported, skipped_docs=skipped,
                          failed_docs=failed,
                          completed_at=_dt.datetime.utcnow().isoformat())

        # Fire post-import AI analysis in a daemon thread
        if imported > 0 and hasattr(current_app, 'document_analyzer') and current_app.document_analyzer:
            import threading
            threading.Thread(
                target=_post_import_analyze,
                args=(job_id, project_slug, collected_task_ids, collected_doc_ids),
                daemon=True,
            ).start()
            _log("Background AI analysis started for imported documents.")

    except Exception as e:
        logger.error(f"Court import job {job_id} failed: {e}", exc_info=True)
        try:
            update_import_job(job_id, status='failed',
                              error_message=str(e)[:500],
                              completed_at=_dt.datetime.utcnow().isoformat(),
                              log_append=[f"[ERROR] {str(e)[:300]}"])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Credential routes
# ---------------------------------------------------------------------------

