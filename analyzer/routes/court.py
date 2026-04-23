import logging
from flask import Blueprint, request, jsonify, session
from flask_login import login_required, current_user

from analyzer.app import advanced_required, _get_project_client, safe_json_body

logger = logging.getLogger(__name__)

bp = Blueprint('court', __name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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

@bp.route('/api/court/credentials', methods=['POST'])
@login_required
@advanced_required
def court_save_credentials():
    """Save (upsert) court credentials for the current project."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = safe_json_body()
    court_system = data.get('court_system', '')
    username = data.get('username', '').strip()
    password = data.get('password', '')
    extra_config = data.get('extra_config', {})
    project_slug = data.get('project_slug', '') or 'default'

    if court_system not in ('federal', 'nyscef'):
        return jsonify({'error': 'court_system must be "federal" or "nyscef"'}), 400
    # NYSCEF public-only access has no username — allow empty username in that case
    if not username and not (court_system == 'nyscef' and extra_config.get('public_only')):
        return jsonify({'error': 'username is required'}), 400

    try:
        from analyzer.court_connectors.credential_store import encrypt_password, is_cryptography_available
        from analyzer.court_db import save_credentials
        if not is_cryptography_available():
            return jsonify({'error': 'cryptography package not installed — cannot encrypt credentials'}), 500
        encrypted = encrypt_password(password) if password else b''
        save_credentials(project_slug, court_system, username, encrypted, extra_config)
        return jsonify({'ok': True, 'message': f'{court_system} credentials saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/credentials', methods=['GET'])
@login_required
def court_list_credentials():
    """List configured court systems for the current project (no passwords)."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')
    try:
        from analyzer.court_db import list_credentials
        creds = list_credentials(project_slug)
        return jsonify({'credentials': creds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/credentials/test', methods=['POST'])
@login_required
@advanced_required
def court_test_credentials():
    """Test court credentials and return account info."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = safe_json_body()
    court_system = data.get('court_system', '')
    username = data.get('username', '').strip()
    password = data.get('password', '')
    extra_config = data.get('extra_config', {})
    project_slug = data.get('project_slug', 'default')

    if court_system not in ('federal', 'nyscef'):
        return jsonify({'error': 'court_system must be "federal" or "nyscef"'}), 400

    try:
        # Build a temporary credential dict for testing (don't require DB save first)
        import json as _json
        temp_creds = {
            'username': username,
            'extra_config_json': _json.dumps(extra_config),
        }
        if court_system == 'federal':
            from analyzer.court_connectors.federal import FederalConnector
            connector = FederalConnector(project_slug, temp_creds, pacer_password=password)
        else:
            from analyzer.court_connectors.nyscef import NYSCEFConnector
            connector = NYSCEFConnector(project_slug, temp_creds, password=password)

        result = connector.test_connection()

        # Update last_tested_at in DB if credentials already exist
        try:
            from analyzer.court_db import update_credential_test
            update_credential_test(project_slug, court_system, result['ok'])
        except Exception:
            pass

        return jsonify(result)
    except Exception as e:
        return jsonify({'ok': False, 'account_info': '', 'error': str(e)}), 500


@bp.route('/api/court/credentials/<court_system>', methods=['DELETE'])
@login_required
@advanced_required
def court_delete_credentials(court_system):
    """Remove court credentials for the current project."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')
    try:
        from analyzer.court_db import delete_credentials
        deleted = delete_credentials(project_slug, court_system)
        return jsonify({'ok': deleted,
                        'message': f'{court_system} credentials removed' if deleted
                                   else 'No credentials found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Search / docket routes
# ---------------------------------------------------------------------------

@bp.route('/api/court/search', methods=['POST'])
@login_required
def court_search():
    """Search for cases by case number or party name."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = safe_json_body()
    court_system = data.get('court_system', 'federal')
    case_number = data.get('case_number', '').strip()
    party_name  = data.get('party_name', '').strip()
    court       = data.get('court', '').strip()
    project_slug = data.get('project_slug', 'default')

    if not case_number and not party_name:
        return jsonify({'error': 'Provide case_number or party_name'}), 400

    try:
        connector = _build_court_connector(court_system, project_slug)
        results = connector.search_cases(case_number=case_number,
                                         party_name=party_name,
                                         court=court)
        return jsonify({
            'results': [
                {
                    'case_id':    r.case_id,
                    'case_number': r.case_number,
                    'case_title': r.case_title,
                    'court':      r.court,
                    'filing_date': r.filing_date,
                    'source':     r.source,
                }
                for r in results
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/docket/<court_system>/<path:case_id>', methods=['GET'])
@login_required
def court_get_docket(court_system, case_id):
    """Return the full docket entry list for a case."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')

    try:
        connector = _build_court_connector(court_system, project_slug)
    except RuntimeError as e:
        # v3.9.4: unknown court_system → 400 with clear error (was 500)
        msg = str(e)
        if 'Unknown court system' in msg:
            return jsonify({
                'error': msg,
                'supported': ['federal', 'nyscef'],
            }), 400
        return jsonify({'error': msg}), 500
    try:
        docket = connector.get_docket(case_id)
        return jsonify({
            'case_id': case_id,
            'court_system': court_system,
            'total': len(docket),
            'entries': [
                {
                    'seq':        e.seq,
                    'title':      e.title,
                    'date':       e.date,
                    'source_url': e.source_url,
                    'source':     e.source,
                    'doc_type':   e.doc_type,
                }
                for e in docket
            ],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Import job routes
# ---------------------------------------------------------------------------

@bp.route('/api/projects/<slug>/analyze-missing', methods=['POST'])
@login_required
def api_project_analyze_missing(slug):
    """Trigger background AI analysis for all Paperless docs in <slug> not yet in ChromaDB."""
    from flask import current_app
    from threading import Thread
    if not hasattr(current_app, 'document_analyzer') or not current_app.document_analyzer:
        return jsonify({'error': 'Analyzer not running'}), 503
    Thread(target=_analyze_missing_for_project, args=(slug,), daemon=True).start()
    return jsonify({'success': True, 'message': f'Scanning project {slug} for unanalyzed docs'})


@bp.route('/api/court/import/start', methods=['POST'])
@login_required
def court_import_start():
    """Create and start a background import job."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = safe_json_body()
    court_system  = data.get('court_system', 'federal')
    case_id       = data.get('case_id', '').strip()
    case_number   = data.get('case_number', case_id).strip()
    case_title    = data.get('case_title', '').strip()
    project_slug  = data.get('project_slug', 'default')

    if not case_id:
        return jsonify({'error': 'case_id is required'}), 400

    import uuid
    job_id = str(uuid.uuid4())

    try:
        from analyzer.court_db import create_import_job
        create_import_job(job_id, project_slug, current_user.id,
                          court_system, case_number, case_title)

        from analyzer.court_connectors.import_job import get_job_manager
        jm = get_job_manager()
        started = jm.start_job(
            job_id, _run_court_import,
            project_slug=project_slug,
            court_system=court_system,
            case_id=case_id,
        )
        if not started:
            return jsonify({'error': 'Could not start job (already running?)'}), 409

        return jsonify({'job_id': job_id, 'status': 'queued'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/import/status/<job_id>', methods=['GET'])
@login_required
def court_import_status(job_id):
    """Poll import job progress and last N log lines."""
    ok, err = _court_gate()
    if not ok:
        return err
    try:
        from analyzer.court_db import get_import_job
        import json as _json
        job = get_import_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        log_lines = _json.loads(job.get('job_log_json') or '[]')
        n = int(request.args.get('log_lines', 20))
        return jsonify({
            'job_id':       job['id'],
            'status':       job['status'],
            'total_docs':   job['total_docs'],
            'imported_docs': job['imported_docs'],
            'skipped_docs': job['skipped_docs'],
            'failed_docs':  job['failed_docs'],
            'error_message': job.get('error_message', ''),
            'created_at':   job['created_at'],
            'started_at':   job.get('started_at', ''),
            'completed_at': job.get('completed_at', ''),
            'log_tail':     log_lines[-n:] if log_lines else [],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/import/cancel/<job_id>', methods=['POST'])
@login_required
def court_import_cancel(job_id):
    """Signal a running import job to cancel."""
    ok, err = _court_gate()
    if not ok:
        return err
    try:
        from analyzer.court_connectors.import_job import get_job_manager
        jm = get_job_manager()
        sent = jm.cancel_job(job_id)
        return jsonify({'ok': sent,
                        'message': 'Cancel signal sent' if sent else 'Job not active'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/import/history', methods=['GET'])
@login_required
def court_import_history():
    """Return recent import jobs for the current project."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')
    limit = int(request.args.get('limit', 20))
    try:
        import json as _json
        from analyzer.court_db import get_import_history
        jobs = get_import_history(project_slug, limit=limit)
        for job in jobs:
            raw = job.pop('job_log_json', '[]') or '[]'
            try:
                job['log_tail'] = _json.loads(raw)[-15:]
            except Exception:
                job['log_tail'] = []
        return jsonify({'jobs': jobs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/credentials/parse', methods=['POST'])
@login_required
@advanced_required
def court_parse_credentials():
    """
    Use AI to parse free-form text (email, Slack message, lawyer notes) into
    structured court credential fields.

    Request JSON:
        raw_text   (str)  — optional on follow-up turns
        conversation (list) — [{role, content}, ...] full history including
                              the latest user message to send

    Returns JSON:
        court_system, username, password, pacer_client_code,
        courtlistener_api_token, nyscef_county, public_only,
        summary, follow_up, complete, notes
    """
    from flask import current_app
    ok, err = _court_gate()
    if not ok:
        return err

    data = safe_json_body()
    raw_text     = (data.get('raw_text') or '').strip()
    conversation = data.get('conversation') or []

    if not raw_text and not conversation:
        return jsonify({'error': 'raw_text or conversation required'}), 400

    llm = getattr(current_app, 'llm_client', None)
    if not llm or not llm.client:
        return jsonify({
            'error': 'AI not configured — set up an AI provider in Settings first'
        }), 503

    system_prompt = (
        "You are an expert at extracting court system login credentials from "
        "unstructured text (emails, Slack messages, attorney notes, etc.).\n\n"
        "Supported court systems:\n"
        "  - \"federal\": Uses PACER (username + password + optional billing "
        "client code) and/or a free CourtListener API token.\n"
        "  - \"nyscef\": New York state courts — NY Attorney Registration # + "
        "NYSCEF e-Filing password + optional default county.\n\n"
        "RULES:\n"
        "  - Phrases like 'public access', 'no login required', 'free access', "
        "'I am a party', 'I am a defendant', 'I am a plaintiff' (not an attorney) "
        "mean the user has no professional credentials — set public_only:true.\n"
        "  - For federal + public_only: CourtListener works without PACER.\n"
        "  - For nyscef + public_only: parties/defendants/plaintiffs can use the "
        "public NYSCEF portal with just an index number — no attorney login needed.\n"
        "  - Extract usernames/passwords even if labelled differently "
        "(e.g. 'login: X', 'user: X', 'pw: Y').\n"
        "  - If court system is unclear, ask ONE clarifying question.\n"
        "  - Ask follow-up questions ONE AT A TIME — never ask multiple at once.\n"
        "  - Set complete:true when you have enough to configure the system "
        "(public_only + court_system is sufficient for both public federal and "
        "public NYSCEF; no password required in public_only mode).\n\n"
        "Respond with ONLY valid JSON — no markdown fences, no extra text:\n"
        '{"court_system":"federal"|"nyscef"|null,'
        '"username":null,"password":null,'
        '"pacer_client_code":null,"courtlistener_api_token":null,'
        '"nyscef_county":null,"public_only":false,'
        '"summary":"plain English of what was found",'
        '"follow_up":"single question or null",'
        '"complete":false,'
        '"notes":"any other important observations or null"}'
    )

    # Build message list — client sends the full conversation including the
    # latest user turn, so we just pass it through.
    if conversation:
        messages = conversation
    elif raw_text:
        messages = [{'role': 'user',
                     'content': f"Please parse these court credentials:\n\n{raw_text}"}]
    else:
        return jsonify({'error': 'No input to parse'}), 400

    try:
        import json as _json
        raw_response = ''

        if llm.provider == 'openai':
            resp = llm.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'system', 'content': system_prompt}] + messages,
                temperature=0,
                max_tokens=600,
            )
            raw_response = resp.choices[0].message.content or ''
        else:
            # Anthropic — system param is separate
            resp = llm.client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=600,
                system=system_prompt,
                messages=messages,
            )
            raw_response = resp.content[0].text if resp.content else ''

        # Strip markdown code fences if present
        raw_stripped = raw_response.strip()
        if raw_stripped.startswith('```'):
            raw_stripped = raw_stripped.split('\n', 1)[1]
            raw_stripped = raw_stripped.rsplit('```', 1)[0]

        parsed = _json.loads(raw_stripped)
        return jsonify(parsed)

    except Exception as e:
        logger.error(f"Court credential parse failed: {e}")
        return jsonify({'error': str(e)}), 500
