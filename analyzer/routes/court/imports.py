"""Court document import lifecycle: start job, poll status, cancel, history,
and the related analyze-missing helper."""
import logging

from flask import jsonify, request, session
from flask_login import login_required

from analyzer.app import advanced_required, safe_json_body
from . import bp
from .helpers import (
    _analyze_missing_for_project,
    _court_gate,
    _get_current_project_slug,
    _run_court_import,
)

logger = logging.getLogger(__name__)

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


