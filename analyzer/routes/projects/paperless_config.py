"""Per-project Paperless-ngx configuration, health-check, and document-link endpoints."""
import logging

from flask import current_app, jsonify
from flask_login import login_required

from analyzer.app import _project_client_cache, safe_json_body

from . import bp

logger = logging.getLogger(__name__)


@bp.route('/api/projects/<slug>/paperless-config', methods=['POST'])
@login_required
def api_set_project_paperless_config(slug):
    """Save per-project Paperless-ngx connection config."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = current_app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        data = safe_json_body()
        updates = {}
        if 'url' in data:
            updates['paperless_url'] = data['url'] or None
        if 'token' in data:
            updates['paperless_token'] = data['token'] or None
        if 'doc_base_url' in data:
            updates['paperless_doc_base_url'] = data['doc_base_url'] or None

        current_app.project_manager.update_project(slug, **updates)
        _project_client_cache.pop(slug, None)

        logger.info(f"Updated Paperless config for project '{slug}'")
        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Failed to set Paperless config for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/<slug>/paperless-config', methods=['GET'])
@login_required
def api_get_project_paperless_config(slug):
    """Get per-project Paperless-ngx connection config (token is masked)."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = current_app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        cfg = current_app.project_manager.get_paperless_config(slug)
        return jsonify({
            'url': cfg.get('url') or '',
            'token_set': bool(cfg.get('token')),
            'doc_base_url': cfg.get('doc_base_url') or '',
        })
    except Exception as e:
        logger.error(f"Failed to get Paperless config for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/<slug>/paperless-health-check', methods=['POST'])
@login_required
def api_paperless_health_check(slug):
    """Test-connect a Paperless URL + token without saving."""
    try:
        data = safe_json_body()
        url = (data.get('url') or '').strip().rstrip('/')
        token = (data.get('token') or '').strip()
        if not url:
            return jsonify({'ok': False, 'error': 'url is required'}), 400
        if not token and current_app.project_manager:
            cfg = current_app.project_manager.get_paperless_config(slug)
            token = cfg.get('token') or ''
        if not token:
            return jsonify({'ok': False, 'error': 'No token provided and none saved for this project'}), 400
        from analyzer.paperless_client import PaperlessClient
        pc = PaperlessClient(base_url=url, api_token=token)
        healthy = pc.health_check()
        if healthy:
            return jsonify({'ok': True, 'message': 'Paperless API is reachable'})
        else:
            return jsonify({'ok': False, 'error': 'Health check failed — check URL and token'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@bp.route('/api/projects/<slug>/doc-link/<int:doc_id>', methods=['GET'])
@login_required
def api_project_doc_link(slug, doc_id):
    """Return the public Paperless URL for a specific document in a project."""
    if not current_app.project_manager:
        return jsonify({'url': None})
    try:
        cfg = current_app.project_manager.get_paperless_config(slug)
        base = (cfg.get('doc_base_url') or '').rstrip('/')
        url = f"{base}/documents/{doc_id}/details" if base else None
        return jsonify({'url': url})
    except Exception as e:
        logger.error(f"doc-link error for {slug}/{doc_id}: {e}")
        return jsonify({'url': None})
