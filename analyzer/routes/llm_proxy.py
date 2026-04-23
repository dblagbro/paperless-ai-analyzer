"""Admin routes for managing the LLM proxy endpoint pool.

Endpoints (all admin-only):
    GET    /api/llm-proxy/endpoints            — list all endpoints (api_key masked)
    POST   /api/llm-proxy/endpoints            — create a new endpoint
    GET    /api/llm-proxy/endpoints/<id>       — get single endpoint (api_key masked)
    PATCH  /api/llm-proxy/endpoints/<id>       — update (partial; empty api_key preserves existing)
    DELETE /api/llm-proxy/endpoints/<id>       — delete
    POST   /api/llm-proxy/endpoints/<id>/test  — send a ping request; returns {ok, model, error}
"""
import logging
from functools import wraps

from flask import Blueprint, request, jsonify, session
from flask_login import login_required, current_user

from analyzer.db import (
    llm_proxy_list_all,
    llm_proxy_get,
    llm_proxy_create,
    llm_proxy_update,
    llm_proxy_delete,
)
from analyzer.llm import proxy_manager

logger = logging.getLogger(__name__)

bp = Blueprint('llm_proxy', __name__)


def _admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated


def _mask_key(key: str) -> str:
    """Show first 6 chars + last 4, mask the middle."""
    if not key:
        return ''
    if len(key) <= 12:
        return '•' * len(key)
    return f"{key[:6]}••••{key[-4:]}"


def _serialize(ep: dict) -> dict:
    out = dict(ep)
    out['api_key_masked'] = _mask_key(out.pop('api_key', ''))
    out['enabled'] = bool(out.get('enabled'))
    # Attach live circuit-breaker status for admin UI display
    out['breaker'] = proxy_manager.get_breaker_status(ep['id'])
    return out


@bp.route('/api/llm-proxy/endpoints', methods=['GET'])
@login_required
@_admin_required
def list_endpoints():
    eps = llm_proxy_list_all()
    return jsonify({'endpoints': [_serialize(e) for e in eps]})


@bp.route('/api/llm-proxy/endpoints', methods=['POST'])
@login_required
@_admin_required
def create_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    label = (data.get('label') or '').strip()
    url = (data.get('url') or '').strip()
    api_key = (data.get('api_key') or '').strip()

    if not label or not url or not api_key:
        return jsonify({'error': 'label, url, and api_key are required'}), 400

    try:
        version = int(data.get('version', 1))
        priority = int(data.get('priority', 10))
    except (TypeError, ValueError):
        return jsonify({'error': 'version and priority must be integers'}), 400

    if version not in (1, 2):
        return jsonify({'error': 'version must be 1 or 2'}), 400

    enabled = bool(data.get('enabled', True))
    eid = llm_proxy_create(label=label, url=url, api_key=api_key,
                            version=version, priority=priority, enabled=enabled)
    logger.info(f"[llm-proxy] admin {current_user.username} created endpoint {eid} ({label})")
    return jsonify({'id': eid, 'success': True}), 201


@bp.route('/api/llm-proxy/endpoints/<eid>', methods=['GET'])
@login_required
@_admin_required
def get_endpoint(eid):
    ep = llm_proxy_get(eid)
    if not ep:
        return jsonify({'error': 'Endpoint not found'}), 404
    return jsonify(_serialize(ep))


@bp.route('/api/llm-proxy/endpoints/<eid>', methods=['PATCH', 'PUT'])
@login_required
@_admin_required
def update_endpoint(eid):
    existing = llm_proxy_get(eid)
    if not existing:
        return jsonify({'error': 'Endpoint not found'}), 404

    data = request.get_json(force=True, silent=True) or {}
    # Whitelist + coerce
    payload = {}
    for field in ('label', 'url', 'api_key'):
        if field in data and data[field] is not None:
            payload[field] = str(data[field]).strip()
    if 'version' in data:
        try:
            payload['version'] = int(data['version'])
            if payload['version'] not in (1, 2):
                return jsonify({'error': 'version must be 1 or 2'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'version must be an integer'}), 400
    if 'priority' in data:
        try:
            payload['priority'] = int(data['priority'])
        except (TypeError, ValueError):
            return jsonify({'error': 'priority must be an integer'}), 400
    if 'enabled' in data:
        payload['enabled'] = bool(data['enabled'])

    llm_proxy_update(eid, **payload)
    logger.info(f"[llm-proxy] admin {current_user.username} updated endpoint {eid} ({list(payload.keys())})")
    return jsonify({'success': True})


@bp.route('/api/llm-proxy/endpoints/<eid>', methods=['DELETE'])
@login_required
@_admin_required
def delete_endpoint(eid):
    if not llm_proxy_delete(eid):
        return jsonify({'error': 'Endpoint not found'}), 404
    logger.info(f"[llm-proxy] admin {current_user.username} deleted endpoint {eid}")
    return jsonify({'success': True})


@bp.route('/api/llm-proxy/endpoints/<eid>/test', methods=['POST'])
@login_required
@_admin_required
def test_endpoint(eid):
    """Send a minimal ping request through the endpoint. Returns ok/model/error.
    Marks circuit-breaker success/failure."""
    ep = llm_proxy_get(eid)
    if not ep:
        return jsonify({'error': 'Endpoint not found'}), 404

    try:
        client = proxy_manager.build_client(ep)
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': 'ping'}],
            max_tokens=5,
            timeout=15.0,
        )
        model_used = getattr(resp, 'model', 'unknown')
        proxy_manager.mark_success(eid)
        return jsonify({'ok': True, 'model': model_used})
    except Exception as e:
        proxy_manager.mark_failure(eid)
        logger.warning(f"[llm-proxy] test endpoint {eid} failed: {e}")
        return jsonify({'ok': False, 'error': str(e)[:300]}), 200
