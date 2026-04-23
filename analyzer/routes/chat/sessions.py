"""Chat session CRUD + sharing endpoints.

Extracted from routes/chat.py during the v3.9.3 maintainability refactor.
"""
import logging
import os
from datetime import datetime
from threading import Thread
from flask import Blueprint, request, jsonify, session, make_response, render_template
from flask_login import login_required, current_user

from analyzer.app import ui_state, safe_json_body
from analyzer.db import (
    get_session, create_session, can_access_session, get_sessions,
    get_all_sessions_by_user, get_messages, append_message, update_session_title,
    delete_session, get_session_shares, share_session, unshare_session,
    get_active_leaf, set_active_leaf, get_message_by_id, update_message_content,
    delete_messages_from,
)
from analyzer.db import get_user_by_username
from analyzer.services.ai_config_service import load_ai_config, get_project_ai_config

# Business-logic helpers extracted 2026-04-23 (v3.9.1 refactor).
# These used to live in this file as _vision_extract_doc, _load_session_web_context,
# _save_session_web_context, _ddg_search, _resolve_court_docket_url,
# _fetch_url_text, _compute_branch_data. See refactor-log Entry 005.
from analyzer.services.web_research_service import (
    SEARCH_INTENT_PHRASES as _SEARCH_INTENT_PHRASES,
    load_session_web_context as _load_session_web_context,
    save_session_web_context as _save_session_web_context,
    ddg_search as _ddg_search,
    resolve_court_docket_url as _resolve_court_docket_url,
    fetch_url_text as _fetch_url_text,
)
from analyzer.services.vision_service import vision_extract_doc as _vision_extract_doc
from analyzer.services.chat_branch_service import compute_branch_data as _compute_branch_data

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Chat routes (helpers moved to analyzer/services/ — see imports above)
# ---------------------------------------------------------------------------

from analyzer.routes.chat import bp

@bp.route('/api/chat/sessions', methods=['GET'])
@login_required
def api_chat_sessions_list():
    """List chat sessions: own + shared. Admin sees all grouped by user."""
    try:
        if current_user.is_admin:
            rows = get_all_sessions_by_user()
            # Group by owner
            by_user = {}
            for r in rows:
                owner = r['owner_username']
                if owner not in by_user:
                    by_user[owner] = []
                by_user[owner].append({
                    'id': r['id'],
                    'title': r['title'],
                    'document_type': r['document_type'],
                    'created_at': r['created_at'],
                    'updated_at': r['updated_at'],
                    'owner_username': r['owner_username'],
                    'is_shared': False,
                    'is_own': r['user_id'] == current_user.id,
                })
            return jsonify({'sessions_by_user': by_user, 'is_admin': True})
        else:
            rows = get_sessions(current_user.id, project_slug=session.get('current_project', 'default'))
            sessions_list = [{
                'id': r['id'],
                'title': r['title'],
                'document_type': r['document_type'],
                'created_at': r['created_at'],
                'updated_at': r['updated_at'],
                'owner_username': r['owner_username'],
                'is_shared': bool(r['is_shared']),
                'is_own': r['user_id'] == current_user.id,
            } for r in rows]
            return jsonify({'sessions': sessions_list, 'is_admin': False})
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions', methods=['POST'])
@login_required
def api_chat_sessions_create():
    """Create a new chat session."""
    try:
        data = safe_json_body()
        title = data.get('title', 'New Chat')
        document_type = data.get('document_type')
        project_slug = session.get('current_project', 'default')
        session_id = create_session(current_user.id, title=title, document_type=document_type,
                                    project_slug=project_slug)
        return jsonify({'session_id': session_id, 'title': title})
    except Exception as e:
        logger.error(f"Create session error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions/<session_id>', methods=['GET'])
@login_required
def api_chat_session_get(session_id):
    """Get session details + messages."""
    try:
        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        # Check access
        if not current_user.is_admin and not can_access_session(session_id, current_user.id):
            return jsonify({'error': 'Access denied'}), 403
        active_path, fork_points = _compute_branch_data(session_id)
        shares = get_session_shares(session_id)
        # All messages in active path (with parent_id for client-side tree awareness)
        return jsonify({
            'session': {
                'id': sess['id'],
                'title': sess['title'],
                'document_type': sess['document_type'],
                'created_at': sess['created_at'],
                'updated_at': sess['updated_at'],
                'user_id': sess['user_id'],
                'is_owner': sess['user_id'] == current_user.id,
            },
            'messages': [
                {
                    'id': m['id'], 'role': m['role'], 'content': m['content'],
                    'created_at': m['created_at'], 'parent_id': m.get('parent_id'),
                }
                for m in active_path
            ],
            'active_path': [m['id'] for m in active_path],
            'fork_points': fork_points,
            'shared_with': [{'id': s['id'], 'username': s['username']} for s in shares],
        })
    except Exception as e:
        logger.error(f"Get session error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
@login_required
def api_chat_session_delete(session_id):
    """Delete a session (owner or admin only)."""
    try:
        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and sess['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        delete_session(session_id)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions/<session_id>', methods=['PATCH'])
@login_required
def api_chat_session_rename(session_id):
    """Rename a session title."""
    try:
        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and sess['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        data = safe_json_body()
        title = data.get('title', '').strip()
        if not title:
            return jsonify({'error': 'Title required'}), 400
        update_session_title(session_id, title)
        return jsonify({'success': True, 'title': title})
    except Exception as e:
        logger.error(f"Rename session error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions/<session_id>/share', methods=['POST'])
@login_required
def api_chat_session_share(session_id):
    """Share session with a user.

    v3.9.4: accepts either `username` (string) or `uid` (int). Older API
    consumers sent `uid`; the UI sends `username`.
    """
    try:
        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and sess['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        data = safe_json_body()
        target_username = (data.get('username') or '').strip()
        target_uid = data.get('uid')
        if target_uid is not None:
            from analyzer.db import get_user_by_id
            target_user = get_user_by_id(target_uid)
            if not target_user:
                return jsonify({'error': f"User id {target_uid} not found"}), 404
            target_username = target_user['username']
        elif target_username:
            target_user = get_user_by_username(target_username)
            if not target_user:
                return jsonify({'error': f"User '{target_username}' not found"}), 404
        else:
            return jsonify({'error': 'Username or uid required'}), 400
        if target_user['id'] == sess['user_id']:
            return jsonify({'error': 'Cannot share with the owner'}), 400
        share_session(session_id, target_user['id'], current_user.id)
        return jsonify({'success': True, 'shared_with': target_username})
    except Exception as e:
        logger.error(f"Share session error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions/<session_id>/share/<int:uid>', methods=['DELETE'])
@login_required
def api_chat_session_unshare(session_id, uid):
    """Remove share from a user."""
    try:
        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and sess['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        unshare_session(session_id, uid)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Unshare session error: {e}")
        return jsonify({'error': str(e)}), 500


