"""Chat branching (edit message, branch, set-leaf).

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

@bp.route('/api/chat/sessions/<session_id>/messages/<int:message_id>/edit', methods=['PATCH'])
@login_required
def api_chat_message_edit(session_id, message_id):
    """Edit a user message and delete subsequent messages so the conversation can be resent."""
    try:
        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        if not can_access_session(session_id, current_user.id):
            return jsonify({'error': 'Access denied'}), 403
        data = safe_json_body()
        new_content = data.get('content', '').strip()
        if not new_content:
            return jsonify({'error': 'Content required'}), 400
        update_message_content(message_id, session_id, new_content)
        delete_messages_from(session_id, message_id + 1)
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Edit message error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions/<session_id>/branch', methods=['POST'])
@login_required
def api_chat_branch(session_id):
    """
    Compute branch parent for an edit without inserting any message.
    Returns {parent_id, history_path} so the client can call /api/chat with
    branch_parent_id and rebuild chatHistory from the shared prefix.

    Body: {edit_from_id: N}
    """
    try:
        sess = get_session(session_id)
        if not sess or not can_access_session(session_id, current_user.id):
            return jsonify({'error': 'Access denied'}), 403
        data = safe_json_body()
        edit_from_id = data.get('edit_from_id')
        if not edit_from_id:
            return jsonify({'error': 'edit_from_id required'}), 400

        original = get_message_by_id(edit_from_id, session_id)
        if not original:
            return jsonify({'error': 'Message not found'}), 404

        # The new user message will be a sibling of the original — same parent
        parent_id = original['parent_id']

        # For legacy sessions where parent_id is NULL, infer the parent as the
        # message just before this one in insertion order
        if parent_id is None:
            all_msgs = get_messages(session_id)
            before = [m for m in all_msgs if m['id'] < edit_from_id]
            parent_id = before[-1]['id'] if before else None

        # Build the shared history path (root up to and including parent)
        _, _ = _compute_branch_data(session_id)   # warm the cache (no-op here)
        all_msgs = get_messages(session_id)
        msg_map = {m['id']: dict(m) for m in all_msgs}

        # Walk from parent to root
        history_path = []
        cur = parent_id
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            msg = msg_map.get(cur)
            if not msg:
                break
            history_path.append({'id': msg['id'], 'role': msg['role'], 'content': msg['content']})
            cur = msg.get('parent_id')
        history_path.reverse()

        return jsonify({
            'parent_id': parent_id,
            'history_path': history_path,
        })
    except Exception as e:
        logger.error(f"Branch error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/chat/sessions/<session_id>/set-leaf', methods=['PATCH'])
@login_required
def api_chat_set_leaf(session_id):
    """
    Switch the active branch by pointing active_leaf_id to a different leaf.
    Body: {leaf_id: N}
    Returns: {active_path: [...ids], fork_points: [...]}
    """
    try:
        sess = get_session(session_id)
        if not sess or not can_access_session(session_id, current_user.id):
            return jsonify({'error': 'Access denied'}), 403
        data = safe_json_body()
        leaf_id = data.get('leaf_id')
        if not leaf_id:
            return jsonify({'error': 'leaf_id required'}), 400
        # Verify the leaf belongs to this session
        leaf_msg = get_message_by_id(leaf_id, session_id)
        if not leaf_msg:
            return jsonify({'error': 'Message not found in this session'}), 404
        set_active_leaf(session_id, leaf_id)
        active_path, fork_points = _compute_branch_data(session_id)
        return jsonify({
            'active_path': [m['id'] for m in active_path],
            'fork_points': fork_points,
            'messages': [
                {'id': m['id'], 'role': m['role'], 'content': m['content'],
                 'created_at': m['created_at'], 'parent_id': m.get('parent_id')}
                for m in active_path
            ],
        })
    except Exception as e:
        logger.error(f"Set-leaf error: {e}")
        return jsonify({'error': str(e)}), 500


