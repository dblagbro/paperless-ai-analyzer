"""Chat session PDF export.

Extracted from routes/chat.py during the v3.9.3 maintainability refactor.
"""
import logging
import os
from datetime import datetime
from threading import Thread
from flask import Blueprint, request, jsonify, session, make_response, render_template
from flask_login import login_required, current_user

from analyzer.app import ui_state
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

@bp.route('/api/chat/sessions/<session_id>/export', methods=['GET'])
@login_required
def api_chat_session_export(session_id):
    """Export chat session as PDF."""
    try:
        import mistune
        import weasyprint

        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and not can_access_session(session_id, current_user.id):
            return jsonify({'error': 'Access denied'}), 403

        msgs = get_messages(session_id)
        md = mistune.create_markdown(plugins=['table', 'strikethrough'])

        messages_with_html = []
        for m in msgs:
            messages_with_html.append({
                'role': m['role'],
                'content': m['content'],
                'html_content': md(m['content']),
                'created_at': m['created_at'],
            })

        html_content = render_template(
            'chat_export.html',
            session_title=sess['title'],
            username=current_user.display_name,
            export_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
            document_type=sess['document_type'],
            messages=messages_with_html,
        )

        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        safe_title = ''.join(c for c in sess['title'] if c.isalnum() or c in ' -_')[:40]
        response.headers['Content-Disposition'] = (
            f'attachment; filename="chat-{session_id[:8]}-{safe_title}.pdf"'
        )
        return response
    except Exception as e:
        logger.error(f"Export session error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
