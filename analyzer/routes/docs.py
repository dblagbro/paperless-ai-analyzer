import logging
from flask import Blueprint, request, jsonify, session, render_template, current_app
from flask_login import login_required, current_user

from analyzer.services.ai_config_service import load_ai_config, get_project_ai_config

logger = logging.getLogger(__name__)

bp = Blueprint('docs', __name__)

_DOCS_PAGES = {
    '': 'overview',
    'overview': 'overview',
    'getting-started': 'getting-started',
    'projects': 'projects',
    'search': 'search',
    'chat': 'chat',
    'configuration': 'configuration',
    'anomaly-detection': 'anomaly-detection',
    'upload': 'upload',
    'tools': 'tools',
    'users': 'users',
    'llm-usage': 'llm-usage',
    'api': 'api',
    'case-intelligence': 'case-intelligence',
    'court-import': 'court-import',
}

try:
    from analyzer.app import _APP_VERSION
except ImportError:
    _APP_VERSION = '0.0.0'


@bp.route('/docs')
@bp.route('/docs/')
@bp.route('/docs/<path:page>')
@login_required
def docs(page=''):
    """Serve the user manual."""
    slug = _DOCS_PAGES.get(page.strip('/'), 'overview')
    url_prefix = request.script_root
    github_url = 'https://github.com/dblagbro/paperless-ai-analyzer'
    version = _APP_VERSION
    return render_template(
        'docs.html',
        page=slug,
        url_prefix=url_prefix,
        github_url=github_url,
        version=version,
        is_admin=current_user.is_authenticated and current_user.is_admin,
    )


@bp.route('/api/docs/ask', methods=['POST'])
@login_required
def api_docs_ask():
    """Answer a question about the application using the configured LLM (no RAG)."""
    try:
        data = request.get_json() or {}
        question = data.get('question', '').strip()
        history = data.get('history', [])
        if not question:
            return jsonify({'error': 'question required'}), 400

        system_prompt = (
            "You are the built-in help assistant for Paperless AI Analyzer, "
            "an intelligent document intelligence platform for Paperless-ngx. "
            "Answer questions about how to use the application concisely and accurately. "
            "Key features: Projects (isolated workspaces), Smart Upload (File/URL/Cloud/Court Import), "
            "AI Chat with RAG (edit messages, stop, compare LLMs), Case Intelligence AI "
            "(Director→Manager→Worker orchestration, 6 domains, report builder), "
            "Search & Analysis (semantic + exact), Anomaly Detection (balance mismatch, duplicate amounts, risk score), "
            "Configuration (AI settings, profiles, vector store, SMTP, users), "
            "Debug & Tools (health, containers, reprocess, reconcile, logs). "
            "Be concise. If unsure, say so."
        )
        messages = [{'role': h['role'], 'content': h['content']}
                    for h in history[-8:] if h.get('role') in ('user', 'assistant')]
        messages.append({'role': 'user', 'content': question})

        _proj = session.get('current_project', 'default')
        ai_cfg = get_project_ai_config(_proj, 'chat')

        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        try:
            result = call_llm(
                messages=[{'role': 'system', 'content': system_prompt}] + messages,
                task='qa',
                max_tokens=1024,
                project_slug=_proj,
                operation='docs_ask',
                direct_provider=ai_cfg.get('provider'),
                direct_api_key=ai_cfg.get('api_key'),
                direct_model=ai_cfg.get('model'),
            )
            return jsonify({'answer': result['content']})
        except LLMUnavailableError as e:
            logger.warning(f"docs/ask: {e} attempted={e.attempted}")
            return jsonify({
                'error': 'No AI provider available — check Configuration → AI Settings.',
                'source': 'llm-pool-exhausted',
            }), 503
    except Exception as e:
        logger.error(f"docs/ask error: {e}")
        return jsonify({'error': str(e)}), 500
