"""
Flask application factory, core middleware, auth decorators, and server entry points.

This module owns the global Flask `app` instance. Route modules import `current_app`
from Flask (Blueprint pattern); this module is the only place that references `app`
directly at module scope.
"""

import os
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, Thread
from collections import deque
from functools import wraps
from typing import Dict, Any

from flask import Flask, request, session as flask_session, jsonify
from flask_login import current_user

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory log buffer (last 200 lines)
# ---------------------------------------------------------------------------
log_buffer = deque(maxlen=200)


class LogBufferHandler(logging.Handler):
    def emit(self, record):
        try:
            log_buffer.append(self.format(record))
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Global UI state (thread-safe)
# ---------------------------------------------------------------------------
ui_state = {
    'recent_analyses': [],
    'stats': {
        'total_analyzed': 0,
        'anomalies_detected': 0,
        'profiles_matched': 0,
        'profiles_needed': 0,
        'high_risk_count': 0,
    },
    'last_update': None,
    'lock': Lock(),
}


# ---------------------------------------------------------------------------
# Flask app instance
# ---------------------------------------------------------------------------
def _load_or_generate_secret_key() -> str:
    env_key = os.environ.get('FLASK_SECRET_KEY', '').strip()
    if env_key:
        return env_key
    key_file = Path('/app/data/.flask_secret_key')
    try:
        if key_file.exists():
            stored = key_file.read_text().strip()
            if stored:
                return stored
    except Exception:
        pass
    import secrets
    new_key = secrets.token_hex(32)
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(new_key)
        key_file.chmod(0o600)
        logger.info("Generated new Flask secret key and saved to /app/data/.flask_secret_key")
    except Exception as e:
        logger.warning(f"Could not persist Flask secret key: {e} — key will change on restart")
    return new_key


class _ReverseProxied:
    """WSGI middleware that reads URL_PREFIX env var and patches SCRIPT_NAME.

    When nginx strips a sub-path prefix before proxying, Flask never sees it and
    generates bare URLs. This middleware restores the prefix so url_for() includes it.
    """
    def __init__(self, wsgi_app):
        self.app = wsgi_app

    def __call__(self, environ, start_response):
        script_name = os.environ.get('URL_PREFIX', '').rstrip('/')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ.get('PATH_INFO', '')
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):] or '/'
        return self.app(environ, start_response)


app = Flask(__name__, template_folder='/app/analyzer/templates', static_folder='/app/analyzer/static')

app.secret_key = _load_or_generate_secret_key()
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

_url_prefix = os.environ.get('URL_PREFIX', '').strip('/')
_cookie_suffix = _url_prefix.replace('-', '_').replace('/', '_') if _url_prefix else 'paperless'
app.config['SESSION_COOKIE_NAME'] = f'{_cookie_suffix}_session'
app.config['REMEMBER_COOKIE_NAME'] = f'{_cookie_suffix}_remember_token'
if _url_prefix:
    app.config['SESSION_COOKIE_PATH'] = f'/{_url_prefix}/'
    app.config['REMEMBER_COOKIE_PATH'] = f'/{_url_prefix}/'

app.wsgi_app = _ReverseProxied(app.wsgi_app)

from analyzer import __version__ as _APP_VERSION  # noqa: E402

from analyzer.auth import login_manager  # noqa: E402
from analyzer.db import (  # noqa: E402
    init_db, get_user_by_username, get_user_by_id, update_last_login,
    get_sessions, get_all_sessions_by_user, create_session, get_session,
    get_messages, append_message, update_session_title, delete_session,
    update_message_content, delete_messages_from,
    get_message_by_id, get_active_leaf, set_active_leaf,
    share_session, unshare_session, get_session_shares, can_access_session,
    list_users, create_user as db_create_user, update_user as db_update_user,
    log_import, get_import_history,
    mark_document_processed, count_processed_documents, get_analyzed_doc_ids,
)

login_manager.init_app(app)


# ---------------------------------------------------------------------------
# Request hooks
# ---------------------------------------------------------------------------
@app.before_request
def make_session_permanent():
    flask_session.permanent = True


@app.after_request
def add_no_cache(response):
    if request.path.startswith('/api/') or request.path == '/':
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


# ---------------------------------------------------------------------------
# Auth decorators
# ---------------------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated


def advanced_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        if current_user.role not in ('advanced', 'admin'):
            return jsonify({'error': 'Advanced user access required'}), 403
        return f(*args, **kwargs)
    return decorated


def _ci_gate():
    """Returns (True, None) if CI is allowed, else (False, (response, code))."""
    try:
        from analyzer.case_intelligence import _ci_safety_check
        if not _ci_safety_check():
            return False, (jsonify({'error': 'Case Intelligence AI is not enabled'}), 404)
    except RuntimeError as e:
        return False, (jsonify({'error': str(e)}), 403)
    except ImportError:
        return False, (jsonify({'error': 'Case Intelligence AI module not available'}), 404)
    return True, None


def _ci_can_read(run) -> bool:
    if current_user.is_admin:
        return True
    if run['user_id'] == current_user.id:
        return True
    try:
        from analyzer.case_intelligence.db import is_run_shared_with
        return is_run_shared_with(run['id'], current_user.id)
    except Exception:
        return False


def _ci_can_write(run) -> bool:
    return run['user_id'] == current_user.id or current_user.is_admin


@app.context_processor
def inject_user():
    is_admin = current_user.is_authenticated and current_user.is_admin
    is_advanced = current_user.is_authenticated and current_user.role in ('advanced', 'admin')
    return {
        'current_user': current_user,
        'is_admin': is_admin,
        'is_advanced': is_advanced,
        'CI_ENABLED': True,
        'COURT_IMPORT_ENABLED': True,
    }


# ---------------------------------------------------------------------------
# UI state initializer (called from create_app after deps are attached)
# ---------------------------------------------------------------------------
def initialize_ui_state():
    try:
        logger.info("Initializing UI state from existing documents...")
        analyzed_tag_id = app.paperless_client.get_or_create_tag('analyzed:deterministic:v1')
        if not analyzed_tag_id:
            logger.info("No analyzed tag found, UI state will start empty")
            return
        documents_list = app.paperless_client.get_documents_by_tag(analyzed_tag_id)
        if not documents_list:
            logger.info("No previously analyzed documents found")
            return
        total_docs = len(documents_list)
        results = documents_list
        logger.info(f"Found {total_docs} previously analyzed documents")
        db_count = count_processed_documents()
        logger.info(f"Persistent processed_documents count: {db_count}")
        with ui_state['lock']:
            ui_state['stats']['total_analyzed'] = db_count
            high_risk_count = 0
            anomalies_count = 0
            chroma_summaries = {}
            try:
                if (hasattr(app, 'document_analyzer') and app.document_analyzer
                        and app.document_analyzer.vector_store
                        and app.document_analyzer.vector_store.enabled):
                    ids_to_fetch = [str(doc['id']) for doc in results[:100]]
                    chroma_result = app.document_analyzer.vector_store.collection.get(
                        ids=ids_to_fetch, include=['metadatas']
                    )
                    for i, cid in enumerate(chroma_result.get('ids', [])):
                        m = chroma_result['metadatas'][i]
                        chroma_summaries[int(cid)] = {
                            'brief_summary': m.get('brief_summary', ''),
                            'full_summary': m.get('full_summary', ''),
                        }
            except Exception as _ce:
                logger.debug(f"Could not pre-fetch Chroma summaries at startup: {_ce}")
            for doc in results[:100]:
                doc_id = doc['id']
                tags = []
                try:
                    for tag_id in doc.get('tags', [])[:10]:
                        tag_response = app.paperless_client.session.get(
                            f'{app.paperless_client.base_url}/api/tags/{tag_id}/'
                        )
                        if tag_response.ok:
                            tags.append(tag_response.json().get('name', ''))
                except Exception:
                    pass
                anomalies = [t.replace('anomaly:', '') for t in tags if t.startswith('anomaly:')]
                if anomalies:
                    anomalies_count += 1
                risk_score = 0
                if 'anomaly:forensic_risk_high' in tags:
                    risk_score = 80
                    high_risk_count += 1
                elif 'anomaly:forensic_risk_medium' in tags:
                    risk_score = 60
                elif 'anomaly:forensic_risk_low' in tags:
                    risk_score = 30
                chroma_data = chroma_summaries.get(doc_id, {})
                brief_summary = chroma_data.get('brief_summary', '') or f"Financial document: {doc['title']}"
                full_summary = chroma_data.get('full_summary', '')
                ui_state['recent_analyses'].append({
                    'document_id': doc_id,
                    'document_title': doc['title'],
                    'anomalies_found': anomalies[:5],
                    'risk_score': risk_score,
                    'timestamp': doc['modified'],
                    'ai_analysis': '',
                    'created': doc.get('created', ''),
                    'correspondent': doc.get('correspondent', None),
                    'brief_summary': brief_summary,
                    'full_summary': full_summary,
                })
            ui_state['stats']['anomalies_detected'] = anomalies_count
            ui_state['stats']['high_risk_count'] = high_risk_count
            ui_state['last_update'] = results[0].get('modified') if results else None
        logger.info(f"✓ UI state initialized: {total_docs} analyzed, {anomalies_count} with anomalies, {high_risk_count} high risk")
    except Exception as e:
        logger.error(f"Failed to initialize UI state: {e}")
        logger.exception(e)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(state_manager, profile_loader, paperless_client,
               project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    app.state_manager = state_manager
    app.profile_loader = profile_loader
    app.paperless_client = paperless_client
    app.document_analyzer = document_analyzer
    app.project_manager = project_manager
    app.llm_client = llm_client
    app.smart_uploader = smart_uploader

    init_db()

    try:
        from analyzer.case_intelligence import init_case_intelligence
        init_case_intelligence()
    except Exception as _ci_err:
        logger.debug(f"Case Intelligence AI not initialized: {_ci_err}")

    try:
        from analyzer.court_connectors import init_court_import
        init_court_import()
    except Exception as _court_err:
        logger.debug(f"Court Document Importer not initialized: {_court_err}")

    initialize_ui_state()
    return app


# ---------------------------------------------------------------------------
# Per-project Paperless client factory (cached)
# ---------------------------------------------------------------------------
_project_client_cache: dict = {}
_PROJECT_CLIENT_TTL = 300


def _get_project_client(slug: str):
    """Return a PaperlessClient for the given project, falling back to the global client."""
    now = time.time()
    cached = _project_client_cache.get(slug)
    if cached and now - cached[1] < _PROJECT_CLIENT_TTL:
        return cached[0]
    cfg = app.project_manager.get_paperless_config(slug) if app.project_manager else {}
    if cfg.get('url') and cfg.get('token'):
        from analyzer.paperless_client import PaperlessClient
        client = PaperlessClient(base_url=cfg['url'], api_token=cfg['token'])
    else:
        client = app.paperless_client
    _project_client_cache[slug] = (client, now)
    return client


# ---------------------------------------------------------------------------
# UI stats updater (called from main.py DocumentAnalyzer)
# ---------------------------------------------------------------------------
def update_ui_stats(analysis_result: Dict[str, Any]) -> None:
    with ui_state['lock']:
        ui_state['recent_analyses'].append(analysis_result)
        if len(ui_state['recent_analyses']) > 100:
            ui_state['recent_analyses'] = ui_state['recent_analyses'][-100:]
        ui_state['stats']['total_analyzed'] += 1
        doc_id = analysis_result.get('doc_id')
        if doc_id:
            try:
                mark_document_processed(doc_id, project_slug=analysis_result.get('project_slug', 'default'))
            except Exception:
                pass
        if analysis_result.get('anomalies_found'):
            ui_state['stats']['anomalies_detected'] += len(analysis_result['anomalies_found'])
        if analysis_result.get('profile_matched'):
            ui_state['stats']['profiles_matched'] += 1
        else:
            ui_state['stats']['profiles_needed'] += 1
        if analysis_result.get('risk_score', 0) >= 70:
            ui_state['stats']['high_risk_count'] += 1
        ui_state['last_update'] = datetime.utcnow().isoformat()


def _get_uptime() -> int:
    try:
        with open('/proc/uptime', 'r') as f:
            return int(float(f.readline().split()[0]))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Server entry points
# ---------------------------------------------------------------------------
def run_web_server(state_manager, profile_loader, paperless_client, host='0.0.0.0', port=8051,
                   project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    create_app(state_manager, profile_loader, paperless_client,
               project_manager=project_manager, llm_client=llm_client,
               smart_uploader=smart_uploader, document_analyzer=document_analyzer)
    logger.info(f"Starting production web UI on {host}:{port}")
    from waitress import serve
    serve(app, host=host, port=port, threads=4, channel_timeout=300,
          cleanup_interval=10, _quiet=False)


def start_web_server_thread(state_manager, profile_loader, paperless_client, host='0.0.0.0', port=8051,
                            project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    root_logger = logging.getLogger()
    buffer_handler = LogBufferHandler()
    buffer_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(buffer_handler)
    logger.info("Log buffer handler registered - live logs enabled")
    thread = Thread(
        target=run_web_server,
        args=(state_manager, profile_loader, paperless_client, host, port,
              project_manager, llm_client, smart_uploader, document_analyzer),
        daemon=True,
    )
    thread.start()
    logger.info(f"Web UI thread started on {host}:{port}")
    return thread
