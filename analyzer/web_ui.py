"""
Web UI for Paperless AI Analyzer

Simple Flask-based dashboard for monitoring and control.
"""

import os
import json
import logging
import smtplib
import ssl
import time
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
from flask import Flask, render_template, jsonify, request, redirect, url_for, make_response, session
from flask_login import login_required, login_user, logout_user, current_user
from werkzeug.security import check_password_hash
from threading import Thread, Lock
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)

# In-memory log buffer
log_buffer = deque(maxlen=200)  # Keep last 200 log lines


class LogBufferHandler(logging.Handler):
    """Custom log handler that stores logs in memory."""

    def emit(self, record):
        try:
            msg = self.format(record)
            log_buffer.append(msg)
        except Exception:
            self.handleError(record)

# Global state
ui_state = {
    'recent_analyses': [],
    'stats': {
        'total_analyzed': 0,
        'anomalies_detected': 0,
        'profiles_matched': 0,
        'profiles_needed': 0,
        'high_risk_count': 0
    },
    'last_update': None,
    'lock': Lock()
}

app = Flask(__name__, template_folder='/app/analyzer/templates', static_folder='/app/analyzer/static')


def _load_or_generate_secret_key() -> str:
    """Load secret key from env var, persistent file, or generate a new one.

    Priority:
      1. FLASK_SECRET_KEY env var (explicit override)
      2. /app/data/.flask_secret_key file (auto-generated on first run, persists across restarts)
      3. Generate a new cryptographically random key, save it, and use it
    """
    # 1. Explicit env var always wins
    env_key = os.environ.get('FLASK_SECRET_KEY', '').strip()
    if env_key:
        return env_key

    # 2. Persistent file in the data volume
    key_file = Path('/app/data/.flask_secret_key')
    try:
        if key_file.exists():
            stored = key_file.read_text().strip()
            if stored:
                return stored
    except Exception:
        pass

    # 3. Generate, persist, and return
    import secrets
    new_key = secrets.token_hex(32)  # 256-bit random key
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(new_key)
        key_file.chmod(0o600)
        logger.info("Generated new Flask secret key and saved to /app/data/.flask_secret_key")
    except Exception as e:
        logger.warning(f"Could not persist Flask secret key: {e} — key will change on restart")
    return new_key


app.secret_key = _load_or_generate_secret_key()
# Sessions last 7 days (survive browser close)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
# Unique cookie names per deployment so that prod and dev on the same domain
# do not overwrite each other's session/remember cookies.
_url_prefix = os.environ.get('URL_PREFIX', '').strip('/')
_cookie_suffix = _url_prefix.replace('-', '_').replace('/', '_') if _url_prefix else 'paperless'
app.config['SESSION_COOKIE_NAME'] = f'{_cookie_suffix}_session'
app.config['REMEMBER_COOKIE_NAME'] = f'{_cookie_suffix}_remember_token'
if _url_prefix:
    app.config['SESSION_COOKIE_PATH'] = f'/{_url_prefix}/'
    app.config['REMEMBER_COOKIE_PATH'] = f'/{_url_prefix}/'


class _ReverseProxied:
    """WSGI middleware that reads X-Script-Name from nginx and sets SCRIPT_NAME.

    When nginx strips a sub-path prefix (e.g. /paperless-ai-analyzer-dev)
    before proxying, Flask never sees it and generates bare URLs like /login.
    This middleware restores the prefix into the WSGI environ so that
    url_for() automatically includes it in every generated URL.
    """
    def __init__(self, wsgi_app):
        self.app = wsgi_app

    def __call__(self, environ, start_response):
        script_name = os.environ.get('URL_PREFIX', '').rstrip('/')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            # When the request comes in via nginx, nginx already strips the
            # prefix before proxying, so PATH_INFO is already bare (e.g. /login).
            # When the request comes in directly (e.g. Playwright, curl on port
            # 8052), PATH_INFO still contains the full path including the prefix
            # (e.g. /paperless-ai-analyzer-dev/login).  Strip it here so both
            # access paths work correctly.
            path_info = environ.get('PATH_INFO', '')
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):] or '/'
        return self.app(environ, start_response)


app.wsgi_app = _ReverseProxied(app.wsgi_app)

# Version
from analyzer import __version__ as _APP_VERSION

# Auth setup
from analyzer.auth import login_manager
from analyzer.db import (
    init_db, get_user_by_username, get_user_by_id, update_last_login,
    get_sessions, get_all_sessions_by_user, create_session, get_session,
    get_messages, append_message, update_session_title, delete_session,
    update_message_content, delete_messages_from,
    share_session, unshare_session, get_session_shares, can_access_session,
    list_users, create_user as db_create_user, update_user as db_update_user,
    log_import, get_import_history,
    mark_document_processed, count_processed_documents, get_analyzed_doc_ids,
)
login_manager.init_app(app)


@app.before_request
def make_session_permanent():
    """Make every session permanent so the cookie survives browser close/reopen."""
    from flask import session as flask_session
    flask_session.permanent = True


@app.after_request
def add_no_cache(response):
    """Prevent browsers from caching the dashboard or API responses."""
    if request.path.startswith('/api/') or request.path == '/':
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


def admin_required(f):
    """Decorator: requires logged-in admin user, returns 403 otherwise."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated


def advanced_required(f):
    """Decorator: requires 'advanced' or 'admin' role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        if current_user.role not in ('advanced', 'admin'):
            return jsonify({'error': 'Advanced user access required'}), 403
        return f(*args, **kwargs)
    return decorated


def _ci_gate():
    """
    Check Case Intelligence safety gate. Returns (True, None) if allowed,
    or (False, response) if the request should be rejected.
    """
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
    """True if current_user may read this run (owner, admin, or shared-with)."""
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
    """True if current_user may mutate this run (owner or admin only)."""
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


def initialize_ui_state():
    """
    Initialize UI state from existing analyzed documents on startup.
    This ensures counts and recent analyses persist across container restarts.
    """
    try:
        logger.info("Initializing UI state from existing documents...")

        # Get tag ID for analyzed documents
        analyzed_tag_id = app.paperless_client.get_or_create_tag('analyzed:deterministic:v1')
        if not analyzed_tag_id:
            logger.info("No analyzed tag found, UI state will start empty")
            return

        # Query Paperless for all documents with the analyzed tag
        documents_list = app.paperless_client.get_documents_by_tag(analyzed_tag_id)

        if not documents_list:
            logger.info("No previously analyzed documents found")
            return

        total_docs = len(documents_list)
        results = documents_list

        logger.info(f"Found {total_docs} previously analyzed documents")

        # Load persisted analyzed count from app.db (survives restarts)
        db_count = count_processed_documents()
        logger.info(f"Persistent processed_documents count: {db_count}")

        # Update stats
        with ui_state['lock']:
            ui_state['stats']['total_analyzed'] = db_count

            # Count high risk documents
            high_risk_count = 0
            anomalies_count = 0

            # Pre-fetch cached summaries from Chroma vector store (batch lookup)
            chroma_summaries = {}
            try:
                if (hasattr(app, 'document_analyzer') and app.document_analyzer
                        and app.document_analyzer.vector_store
                        and app.document_analyzer.vector_store.enabled):
                    ids_to_fetch = [str(doc['id']) for doc in results[:100]]
                    chroma_result = app.document_analyzer.vector_store.collection.get(
                        ids=ids_to_fetch,
                        include=['metadatas']
                    )
                    for i, cid in enumerate(chroma_result.get('ids', [])):
                        m = chroma_result['metadatas'][i]
                        chroma_summaries[int(cid)] = {
                            'brief_summary': m.get('brief_summary', ''),
                            'full_summary': m.get('full_summary', ''),
                        }
            except Exception as _ce:
                logger.debug(f"Could not pre-fetch Chroma summaries at startup: {_ce}")

            # Process recent documents for the list (last 100 to include more anomalies)
            for doc in results[:100]:
                doc_id = doc['id']

                # Get tags
                tags = []
                try:
                    for tag_id in doc.get('tags', [])[:10]:
                        tag_response = app.paperless_client.session.get(
                            f'{app.paperless_client.base_url}/api/tags/{tag_id}/'
                        )
                        if tag_response.ok:
                            tags.append(tag_response.json().get('name', ''))
                except:
                    pass

                anomalies = [t.replace('anomaly:', '') for t in tags if t.startswith('anomaly:')]
                if anomalies:
                    anomalies_count += 1

                # Determine risk score
                risk_score = 0
                if 'anomaly:forensic_risk_high' in tags:
                    risk_score = 80
                    high_risk_count += 1
                elif 'anomaly:forensic_risk_medium' in tags:
                    risk_score = 60
                elif 'anomaly:forensic_risk_low' in tags:
                    risk_score = 30

                # Use cached summaries from Chroma if available, fall back to placeholder
                chroma_data = chroma_summaries.get(doc_id, {})
                brief_summary = chroma_data.get('brief_summary', '') or f"Financial document: {doc['title']}"
                full_summary = chroma_data.get('full_summary', '')

                # Add to recent analyses
                ui_state['recent_analyses'].append({
                    'document_id': doc_id,
                    'document_title': doc['title'],
                    'anomalies_found': anomalies[:5],
                    'risk_score': risk_score,
                    'timestamp': doc['modified'],
                    'ai_analysis': "",
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


def create_app(state_manager, profile_loader, paperless_client,
               project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    """
    Create and configure Flask app.

    Args:
        state_manager: StateManager instance
        profile_loader: ProfileLoader instance
        paperless_client: PaperlessClient instance
        project_manager: ProjectManager instance (v1.5.0)
        llm_client: LLM client instance (v1.5.0)
        smart_uploader: SmartUploader instance (v1.5.0)
        document_analyzer: DocumentAnalyzer instance (v1.5.0 - for re-analysis)

    Returns:
        Flask app
    """
    app.state_manager = state_manager
    app.profile_loader = profile_loader
    app.paperless_client = paperless_client
    app.document_analyzer = document_analyzer

    # v1.5.0: Project management
    app.project_manager = project_manager
    app.llm_client = llm_client
    app.smart_uploader = smart_uploader

    # Initialize auth database (idempotent)
    init_db()

    # Initialize Case Intelligence AI (DEV only, idempotent)
    try:
        from analyzer.case_intelligence import init_case_intelligence
        init_case_intelligence()
    except Exception as _ci_err:
        logger.debug(f"Case Intelligence AI not initialized: {_ci_err}")

    # Initialize Court Document Importer (idempotent)
    try:
        from analyzer.court_connectors import init_court_import
        init_court_import()
    except Exception as _court_err:
        logger.debug(f"Court Document Importer not initialized: {_court_err}")

    # Initialize UI state from existing documents
    initialize_ui_state()

    return app


# ── v3.6.0: Per-project Paperless client factory ─────────────────────────────
_project_client_cache: dict = {}  # slug -> (PaperlessClient, timestamp)
_PROJECT_CLIENT_TTL = 300  # 5-minute cache


def _get_project_client(slug: str):
    """
    Return a PaperlessClient for the given project slug.

    If the project has its own paperless_url + token configured, returns a
    dedicated client for that instance.  Otherwise falls back to the global
    app.paperless_client (shared instance).

    Results are cached for _PROJECT_CLIENT_TTL seconds so each request doesn't
    hit the DB.
    """
    now = time.time()
    cached = _project_client_cache.get(slug)
    if cached and now - cached[1] < _PROJECT_CLIENT_TTL:
        return cached[0]

    cfg = app.project_manager.get_paperless_config(slug) if app.project_manager else {}
    if cfg.get('url') and cfg.get('token'):
        from analyzer.paperless_client import PaperlessClient
        client = PaperlessClient(base_url=cfg['url'], api_token=cfg['token'])
    else:
        client = app.paperless_client  # fallback: global client

    _project_client_cache[slug] = (client, now)
    return client


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    error = None
    username_val = ''
    if request.method == 'POST':
        username_val = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        row = get_user_by_username(username_val)
        if row and check_password_hash(row['password_hash'], password):
            from analyzer.auth import User
            user = User(row)
            login_user(user, remember=True)
            update_last_login(user.id)
            next_page = request.args.get('next') or url_for('index')
            return redirect(next_page)
        error = 'Invalid username or password.'

    return render_template('login.html', error=error, username=username_val)


@app.route('/logout')
@login_required
def logout():
    """Log out current user."""
    logout_user()
    return redirect(url_for('login_page'))


@app.route('/')
@login_required
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
@login_required
def api_status():
    """Get current analyzer status."""
    with ui_state['lock']:
        state_stats = app.state_manager.get_stats()

        # Get project-scoped analyzed count from DB (don't mutate shared ui_state —
        # that would corrupt counts for concurrent requests from other projects).
        project_slug = session.get('current_project', 'default')
        try:
            project_analyzed = count_processed_documents(project_slug=project_slug)
        except Exception:
            project_analyzed = ui_state['stats'].get('total_analyzed', 0)

        # Get vector store stats for current project
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=project_slug)
        vector_stats = vector_store.get_stats() if vector_store.enabled else {'enabled': False, 'total_documents': 0}

        # Derive anomaly/high-risk counts from the current project's Chroma collection
        # so they switch correctly when the user changes project (ui_state['stats'] is global)
        try:
            all_meta = vector_store.collection.get(include=['metadatas'])['metadatas']
            project_anomalies = sum(1 for m in all_meta if m.get('anomalies', '').strip())
            project_high_risk = sum(1 for m in all_meta if int(m.get('risk_score') or 0) >= 70)
        except Exception:
            project_anomalies = ui_state['stats'].get('anomalies_detected', 0)
            project_high_risk = ui_state['stats'].get('high_risk_count', 0)

        # Use the ChromaDB count for total_analyzed so the Overview dashboard
        # matches Manage Projects (both now reflect the same live Chroma count).
        # Fall back to SQLite processed_documents count when vector store is off.
        chroma_count = vector_stats.get('total_documents', project_analyzed) if vector_stats.get('enabled') else project_analyzed

        from analyzer.court_db import get_court_doc_count, get_pending_ocr_count
        try:
            court_count = get_court_doc_count(project_slug)
        except Exception:
            court_count = 0

        # Awaiting OCR: uploaded to Paperless (task_id recorded) but doc_id not yet resolved
        # Awaiting AI: doc is in Paperless with content, but not yet embedded in ChromaDB
        try:
            _pc = _get_project_client(project_slug)
            paperless_total = _pc.get_project_document_count(project_slug)
            awaiting_ocr = get_pending_ocr_count(project_slug)
            awaiting_ai = max(0, paperless_total - chroma_count - awaiting_ocr)
        except Exception:
            awaiting_ocr = awaiting_ai = 0

        return jsonify({
            'status': 'running',
            'uptime_seconds': _get_uptime(),
            'state': state_stats,
            'stats': {
                **ui_state['stats'],
                'total_analyzed': chroma_count,
                'anomalies_detected': project_anomalies,
                'high_risk_count': project_high_risk,
            },
            'last_update': ui_state['last_update'],
            'active_profiles': len(app.profile_loader.profiles),
            'vector_store': vector_stats,
            'court_doc_count': court_count,
            'awaiting_ocr': awaiting_ocr,
            'awaiting_ai': awaiting_ai,
        })


@app.route('/api/recent')
@login_required
def api_recent():
    """Get recent analysis results for the current project."""
    project_slug = session.get('current_project', 'default')

    # Query Chroma for the most recent docs in this project's collection.
    # This is always project-scoped and survives container restarts, unlike
    # the global in-memory ui_state['recent_analyses'].
    try:
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=project_slug)
        if vs.enabled:
            raw = vs.collection.get(include=['metadatas'])
            metas = raw.get('metadatas') or []
            # Sort by timestamp descending and take the 50 most recent
            metas_sorted = sorted(
                metas,
                key=lambda m: m.get('timestamp', ''),
                reverse=True
            )[:50]
            # Build per-project or global Paperless base URL for doc links
            try:
                _rcfg = app.project_manager.get_paperless_config(project_slug) if app.project_manager else {}
                _rbase = (_rcfg.get('doc_base_url') or '').rstrip('/')
            except Exception:
                _rbase = ''
            if not _rbase:
                _rbase = os.environ.get('PAPERLESS_PUBLIC_BASE_URL', '').rstrip('/')
            analyses = []
            for m in metas_sorted:
                anomalies_str = m.get('anomalies', '')
                anomalies_list = [a.strip() for a in anomalies_str.split(',') if a.strip()] if anomalies_str else []
                _rdid = m.get('document_id')
                analyses.append({
                    'document_id': _rdid,
                    'document_title': m.get('title', ''),
                    'anomalies_found': anomalies_list,
                    'risk_score': m.get('risk_score', 0),
                    'timestamp': m.get('timestamp', ''),
                    'brief_summary': m.get('brief_summary', ''),
                    'full_summary': m.get('full_summary', ''),
                    'ai_analysis': m.get('ai_analysis', ''),
                    'paperless_link': f"{_rbase}/documents/{_rdid}/details" if _rbase and _rdid else None,
                })
            return jsonify({'analyses': analyses})
    except Exception as e:
        logger.warning(f"Chroma recent query failed for {project_slug}, falling back: {e}")

    # Fallback: filter in-memory list by project_slug field
    with ui_state['lock']:
        filtered = [
            a for a in ui_state['recent_analyses']
            if a.get('project_slug', 'default') == project_slug
        ]
        return jsonify({'analyses': filtered[-50:]})


@app.route('/api/profiles')
@login_required
def api_profiles():
    """Get profile information."""
    active_profiles = []
    for profile in app.profile_loader.profiles:
        active_profiles.append({
            'id': profile.profile_id,
            'name': profile.display_name,
            'version': profile.version,
            'checks': profile.checks_enabled
        })

    # Get staging profiles
    staging_dir = Path('/app/profiles/staging')
    staging_profiles = []
    if staging_dir.exists():
        for profile_file in staging_dir.glob('*.yaml'):
            staging_profiles.append({
                'filename': profile_file.name,
                'created': datetime.fromtimestamp(profile_file.stat().st_mtime).isoformat(),
                'size': profile_file.stat().st_size
            })

    return jsonify({
        'active': active_profiles,
        'staging': staging_profiles
    })


@app.route('/api/staging/<filename>')
@login_required
def api_staging_profile(filename):
    """Get staging profile content."""
    staging_file = Path('/app/profiles/staging') / filename
    if not staging_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        import yaml
        with open(staging_file, 'r') as f:
            content = yaml.safe_load(f)
        return jsonify(content)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/staging/<filename>/activate', methods=['POST'])
@login_required
def api_activate_staging_profile(filename):
    """Activate a staging profile by moving it to active profiles."""
    staging_file = Path('/app/profiles/staging') / filename
    if not staging_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        # Move from staging to active
        active_file = Path('/app/profiles/active') / filename
        staging_file.rename(active_file)

        logger.info(f"Activated staging profile: {filename}")

        return jsonify({
            'success': True,
            'message': f'Profile "{filename}" activated! Restart the analyzer to load it.'
        })
    except Exception as e:
        logger.error(f"Failed to activate profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/staging/activate-all', methods=['POST'])
@login_required
def api_activate_all_staging_profiles():
    """Activate all staging profiles at once."""
    staging_dir = Path('/app/profiles/staging')
    if not staging_dir.exists():
        return jsonify({'error': 'Staging directory not found'}), 404

    results = {'success': [], 'failed': []}

    for profile_file in staging_dir.glob('*.yaml'):
        try:
            active_file = Path('/app/profiles/active') / profile_file.name
            profile_file.rename(active_file)
            results['success'].append(profile_file.name)
            logger.info(f"Activated staging profile: {profile_file.name}")
        except Exception as e:
            results['failed'].append({'filename': profile_file.name, 'error': str(e)})
            logger.error(f"Failed to activate {profile_file.name}: {e}")

    return jsonify({
        'success': True,
        'activated': len(results['success']),
        'failed': len(results['failed']),
        'details': results,
        'message': f"Activated {len(results['success'])} profiles. Restart analyzer to load them."
    })


@app.route('/api/staging/<filename>/delete', methods=['POST'])
@login_required
def api_delete_staging_profile(filename):
    """Delete a staging profile."""
    staging_file = Path('/app/profiles/staging') / filename
    if not staging_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        staging_file.unlink()
        logger.info(f"Deleted staging profile: {filename}")

        return jsonify({
            'success': True,
            'message': f'Profile "{filename}" deleted'
        })
    except Exception as e:
        logger.error(f"Failed to delete profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/<filename>', methods=['GET'])
@login_required
def api_get_active_profile(filename):
    """Get an active profile's content."""
    active_file = Path('/app/profiles/active') / filename
    if not active_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        import yaml
        with open(active_file, 'r') as f:
            profile_data = yaml.safe_load(f)
        return jsonify(profile_data)
    except Exception as e:
        logger.error(f"Failed to read profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/<filename>/rename', methods=['POST'])
@login_required
def api_rename_active_profile(filename):
    """Rename an active profile (update display_name)."""
    active_file = Path('/app/profiles/active') / filename
    if not active_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        import yaml
        data = request.json
        new_name = data.get('display_name', '').strip()

        if not new_name:
            return jsonify({'error': 'Display name required'}), 400

        # Load, update, save
        with open(active_file, 'r') as f:
            profile_data = yaml.safe_load(f)

        profile_data['display_name'] = new_name

        with open(active_file, 'w') as f:
            yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Renamed active profile {filename} to '{new_name}'")

        return jsonify({
            'success': True,
            'message': f'Profile renamed to "{new_name}"'
        })
    except Exception as e:
        logger.error(f"Failed to rename profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/<filename>/delete', methods=['POST'])
@login_required
def api_delete_active_profile(filename):
    """Delete an active profile."""
    active_file = Path('/app/profiles/active') / filename
    if not active_file.exists():
        return jsonify({'error': 'Profile not found'}), 404

    try:
        active_file.unlink()
        logger.info(f"Deleted active profile: {filename}")

        return jsonify({
            'success': True,
            'message': f'Profile "{filename}" deleted'
        })
    except Exception as e:
        logger.error(f"Failed to delete profile {filename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/duplicates', methods=['GET'])
@login_required
def api_detect_duplicates():
    """Detect duplicate profiles in active directory."""
    active_dir = Path('/app/profiles/active')
    if not active_dir.exists():
        return jsonify({'duplicates': [], 'groups': []}), 200

    try:
        import yaml
        import hashlib
        from collections import defaultdict

        profiles = []

        # Load all active profiles
        for profile_file in active_dir.glob('*.yaml'):
            if profile_file.name in ['active', 'examples']:
                continue

            try:
                with open(profile_file, 'r') as f:
                    content = f.read()
                    profile_data = yaml.safe_load(content)

                profiles.append({
                    'filename': profile_file.name,
                    'profile_id': profile_data.get('profile_id', ''),
                    'display_name': profile_data.get('display_name', ''),
                    'content': content,
                    'content_hash': hashlib.md5(content.encode()).hexdigest(),
                    'keywords': set(profile_data.get('match', {}).get('keywords', {}).get('any', [])),
                    'checks': set(profile_data.get('checks_enabled', []))
                })
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_file.name}: {e}")
                continue

        # Find duplicates using Union-Find to group similar profiles
        duplicate_groups = []

        # 1. Exact content duplicates
        content_map = defaultdict(list)
        for profile in profiles:
            content_map[profile['content_hash']].append(profile)

        for content_hash, group in content_map.items():
            if len(group) > 1:
                duplicate_groups.append({
                    'type': 'exact',
                    'reason': 'Identical file content',
                    'profiles': [{'filename': p['filename'], 'display_name': p['display_name']} for p in group]
                })

        # 2. Similar profiles (high keyword overlap) - use Union-Find to group properly
        # Build similarity graph
        similar_pairs = []
        exact_hashes = {p['content_hash'] for group in content_map.values() if len(group) > 1 for p in group}

        for i, p1 in enumerate(profiles):
            # Skip profiles already in exact duplicate groups
            if p1['content_hash'] in exact_hashes:
                continue

            for p2 in profiles[i+1:]:
                # Skip if same profile or already in exact duplicate
                if p1['content_hash'] == p2['content_hash'] or p2['content_hash'] in exact_hashes:
                    continue

                # Calculate keyword similarity
                if p1['keywords'] and p2['keywords']:
                    intersection = len(p1['keywords'] & p2['keywords'])
                    union = len(p1['keywords'] | p2['keywords'])
                    similarity = intersection / union if union > 0 else 0

                    # If >70% similar keywords and same checks, flag as similar
                    if similarity > 0.7 and p1['checks'] == p2['checks']:
                        similar_pairs.append((i, profiles[i+1:].index(p2) + i + 1, similarity))

        # Use Union-Find to group similar profiles
        if similar_pairs:
            parent = {i: i for i in range(len(profiles))}

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Union similar profiles
            for i, j, sim in similar_pairs:
                union(i, j)

            # Group profiles by their root parent
            groups = defaultdict(list)
            for i, profile in enumerate(profiles):
                if profile['content_hash'] not in exact_hashes:  # Skip exact duplicates
                    root = find(i)
                    groups[root].append((profile, i))

            # Create duplicate groups for similar profiles
            for root, group_profiles in groups.items():
                if len(group_profiles) > 1:
                    # Calculate average similarity
                    avg_sim = sum(sim for i, j, sim in similar_pairs
                                 if find(i) == root and find(j) == root) / max(len(similar_pairs), 1)

                    duplicate_groups.append({
                        'type': 'similar',
                        'reason': f'~{int(avg_sim * 100)}% keyword overlap, same checks',
                        'profiles': [{'filename': p['filename'], 'display_name': p['display_name']}
                                   for p, _ in group_profiles]
                    })

        return jsonify({
            'total_profiles': len(profiles),
            'duplicate_groups': len(duplicate_groups),
            'groups': duplicate_groups
        })

    except Exception as e:
        logger.error(f"Failed to detect duplicates: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/active/duplicates/remove', methods=['POST'])
@login_required
def api_remove_duplicates():
    """Remove specified duplicate profiles."""
    try:
        data = request.json
        filenames = data.get('filenames', [])

        if not filenames:
            return jsonify({'error': 'No filenames provided'}), 400

        active_dir = Path('/app/profiles/active')
        removed = []
        failed = []

        for filename in filenames:
            profile_file = active_dir / filename
            if not profile_file.exists():
                failed.append({'filename': filename, 'error': 'File not found'})
                continue

            try:
                profile_file.unlink()
                removed.append(filename)
                logger.info(f"Removed duplicate profile: {filename}")
            except Exception as e:
                failed.append({'filename': filename, 'error': str(e)})
                logger.error(f"Failed to remove {filename}: {e}")

        # Auto-reload profiles after removal
        if removed and hasattr(app, 'profile_loader'):
            try:
                app.profile_loader.load_profiles()
                logger.info(f"Profiles reloaded after removing {len(removed)} duplicates")
                reload_msg = " Profiles automatically reloaded."
            except Exception as e:
                logger.error(f"Failed to reload profiles: {e}")
                reload_msg = " Please use the 'Reload Profiles' button to refresh."
        else:
            reload_msg = ""

        return jsonify({
            'success': True,
            'removed': len(removed),
            'failed': len(failed),
            'removed_files': removed,
            'failed_files': failed,
            'message': f'Removed {len(removed)} duplicate profiles.{reload_msg}'
        })

    except Exception as e:
        logger.error(f"Failed to remove duplicates: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/reload-profiles', methods=['POST'])
@login_required
def api_reload_profiles():
    """Reload all active profiles without restarting the container."""
    try:
        if not hasattr(app, 'profile_loader'):
            return jsonify({'error': 'Profile loader not available'}), 500

        # Get count before reload
        old_count = len(app.profile_loader.profiles)

        # Reload profiles from disk
        app.profile_loader.load_profiles()

        # Get count after reload
        new_count = len(app.profile_loader.profiles)

        logger.info(f"Profiles reloaded: {old_count} → {new_count}")

        return jsonify({
            'success': True,
            'message': f'Profiles reloaded successfully',
            'old_count': old_count,
            'new_count': new_count,
            'profiles': [p.profile_id for p in app.profile_loader.profiles]
        })

    except Exception as e:
        logger.error(f"Failed to reload profiles: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def _vision_extract_doc(doc_id: int, title: str, paperless_client, ai_config: dict) -> str:
    """
    Extract text from a Paperless document using Vision AI (GPT-4o or Claude Vision).
    Downloads the archived PDF, converts each page to a PNG, and sends to the configured
    LLM with vision capability.  Returns concatenated page text, or '' on any failure.
    Used during AI chat to enrich documents whose OCR content is empty or too short.
    """
    try:
        import base64
        from io import BytesIO

        pdf_bytes = paperless_client.download_document(doc_id, archived=True)
        if not pdf_bytes or len(pdf_bytes) < 200:
            return ''

        # Build list of base64-encoded page PNGs
        page_images = []
        try:
            from pdf2image import convert_from_bytes
            imgs = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=8)
            for img in imgs:
                buf = BytesIO()
                img.save(buf, format='PNG')
                page_images.append(base64.b64encode(buf.getvalue()).decode())
        except ImportError:
            # pdf2image not installed — treat bytes as a raw image
            page_images.append(base64.b64encode(pdf_bytes).decode())
        except Exception:
            page_images.append(base64.b64encode(pdf_bytes).decode())

        if not page_images:
            return ''

        vision_prompt = (
            "Extract ALL text from this document page accurately. "
            "Include all numbers, dates, names, headings, account numbers, addresses, "
            "and table data (format table columns separated by |). "
            "Output the raw extracted text only — no commentary."
        )

        # Pick the first enabled provider that has an API key
        provider_name, api_key = None, None
        for p in ai_config.get('chat', {}).get('providers', []):
            if p.get('enabled') and p.get('api_key', '').strip():
                provider_name = p['name']
                api_key = p['api_key'].strip()
                break

        if not provider_name or not api_key:
            return ''

        extracted_pages = []
        for i, img_b64 in enumerate(page_images[:8]):
            try:
                page_text = ''
                if provider_name == 'openai':
                    import openai as _openai
                    oc = _openai.OpenAI(api_key=api_key)
                    resp = oc.chat.completions.create(
                        model='gpt-4o',
                        max_tokens=2000,
                        messages=[{'role': 'user', 'content': [
                            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}},
                            {'type': 'text', 'text': vision_prompt}
                        ]}]
                    )
                    page_text = resp.choices[0].message.content or ''
                elif provider_name == 'anthropic':
                    import anthropic as _anthropic
                    ac = _anthropic.Anthropic(api_key=api_key)
                    resp = ac.messages.create(
                        model='claude-3-5-sonnet-20241022',
                        max_tokens=2000,
                        messages=[{'role': 'user', 'content': [
                            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': img_b64}},
                            {'type': 'text', 'text': vision_prompt}
                        ]}]
                    )
                    page_text = resp.content[0].text if resp.content else ''
                if page_text:
                    extracted_pages.append(f"[Page {i + 1}]\n{page_text}")
            except Exception as page_err:
                logger.warning(f"Vision AI page {i + 1} failed for doc {doc_id}: {page_err}")

        result = '\n\n'.join(extracted_pages)
        logger.info(f"Vision AI extracted {len(result)} chars from doc {doc_id} ({len(page_images)} pages)")
        return result

    except Exception as e:
        logger.warning(f"_vision_extract_doc failed for doc {doc_id} ({title}): {e}")
        return ''


@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    """Chat with AI about documents using RAG (semantic search)."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        document_type = data.get('document_type', None)  # Optional filter by document type
        session_id = data.get('session_id', None)

        if not user_message:
            return jsonify({'error': 'Message required'}), 400

        # Resolve or create a chat session
        if session_id:
            sess = get_session(session_id)
            if not sess or not can_access_session(session_id, current_user.id):
                session_id = None  # Fall through to create new
        if not session_id:
            session_id = create_session(
                current_user.id,
                title='New Chat',
                document_type=document_type,
            )

        # Get stats
        with ui_state['lock']:
            stats = ui_state['stats']

        # Use semantic search with vector store — scoped to current project
        from analyzer.vector_store import VectorStore
        _chat_project = session.get('current_project', 'default')
        vector_store = VectorStore(project_slug=_chat_project)

        logger.info(f"Vector store enabled: {vector_store.enabled}")

        if vector_store.enabled:
            vector_stats = vector_store.get_stats()
            logger.info(f"Vector store stats: {vector_stats}")

            # Semantic search for relevant documents
            # Use more results for summary queries, fewer for specific queries
            n_results = 30 if any(word in user_message.lower() for word in ['summary', 'all', 'comprehensive', 'total', 'overview']) else 15

            # Apply document type filter if specified
            if document_type and document_type != 'all':
                logger.info(f"Filtering by document type: {document_type}")
                relevant_docs = vector_store.search(user_message, n_results=n_results, document_type=document_type)
            else:
                relevant_docs = vector_store.search(user_message, n_results=n_results)

            logger.info(f"RAG: Search returned {len(relevant_docs)} documents")

            if relevant_docs:
                recent_analyses = relevant_docs
                logger.info(f"RAG: Using {len(relevant_docs)} semantically relevant documents")
            else:
                # If search returns nothing, fetch from Paperless as fallback
                logger.warning("RAG: No documents found via semantic search, falling back to Paperless query")
                with ui_state['lock']:
                    recent_analyses = ui_state['recent_analyses']
        else:
            logger.warning("Vector store not enabled, using in-memory analyses")
            # Fallback: fetch from memory or Paperless
            with ui_state['lock']:
                recent_analyses = ui_state['recent_analyses']

        # ── Vision AI content enrichment ────────────────────────────────────────
        # For each retrieved doc whose stored content is too short (empty OCR at
        # embed time), first try a fresh Paperless OCR fetch, then fall back to
        # Vision AI.  Cap Vision AI at 2 docs per query to limit latency.
        vision_ai_used = []
        if recent_analyses and hasattr(app, 'paperless_client'):
            _ai_cfg_vision = load_ai_config()
            vision_cap = 0
            enriched = []
            for _a in recent_analyses:
                _content = _a.get('content', _a.get('ai_analysis', ''))
                _doc_id  = _a.get('document_id')
                _doc_title = _a.get('document_title', '')
                if len(_content.strip()) < 500 and _doc_id:
                    try:
                        _fresh = app.paperless_client.get_document(int(_doc_id))
                        _fresh_text = _fresh.get('content', '').strip()
                        if len(_fresh_text) >= 200:
                            _a = dict(_a)
                            _a['content'] = f"Document Content (Paperless OCR — live fetch):\n{_fresh_text}"
                            logger.info(f"Chat enrichment: refreshed OCR for doc {_doc_id} ({len(_fresh_text)} chars)")
                        elif vision_cap < 2:
                            # OCR still short — try Vision AI
                            logger.info(f"Chat enrichment: running Vision AI on doc {_doc_id}")
                            _vtext = _vision_extract_doc(int(_doc_id), _doc_title, app.paperless_client, _ai_cfg_vision)
                            if _vtext and len(_vtext) > 200:
                                _a = dict(_a)
                                _a['content'] = f"Document Content (Vision AI — extracted during this chat):\n{_vtext}"
                                vision_ai_used.append(_doc_title or str(_doc_id))
                                vision_cap += 1
                                # Re-embed in background so future queries benefit
                                def _bg_reembed(_did=int(_doc_id), _dtitle=_doc_title, _dtext=_vtext,
                                                _dtype=_a.get('document_type', 'unknown'),
                                                _drisk=_a.get('risk_score', 0)):
                                    try:
                                        if app.document_analyzer and app.document_analyzer.vector_store:
                                            app.document_analyzer.vector_store.embed_document(
                                                _did, _dtitle, _dtext,
                                                {'risk_score': _drisk, 'anomalies': [],
                                                 'timestamp': datetime.utcnow().isoformat(),
                                                 'paperless_modified': '', 'document_type': _dtype}
                                            )
                                            logger.info(f"Chat Vision AI: re-embedded doc {_did}")
                                    except Exception as _re:
                                        logger.warning(f"Chat Vision AI: re-embed failed for doc {_did}: {_re}")
                                Thread(target=_bg_reembed, daemon=True).start()
                    except Exception as _ee:
                        logger.warning(f"Chat enrichment failed for doc {_doc_id}: {_ee}")
                enriched.append(_a)
            recent_analyses = enriched
        # ────────────────────────────────────────────────────────────────────────

        # If we don't have analyses, fetch from Paperless
        # Only fall back if Chroma returned nothing — do NOT use < 5 threshold
        # because small projects legitimately have few docs and the fallback
        # is not project-scoped (it returns all analyzed docs from Paperless).
        if not recent_analyses:
            try:
                # Get documents with analyzed tags
                paperless_client = app.paperless_client
                documents = paperless_client.session.get(
                    f'{paperless_client.base_url}/api/documents/',
                    params={
                        'tags__name__icontains': 'analyzed',
                        'ordering': '-modified',
                        'page_size': 50
                    }
                ).json()

                recent_analyses = []
                for doc in documents.get('results', [])[:20]:  # Limit to 20 for performance
                    doc_id = doc['id']

                    # Get full document details including notes
                    try:
                        full_doc = paperless_client.get_document(doc_id)
                        notes = full_doc.get('notes', '')

                        # Extract AI analysis from notes (it's the section after "🤖 AI ANOMALY ANALYSIS")
                        ai_analysis = ""
                        if "🤖 AI ANOMALY ANALYSIS" in notes:
                            # Get the latest AI analysis (last occurrence)
                            parts = notes.split("🤖 AI ANOMALY ANALYSIS")
                            if len(parts) > 1:
                                ai_analysis = parts[-1].split("---")[0].strip()[:1000]  # First 1000 chars
                    except:
                        notes = ""
                        ai_analysis = ""

                    # Extract anomaly tags
                    tags = [paperless_client.session.get(
                        f'{paperless_client.base_url}/api/tags/{tag_id}/'
                    ).json().get('name', '') for tag_id in doc.get('tags', [])[:10]]

                    anomalies = [t.replace('anomaly:', '') for t in tags if t.startswith('anomaly:')]

                    # Determine risk score from tags
                    risk_score = 0
                    if 'anomaly:forensic_risk_high' in tags:
                        risk_score = 80
                    elif 'anomaly:forensic_risk_medium' in tags:
                        risk_score = 60
                    elif 'anomaly:forensic_risk_low' in tags:
                        risk_score = 30

                    # Generate AI-powered comparative summary if LLM is available
                    brief_summary = ""
                    full_summary = ""
                    try:
                        if app.document_analyzer and app.document_analyzer.llm_client:
                            # Query vector store for similar documents
                            similar_docs = []
                            if app.document_analyzer.vector_store:
                                try:
                                    # Search for similar documents using title as query
                                    results = app.document_analyzer.vector_store.query(
                                        query_text=doc['title'],
                                        n_results=6  # Get 6 to exclude self
                                    )
                                    if results and 'documents' in results:
                                        # Filter out the current document and format results
                                        for i, result_doc in enumerate(results.get('documents', [[]])[0]):
                                            metadata = results.get('metadatas', [[]])[0][i] if i < len(results.get('metadatas', [[]])[0]) else {}
                                            result_id = metadata.get('document_id', '')
                                            if result_id and str(result_id) != str(doc_id):
                                                similar_docs.append({
                                                    'id': result_id,
                                                    'title': metadata.get('title', 'Unknown'),
                                                    'created': metadata.get('created', 'unknown'),
                                                    'similarity': results.get('distances', [[]])[0][i] if i < len(results.get('distances', [[]])[0]) else 0
                                                })
                                except Exception as ve:
                                    logger.debug(f"Vector search failed for doc {doc_id}: {ve}")

                            # Generate comparative summary
                            doc_info = {
                                'id': doc_id,
                                'title': doc['title'],
                                'document_type': 'financial document',
                                'created': doc.get('created', ''),
                            }
                            summary = app.document_analyzer.llm_client.generate_comparative_summary(
                                doc_info,
                                content_preview=notes[:500] if notes else "",
                                similar_documents=similar_docs
                            )
                            brief_summary = summary.get('brief', '')
                            full_summary = summary.get('full', '')
                    except Exception as sum_err:
                        logger.debug(f"Failed to generate summary for doc {doc_id}: {sum_err}")
                        brief_summary = f"Financial document: {doc['title']}"
                        full_summary = brief_summary

                    recent_analyses.append({
                        'document_id': doc_id,
                        'document_title': doc['title'],
                        'anomalies_found': anomalies[:5],
                        'risk_score': risk_score,
                        'timestamp': doc['modified'],
                        'ai_analysis': ai_analysis,
                        'created': doc.get('created', ''),
                        'correspondent': doc.get('correspondent', None),
                        'brief_summary': brief_summary,
                        'full_summary': full_summary
                    })

                logger.info(f"Fetched {len(recent_analyses)} analyzed documents from Paperless")
            except Exception as e:
                logger.error(f"Failed to fetch documents from Paperless: {e}")

        # Build context for AI
        context = f"""You are an AI assistant helping analyze financial documents.

Current Statistics:
- Total documents analyzed: {stats.get('total_analyzed', 0)}
- Anomalies detected: {stats.get('anomalies_detected', 0)}
- High risk documents: {stats.get('high_risk_count', 0)}

Recent Analyses:
"""
        for analysis in recent_analyses[-20:]:  # Last 20 analyses
            doc_id = analysis.get('document_id', 'Unknown')
            doc_title = analysis.get('document_title', 'Unknown')
            anomalies = analysis.get('anomalies_found', [])
            risk = analysis.get('risk_score', 0)
            context += f"\n- [Document #{doc_id}]: {doc_title}"
            if anomalies:
                context += f" | Anomalies: {', '.join(anomalies)}"
            context += f" | Risk: {risk}%"

        context += f"""

User's question: {user_message}

Provide a helpful, data-driven response based on the actual document content available."""

        # Build conversation history
        messages = []
        for msg in history[-10:]:  # Last 10 messages
            if msg.get('role') in ['user', 'assistant']:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        # Add current user message
        messages.append({'role': 'user', 'content': user_message})

        # Use system parameter for context (Claude's preferred way)
        rag_mode = vector_store.enabled if 'vector_store' in locals() else False

        total_in_vector_store = vector_stats.get('total_documents', 0) if 'vector_stats' in locals() else 0

        # Build document type context and determine document count for prompt
        type_filter_text = ""
        filtered_count = total_in_vector_store
        filter_mode_text = "ALL documents"

        if document_type and document_type != 'all':
            type_breakdown = vector_stats.get('by_type', {}) if 'vector_stats' in locals() else {}
            filtered_count = type_breakdown.get(document_type, 0)
            type_filter_text = f"\n- **SEARCH IS FILTERED**: Only searching '{document_type}' documents"
            filter_mode_text = f"documents of type '{document_type}'"

        system_prompt = f"""You are an AI assistant analyzing financial documents in a document management system.

{'[RAG MODE: Semantic search active across ' + filter_mode_text + ']' if rag_mode else '[Standard mode]'}

CRITICAL - YOU HAVE ACCESS TO:
- **{filtered_count} documents** available for this query{type_filter_text}
- Below are the {len(recent_analyses)} MOST RELEVANT documents for this specific query
- Database contains {total_in_vector_store} total documents across all types

Current Statistics (for available documents):
- Documents in current scope: {filtered_count}
- Total across all types: {total_in_vector_store}
- Anomalies detected across all docs: {stats.get('anomalies_detected', 0)}
- High risk documents: {stats.get('high_risk_count', 0)}

{'Most Relevant Documents for This Query (from ' + str(filtered_count) + ' available):' if rag_mode else 'Recent Document Analyses:'}
"""
        if recent_analyses:
            for analysis in recent_analyses[-20:]:
                doc_id = analysis.get('document_id', 'Unknown')
                doc_title = analysis.get('document_title', 'Unknown')
                anomalies = analysis.get('anomalies_found', [])
                risk = analysis.get('risk_score', 0)
                timestamp = analysis.get('timestamp', 'Unknown')

                # Get content - vector store uses 'content', fallback uses 'ai_analysis'
                content = analysis.get('content', analysis.get('ai_analysis', ''))

                system_prompt += f"\n\n--- [Document #{doc_id}] ---"
                system_prompt += f"\nTitle: {doc_title}"
                system_prompt += f"\nRisk Score: {risk}%"
                if anomalies:
                    system_prompt += f"\nAnomalies: {', '.join(anomalies)}"
                system_prompt += f"\nAnalyzed: {timestamp}"

                # Include full document content
                if content:
                    system_prompt += f"\n\nFull Document Analysis:\n{content}"
        else:
            system_prompt += "\n(No documents analyzed yet)"

        system_prompt += """

VISION AI CAPABILITY:
- Documents whose stored OCR was empty or too short have been enriched LIVE during this chat.
- Content marked "Vision AI — extracted during this chat" was read directly from the original PDF pages using image recognition — treat it as authoritative.
- Content marked "Paperless OCR — live fetch" was retrieved fresh from Paperless at query time.
- You CAN and SHOULD analyze the content in these enriched documents fully.

CRITICAL - NEVER HALLUCINATE DATA:
- NEVER invent dollar amounts, totals, dates, names, or any data not explicitly present in the content shown above.
- Only report numbers and facts that are EXPLICITLY stated in the document content provided above.
- If content is still empty or very short after enrichment, say: "This document's content could not be extracted even with Vision AI. I cannot analyze specific figures without the source file being accessible."
- Do NOT claim you lack access to PDFs — Vision AI has already been applied where needed.

DOCUMENT REFERENCES: When mentioning any document, ALWAYS use the exact format [Document #NNN] where NNN is the document ID. This enables clickable links. Never write "Doc NNN" without the brackets and #.

IMPORTANT INSTRUCTIONS:
When users ask for summaries or "all documents":
- You SHOULD provide comprehensive analysis based on the documents shown above
- These are the most relevant documents selected via semantic search
- Frame responses as "Based on analysis of [total] documents..."
- Provide statistics, patterns, and insights from the documents shown
- Be specific with numbers, document IDs, and findings
- DO NOT say you can't access documents - you have them above

When users ask specific questions:
- Reference the relevant documents from those shown above
- Provide data-driven insights based on actual analyses
- Be specific with document IDs, titles, risk scores, and anomalies
- ONLY use information explicitly present in the document content above

FORMATTING REQUIREMENTS:
- Use markdown formatting for better readability
- Use bullet points (-) for lists
- Use **bold** for important information
- Use tables when presenting structured data
- Use line breaks between sections
- Example table format:
  | Column 1 | Column 2 |
  |----------|----------|
  | Data 1   | Data 2   |

LEDGER AND REPORT GENERATION:
- When asked to generate ledgers, reports, or summaries, use ALL available data from documents
- Extract account numbers, balances, dates, and transaction details from document content
- Create tables with available information, clearly noting any gaps or missing data
- If some documents lack certain information, work with what's available and note limitations
- Provide the most complete analysis possible given the data you can access

HELP & DOCUMENTATION:
When a user asks how to do something in the interface, or asks about a feature, include a helpful link to the user manual. Use markdown link format.
Available documentation pages (prepend the app URL prefix):
- Overview & feature list: /docs/overview
- Quick start guide: /docs/getting-started
- Projects & workspaces: /docs/projects
- Smart Upload (file/URL/cloud): /docs/upload
- AI Chat usage: /docs/chat
- Search & Analysis: /docs/search
- Anomaly detection tags: /docs/anomaly-detection
- Debug & Tools (reprocess, logs): /docs/tools
- Configuration (AI keys, profiles, SMTP): /docs/configuration
- User management: /docs/users
- LLM usage & cost tracking: /docs/llm-usage
- API reference: /docs/api
Example: "You can learn more about projects in the [Projects documentation](/docs/projects)."
Only include a docs link when it is genuinely relevant to the user's question."""

        if vision_ai_used:
            system_prompt += f"\n\n[Note: Vision AI was used during this query to extract content from {len(vision_ai_used)} document(s) with poor OCR: {', '.join(vision_ai_used[:3])}. Their embeddings have been updated for future queries.]"

        # Load AI configuration — v2 format: per-project primary/fallback, global keys as fallback
        project_slug = session.get('current_project', 'default')
        chat_cfg = get_project_ai_config(project_slug, 'chat')
        _full_cfg = load_ai_config()

        def _global_key(provider_name):
            return _full_cfg.get('global', {}).get(provider_name, {}).get('api_key', '').strip()

        providers = []
        prov = chat_cfg.get('provider', 'openai')
        pkey = (chat_cfg.get('api_key') or '').strip() or _global_key(prov)
        if pkey:
            providers.append({'name': prov, 'enabled': True, 'api_key': pkey,
                              'models': [chat_cfg.get('model', 'gpt-4o')]})
        fb_prov = chat_cfg.get('fallback_provider')
        fb_model = chat_cfg.get('fallback_model')
        if fb_prov and fb_model and fb_prov != prov:
            fb_key = _global_key(fb_prov)
            if fb_key:
                providers.append({'name': fb_prov, 'enabled': True, 'api_key': fb_key,
                                  'models': [fb_model]})

        ai_response = None
        last_error = None
        attempted = []

        # Try each enabled provider and their models
        for provider_config in providers:
            if not provider_config.get('enabled', False):
                continue

            provider_name = provider_config.get('name')
            api_key = provider_config.get('api_key', '').strip()
            models = provider_config.get('models', [])

            if not api_key:
                logger.warning(f"Skipping {provider_name}: No API key configured")
                continue

            # Initialize provider client
            try:
                if provider_name == 'openai':
                    import openai
                    client = openai.OpenAI(api_key=api_key)

                    # Try each model for this provider
                    for model in models:
                        try:
                            logger.info(f"Trying chat: OpenAI {model}")
                            attempted.append(f"OpenAI {model}")

                            response = client.chat.completions.create(
                                model=model,
                                messages=[{"role": "system", "content": system_prompt}] + messages,
                                max_tokens=4096  # Increased for legal/court documents - comprehensive answers
                            )
                            ai_response = response.choices[0].message.content
                            logger.info(f"✓ Successfully used: OpenAI {model}")
                            break
                        except Exception as e:
                            error_str = str(e)
                            if '404' in error_str or 'model_not_found' in error_str:
                                logger.warning(f"Model {model} not available, trying next...")
                                last_error = e
                                continue
                            else:
                                logger.warning(f"Error with OpenAI {model}: {e}")
                                last_error = e
                                continue

                elif provider_name == 'anthropic':
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)

                    # Try each model for this provider
                    for model in models:
                        try:
                            logger.info(f"Trying chat: Anthropic {model}")
                            attempted.append(f"Anthropic {model}")

                            response = client.messages.create(
                                model=model,
                                max_tokens=4096,  # Increased for legal/court documents - comprehensive answers
                                system=system_prompt,
                                messages=messages
                            )
                            ai_response = response.content[0].text
                            logger.info(f"✓ Successfully used: Anthropic {model}")
                            break
                        except Exception as e:
                            error_str = str(e)
                            if '404' in error_str or 'not_found' in error_str:
                                logger.warning(f"Model {model} not available, trying next...")
                                last_error = e
                                continue
                            else:
                                logger.warning(f"Error with Anthropic {model}: {e}")
                                last_error = e
                                continue

                if ai_response:
                    break  # Success, stop trying providers

            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} client: {e}")
                last_error = e
                continue

        if ai_response is None:
            if not attempted:
                raise Exception("No AI API key configured. Please go to the Configuration tab → AI Configuration and add an API key.")
            attempted_str = ", ".join(attempted)
            raise Exception(f"No available models responded. Tried: {attempted_str}. Last error: {last_error}")

        logger.info(f"Chat query: {user_message[:100]}")

        # Persist messages to the session
        user_msg_id = append_message(session_id, 'user', user_message)
        append_message(session_id, 'assistant', ai_response)

        # Auto-title: set from first user message if still 'New Chat'
        current_sess = get_session(session_id)
        if current_sess and current_sess['title'] == 'New Chat':
            auto_title = user_message[:60].strip()
            if auto_title:
                update_session_title(session_id, auto_title)

        return jsonify({
            'response': ai_response,
            'success': True,
            'session_id': session_id,
            'user_message_id': user_msg_id,
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/compare', methods=['POST'])
@login_required
def api_chat_compare():
    """Call both configured LLM providers in parallel and return both responses."""
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        data = request.json or {}
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Message required'}), 400

        # Resolve session
        session_id = data.get('session_id') or None
        if session_id:
            sess = get_session(session_id)
            if not sess or not can_access_session(session_id, current_user.id):
                session_id = None
        if not session_id:
            session_id = create_session(current_user.id, title='New Chat')

        # Build provider list (same logic as api_chat)
        project_slug = session.get('current_project', 'default')
        chat_cfg = get_project_ai_config(project_slug, 'chat')
        _full_cfg = load_ai_config()

        def _global_key(pname):
            return _full_cfg.get('global', {}).get(pname, {}).get('api_key', '').strip()

        providers = []
        prov = chat_cfg.get('provider', 'openai')
        pkey = (chat_cfg.get('api_key') or '').strip() or _global_key(prov)
        if pkey:
            providers.append({'name': prov, 'api_key': pkey,
                              'model': chat_cfg.get('model', 'gpt-4o')})
        fb_prov = chat_cfg.get('fallback_provider')
        fb_model = chat_cfg.get('fallback_model')
        if fb_prov and fb_model and fb_prov != prov:
            fb_key = _global_key(fb_prov)
            if fb_key:
                providers.append({'name': fb_prov, 'api_key': fb_key, 'model': fb_model})

        if len(providers) < 2:
            return jsonify({'error': 'Two configured AI providers are required for compare mode. Please add both a primary and a fallback provider in AI Configuration.'}), 400

        # Build messages list (last 10 from history)
        history = data.get('history', [])
        messages = [{'role': m['role'], 'content': m['content']}
                    for m in history[-10:] if m.get('role') in ('user', 'assistant')]
        messages.append({'role': 'user', 'content': user_message})

        # Build system prompt (reuse simple version without full RAG for speed)
        system_prompt = "You are an AI assistant helping analyze documents. Answer helpfully and accurately. When mentioning any document, ALWAYS use the format [Document #NNN] to enable clickable links."

        def _call_provider(pconf):
            name = pconf['name']
            key = pconf['api_key']
            model = pconf['model']
            try:
                if name == 'openai':
                    import openai as _oai
                    client = _oai.OpenAI(api_key=key)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt}] + messages,
                        max_tokens=4096
                    )
                    return name, resp.choices[0].message.content, None
                elif name == 'anthropic':
                    import anthropic as _ant
                    client = _ant.Anthropic(api_key=key)
                    resp = client.messages.create(
                        model=model, max_tokens=4096,
                        system=system_prompt, messages=messages
                    )
                    return name, resp.content[0].text, None
                else:
                    return name, None, f"Unsupported provider: {name}"
            except Exception as e:
                return name, None, str(e)

        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futs = {executor.submit(_call_provider, p): p['name'] for p in providers[:2]}
            for fut in as_completed(futs):
                name, text, err = fut.result()
                results[name] = {'text': text, 'error': err}

        primary = providers[0]
        secondary = providers[1]
        prim_name = primary['name']
        sec_name = secondary['name']
        prim_result = results.get(prim_name, {})
        sec_result = results.get(sec_name, {})

        primary_response = prim_result.get('text') or f"Error: {prim_result.get('error', 'No response')}"
        secondary_response = sec_result.get('text') or f"Error: {sec_result.get('error', 'No response')}"
        secondary_error = bool(sec_result.get('error'))

        # Save primary response to DB for conversation continuity
        user_msg_id = append_message(session_id, 'user', user_message)
        append_message(session_id, 'assistant', primary_response)

        # Auto-title
        cur_sess = get_session(session_id)
        if cur_sess and cur_sess['title'] == 'New Chat':
            update_session_title(session_id, user_message[:60].strip())

        return jsonify({
            'primary_provider': prim_name.capitalize(),
            'primary_response': primary_response,
            'secondary_provider': sec_name.capitalize(),
            'secondary_response': secondary_response,
            'secondary_error': secondary_error,
            'session_id': session_id,
            'user_message_id': user_msg_id,
        })
    except Exception as e:
        logger.error(f"Compare chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/types', methods=['GET'])
@login_required
def api_vector_types():
    """Get list of all document types in vector store."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        document_types = vector_store.get_document_types()
        stats = vector_store.get_stats()

        return jsonify({
            'success': True,
            'document_types': document_types,
            'by_type': stats.get('by_type', {})
        })

    except Exception as e:
        logger.error(f"Failed to get document types: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/delete/<int:document_id>', methods=['POST'])
@login_required
def api_vector_delete_document(document_id):
    """Delete a specific document from vector store."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        success = vector_store.delete_document(document_id)

        if success:
            return jsonify({
                'success': True,
                'message': f'Document {document_id} deleted from vector store'
            })
        else:
            return jsonify({'error': 'Failed to delete document'}), 500

    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/delete-document', methods=['POST'])
@login_required
def api_vector_delete_document_json():
    """Delete a specific document from vector store (JSON body)."""
    try:
        data = request.json
        doc_id = data.get('doc_id')

        if not doc_id:
            return jsonify({'error': 'doc_id required'}), 400

        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        success = vector_store.delete_document(int(doc_id))

        if success:
            return jsonify({
                'success': True,
                'message': f'Document {doc_id} deleted from vector store'
            })
        else:
            return jsonify({'error': 'Failed to delete document'}), 500

    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/delete-by-type', methods=['POST'])
@login_required
def api_vector_delete_by_type():
    """Delete all documents of a specific type from vector store."""
    try:
        data = request.json
        document_type = data.get('document_type', '').strip()

        if not document_type:
            return jsonify({'error': 'document_type required'}), 400

        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        count = vector_store.delete_by_type(document_type)

        return jsonify({
            'success': True,
            'message': f'Deleted {count} documents of type "{document_type}"',
            'count': count
        })

    except Exception as e:
        logger.error(f"Failed to delete documents by type: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/clear', methods=['POST'])
@login_required
def api_vector_clear():
    """Clear all documents from vector store."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        success = vector_store.clear()

        if success:
            return jsonify({
                'success': True,
                'message': 'Vector store cleared successfully'
            })
        else:
            return jsonify({'error': 'Failed to clear vector store'}), 500

    except Exception as e:
        logger.error(f"Failed to clear vector store: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/documents', methods=['GET'])
@login_required
def api_vector_documents():
    """Get all documents from vector store with details, grouped by type."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        # Get all documents from ChromaDB
        try:
            all_docs = vector_store.collection.get(include=['metadatas'])

            # Group by document type
            documents_by_type = {}
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i]
                doc_type = metadata.get('document_type', 'unknown')

                if doc_type not in documents_by_type:
                    documents_by_type[doc_type] = []

                documents_by_type[doc_type].append({
                    'doc_id': metadata.get('document_id'),
                    'title': metadata.get('title', 'Unknown'),
                    'risk_score': metadata.get('risk_score', 0),
                    'timestamp': metadata.get('timestamp', '')
                })

            # Sort documents within each type by doc_id
            for doc_type in documents_by_type:
                documents_by_type[doc_type].sort(key=lambda x: x['doc_id'])

            return jsonify({
                'success': True,
                'documents': documents_by_type
            })

        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"Failed to get vector documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/reembed-stale', methods=['POST'])
@login_required
def api_vector_reembed_stale():
    """Trigger a stale-embedding check: re-analyze docs whose OCR content changed after embedding."""
    try:
        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Analyzer not running'}), 503

        def _run():
            try:
                count = app.document_analyzer.check_stale_embeddings()
                logger.info(f"Manual stale embedding check complete: {count} re-analyzed")
            except Exception as e:
                logger.error(f"Manual stale embedding check failed: {e}")

        from threading import Thread
        Thread(target=_run, daemon=True).start()
        return jsonify({'success': True, 'message': 'Stale embedding check started in background — check logs for progress'})

    except Exception as e:
        logger.error(f"Failed to start stale embedding check: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/scan/process-unanalyzed', methods=['POST'])
@login_required
def api_process_unanalyzed():
    """Find every Paperless document not yet in processed_documents and analyze only those."""
    try:
        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Analyzer not running'}), 503

        # Get already-analyzed IDs from DB
        analyzed_ids = get_analyzed_doc_ids()

        # Fetch all Paperless document IDs (lightweight list, no content yet)
        all_docs = []
        page = 1
        while True:
            resp = app.paperless_client.get_documents(ordering='-modified', page_size=100, page=page)
            page_results = resp.get('results', [])
            all_docs.extend(page_results)
            if not resp.get('next'):
                break
            page += 1

        missing_docs = [d for d in all_docs if d['id'] not in analyzed_ids]

        if not missing_docs:
            return jsonify({'success': True, 'queued': 0, 'message': 'All documents already analyzed'})

        def _run(docs):
            logger.info(f"Process-unanalyzed: starting {len(docs)} documents")
            ok = 0
            for d in docs:
                try:
                    full_doc = app.paperless_client.get_document(d['id'])
                    app.document_analyzer.analyze_document(full_doc)
                    ok += 1
                except Exception as e:
                    logger.warning(f"Process-unanalyzed: failed doc {d['id']}: {e}")
            logger.info(f"Process-unanalyzed: complete — {ok}/{len(docs)} succeeded")

        from threading import Thread
        Thread(target=_run, args=(missing_docs,), daemon=True).start()

        return jsonify({
            'success': True,
            'queued': len(missing_docs),
            'message': f'Queued {len(missing_docs)} unanalyzed documents — check logs for progress',
        })

    except Exception as e:
        logger.error(f"Failed to start process-unanalyzed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/settings/poll-interval', methods=['POST'])
@login_required
def api_settings_poll_interval():
    """Update the poll interval setting."""
    try:
        data = request.json
        interval = data.get('interval')

        if not interval or not isinstance(interval, (int, float)):
            return jsonify({'error': 'Invalid interval value'}), 400

        interval = int(interval)

        # Validate range (5 seconds to 1 hour)
        if interval < 5 or interval > 3600:
            return jsonify({'error': 'Interval must be between 5 and 3600 seconds'}), 400

        # Update docker-compose.yml
        import yaml
        compose_path = '/docker-compose.yml'

        try:
            with open(compose_path, 'r') as f:
                compose_data = yaml.safe_load(f)

            # Update the environment variable
            if 'services' in compose_data and 'paperless-ai-analyzer' in compose_data['services']:
                env_vars = compose_data['services']['paperless-ai-analyzer'].get('environment', {})
                env_vars['POLL_INTERVAL_SECONDS'] = str(interval)
                compose_data['services']['paperless-ai-analyzer']['environment'] = env_vars

                # Write back
                with open(compose_path, 'w') as f:
                    yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)

                logger.info(f"Updated poll interval to {interval} seconds")

                return jsonify({
                    'success': True,
                    'message': f'Poll interval updated to {interval} seconds. Restart container to apply.',
                    'interval': interval
                })
            else:
                return jsonify({'error': 'Could not find service in docker-compose.yml'}), 500

        except Exception as e:
            logger.error(f"Failed to update docker-compose.yml: {e}")
            return jsonify({'error': f'Failed to update config: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Failed to update poll interval: {e}")
        return jsonify({'error': str(e)}), 500


# AI Configuration System
AI_CONFIG_PATH = Path('/app/data/ai_config.json')

_AI_PROVIDER_MODELS = {
    'openai':    ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
    'anthropic': ['claude-sonnet-4-6', 'claude-opus-4-6', 'claude-haiku-4-5-20251001',
                  'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'],
}

_AI_DEFAULTS = {
    'document_analysis': {
        'provider': 'openai', 'model': 'gpt-4o',
        'fallback_provider': 'anthropic', 'fallback_model': 'claude-sonnet-4-6',
    },
    'chat': {
        'provider': 'anthropic', 'model': 'claude-sonnet-4-6',
        'fallback_provider': 'openai', 'fallback_model': 'gpt-4o',
    },
    'case_intelligence': {
        'provider': 'openai', 'model': 'gpt-4o',
        'fallback_provider': 'anthropic', 'fallback_model': 'claude-sonnet-4-6',
    },
}


def get_default_ai_config():
    """Return empty v2 config (backward-compat; prefer load_ai_config())."""
    return _empty_new_ai_config()


def _empty_new_ai_config():
    return {
        'global': {
            'openai':    {'api_key': os.environ.get('OPENAI_API_KEY', ''), 'enabled': bool(os.environ.get('OPENAI_API_KEY'))},
            'anthropic': {'api_key': os.environ.get('LLM_API_KEY', ''),    'enabled': bool(os.environ.get('LLM_API_KEY'))},
        },
        'projects': {},
    }


def _migrate_old_ai_config(old_cfg: dict) -> dict:
    """Convert v1 flat format → v2 per-project format."""
    import sqlite3 as _sqlite3

    def _key(section, provider_name):
        for p in old_cfg.get(section, {}).get('providers', []):
            if p.get('name') == provider_name:
                return p.get('api_key', ''), bool(p.get('enabled', False))
        return '', False

    oai_key, oai_en = _key('document_analysis', 'openai')
    ant_key, ant_en = _key('document_analysis', 'anthropic')

    new_cfg = {
        'global': {
            'openai':    {'api_key': oai_key or os.environ.get('OPENAI_API_KEY', ''), 'enabled': oai_en or bool(os.environ.get('OPENAI_API_KEY'))},
            'anthropic': {'api_key': ant_key or os.environ.get('LLM_API_KEY', ''),    'enabled': ant_en or bool(os.environ.get('LLM_API_KEY'))},
        },
        'projects': {},
    }

    da_prim   = 'openai' if oai_key else ('anthropic' if ant_key else 'openai')
    da_fallbk = 'anthropic' if da_prim == 'openai' else 'openai'
    project_cfg = {
        'document_analysis': {
            'provider': da_prim, 'model': _AI_DEFAULTS['document_analysis']['model'],
            'fallback_provider': da_fallbk, 'fallback_model': _AI_DEFAULTS['document_analysis']['fallback_model'],
        },
        'chat': dict(_AI_DEFAULTS['chat']),
        'case_intelligence': dict(_AI_DEFAULTS['case_intelligence']),
    }

    projects_db = Path('/app/data/projects.db')
    if projects_db.exists():
        try:
            with _sqlite3.connect(str(projects_db)) as conn:
                rows = conn.execute("SELECT slug FROM projects WHERE is_archived = 0").fetchall()
                for (slug,) in rows:
                    import copy
                    new_cfg['projects'][slug] = copy.deepcopy(project_cfg)
        except Exception as _e:
            logger.warning(f"AI config migration: could not read projects.db — {_e}")

    logger.info(f"Migrated AI config v1→v2 ({len(new_cfg['projects'])} projects copied)")
    return new_cfg


def load_ai_config() -> dict:
    """Load AI config from file. Auto-migrates v1 format → v2 on first access."""
    try:
        if AI_CONFIG_PATH.exists():
            with open(AI_CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            if 'document_analysis' in cfg and 'global' not in cfg:
                cfg = _migrate_old_ai_config(cfg)
                save_ai_config(cfg)
            return cfg
    except Exception as e:
        logger.warning(f"Failed to load AI config: {e}")
    return _empty_new_ai_config()


def save_ai_config(config: dict) -> bool:
    """Persist AI config to file."""
    try:
        AI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Saved AI configuration")
        return True
    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")
        return False


def get_project_ai_config(project_slug: str, use_case: str) -> dict:
    """Return {provider, model, fallback_provider, fallback_model, api_key} for a project/use-case.
    Falls back to system defaults if the project or use_case is not configured.
    api_key resolves: project-level override → global key → env var.
    use_case: 'document_analysis' | 'chat' | 'case_intelligence'
    """
    cfg = load_ai_config()
    proj_cfg = cfg.get('projects', {}).get(project_slug, {})
    use_case_cfg = proj_cfg.get(use_case)
    if use_case_cfg and use_case_cfg.get('provider'):
        result = dict(use_case_cfg)
    else:
        result = dict(_AI_DEFAULTS.get(use_case, _AI_DEFAULTS['document_analysis']))
    # Resolve API key: project override → global
    provider = result.get('provider', 'openai')
    project_key = proj_cfg.get(f'{provider}_api_key', '').strip()
    if project_key:
        result['api_key'] = project_key
    else:
        result['api_key'] = cfg.get('global', {}).get(provider, {}).get('api_key', '')
    return result


@app.route('/api/ai-config', methods=['GET'])
@login_required
def api_ai_config_get():
    """Get current AI configuration (full v2 structure)."""
    try:
        return jsonify({'success': True, 'config': load_ai_config()})
    except Exception as e:
        logger.error(f"Failed to get AI config: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config', methods=['POST'])
@login_required
def api_ai_config_save():
    """Save AI configuration (accepts v1 or v2 format)."""
    try:
        data = request.json
        config = data.get('config')
        if not config:
            return jsonify({'error': 'Configuration is required'}), 400
        if 'global' not in config and 'document_analysis' not in config:
            return jsonify({'error': 'Invalid configuration structure'}), 400
        if save_ai_config(config):
            return jsonify({'success': True, 'message': 'AI configuration saved.'})
        return jsonify({'error': 'Failed to save configuration'}), 500
    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config/test', methods=['POST'])
@login_required
def api_ai_config_test():
    """Test an AI provider/model configuration."""
    try:
        data = request.json
        provider = data.get('provider')
        api_key = data.get('api_key', '').strip()
        model = data.get('model', '')

        if not provider or not api_key:
            return jsonify({'error': 'Provider and API key are required'}), 400

        if provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)

            # Validate by listing models — no dependency on a specific model
            models_list = client.models.list()
            model_ids = {m.id for m in models_list.data}
            preferred = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
            available = [m for m in preferred if m in model_ids]
            display = ', '.join(available[:4]) if available else (next(iter(model_ids), 'unknown'))
            return jsonify({
                'success': True,
                'message': f'✓ OpenAI key is valid. Available models include: {display}',
                'model': available[0] if available else ''
            })

        elif provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            # Try models in order
            models_to_try = [model] if model else [
                'claude-3-opus-20240229',
                'claude-3-5-sonnet-20241022',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ]

            for test_model in models_to_try:
                try:
                    response = client.messages.create(
                        model=test_model,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "Say 'test successful'"}]
                    )
                    return jsonify({
                        'success': True,
                        'message': f'✓ Anthropic API key is valid! Using model: {test_model}',
                        'model': test_model
                    })
                except Exception as e:
                    if '404' not in str(e) and 'not_found' not in str(e):
                        raise
                    continue

            return jsonify({
                'success': False,
                'error': 'No Claude models available with this API key'
            }), 400
        else:
            return jsonify({'error': f'Unknown provider: {provider}'}), 400

    except Exception as e:
        logger.error(f"AI config test failed: {e}")
        error_msg = str(e)
        if 'authentication' in error_msg.lower() or 'invalid' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': f'✗ Authentication failed: {error_msg}'
            }), 401
        return jsonify({
            'success': False,
            'error': f'✗ Error: {error_msg}'
        }), 500


# ---------------------------------------------------------------------------
# Per-project AI Config routes (v2)
# ---------------------------------------------------------------------------

def _can_access_project_config(slug: str) -> bool:
    """Non-admin users may only access their own current project's config."""
    if current_user.is_admin:
        return True
    return session.get('current_project', 'default') == slug


@app.route('/api/ai-config/global', methods=['GET'])
@admin_required
def api_ai_config_global_get():
    """Return global provider API keys (admin only)."""
    try:
        cfg = load_ai_config()
        return jsonify({'success': True, 'global': cfg.get('global', {})})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config/global', methods=['POST'])
@admin_required
def api_ai_config_global_save():
    """Update global API keys (admin only)."""
    try:
        data = request.json or {}
        new_global = data.get('global', {})
        if not isinstance(new_global, dict):
            return jsonify({'error': 'global must be a dict'}), 400
        cfg = load_ai_config()
        cfg.setdefault('global', {})
        for provider, vals in new_global.items():
            cfg['global'].setdefault(provider, {}).update(vals)
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': 'Global API keys saved.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config/projects/<slug>', methods=['GET'])
@login_required
def api_ai_config_project_get(slug):
    """Get per-project AI config (owner or admin)."""
    if not _can_access_project_config(slug):
        return jsonify({'error': 'Access denied'}), 403
    cfg = load_ai_config()
    project_cfg = cfg.get('projects', {}).get(slug, {})
    # Return config with API keys masked (never expose raw keys)
    safe_cfg = {k: v for k, v in project_cfg.items()}
    has_openai    = bool(project_cfg.get('openai_api_key', '').strip())
    has_anthropic = bool(project_cfg.get('anthropic_api_key', '').strip())
    if has_openai:    safe_cfg['openai_api_key']    = '••••••••'
    if has_anthropic: safe_cfg['anthropic_api_key'] = '••••••••'
    return jsonify({'success': True, 'slug': slug, 'config': safe_cfg,
                    'has_openai_key': has_openai, 'has_anthropic_key': has_anthropic,
                    'defaults': _AI_DEFAULTS, 'models': _AI_PROVIDER_MODELS})


@app.route('/api/ai-config/projects/<slug>', methods=['POST'])
@login_required
def api_ai_config_project_save(slug):
    """Save per-project AI config (owner or admin)."""
    if not _can_access_project_config(slug):
        return jsonify({'error': 'Access denied'}), 403
    try:
        data = request.json or {}
        new_proj_cfg = data.get('config', {})
        cfg = load_ai_config()
        existing = cfg.get('projects', {}).get(slug, {})
        # Preserve real keys when the UI submits the masked placeholder or empty string
        for key_field in ('openai_api_key', 'anthropic_api_key'):
            submitted = new_proj_cfg.get(key_field, '').strip()
            if not submitted or submitted == '••••••••':
                new_proj_cfg[key_field] = existing.get(key_field, '')
        cfg.setdefault('projects', {})[slug] = new_proj_cfg
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': f'AI config saved for project "{slug}".'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config/projects/<slug>/copy-use-case', methods=['POST'])
@login_required
def api_ai_config_copy_use_case(slug):
    """Copy one use-case config to all three use-cases within the same project."""
    if not _can_access_project_config(slug):
        return jsonify({'error': 'Access denied'}), 403
    try:
        data = request.json or {}
        source_use_case = data.get('use_case')
        if source_use_case not in ('document_analysis', 'chat', 'case_intelligence'):
            return jsonify({'error': 'Invalid use_case'}), 400
        cfg = load_ai_config()
        proj = cfg.setdefault('projects', {}).setdefault(slug, {})
        source_cfg = proj.get(source_use_case, _AI_DEFAULTS[source_use_case])
        import copy
        for uc in ('document_analysis', 'chat', 'case_intelligence'):
            proj[uc] = copy.deepcopy(source_cfg)
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': f'Copied {source_use_case} config to all use-cases in "{slug}".'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config/projects/copy', methods=['POST'])
@admin_required
def api_ai_config_copy_project():
    """Copy one project's full AI config to another project (admin only)."""
    try:
        data = request.json or {}
        src = data.get('source_slug', '').strip()
        dst = data.get('dest_slug', '').strip()
        if not src or not dst:
            return jsonify({'error': 'source_slug and dest_slug are required'}), 400
        cfg = load_ai_config()
        projects = cfg.setdefault('projects', {})
        if src not in projects:
            return jsonify({'error': f'Source project "{src}" has no AI config'}), 404
        import copy
        projects[dst] = copy.deepcopy(projects[src])
        save_ai_config(cfg)
        return jsonify({'success': True, 'message': f'Copied AI config from "{src}" to "{dst}".'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# CI Budget & Completion notification helpers
# ---------------------------------------------------------------------------

def _send_ci_budget_notification(run_id: str, pct_complete: float,
                                  cost_so_far: float, projected_total: float,
                                  budget: float, status: str):
    """Send a budget checkpoint email for a CI run."""
    try:
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return
        email = run['notification_email'] if 'notification_email' in run.keys() else ''
        if not email:
            return
        notify_on_budget = run['notify_on_budget'] if 'notify_on_budget' in run.keys() else 1
        if not notify_on_budget:
            return

        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info("SMTP not configured — skipping CI budget notification")
            return

        goal_text = run['goal_text'] if 'goal_text' in run.keys() else 'Unknown Case'
        status_label = {'under_budget': 'Under Budget', 'on_track': 'On Track',
                        'over_budget': 'OVER BUDGET'}.get(status, status)
        pct_int = int(round(pct_complete))

        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        subject = f"CI Budget Update — {goal_text[:40]} — {pct_int}% complete — {status_label}"
        body = (
            f"Case Intelligence Budget Update\n"
            f"{'=' * 50}\n\n"
            f"Case:        {goal_text}\n"
            f"Run ID:      {run_id}\n"
            f"Progress:    {pct_int}% complete\n"
            f"Status:      {status_label}\n\n"
            f"Cost So Far: ${cost_so_far:.4f}\n"
            f"Projected:   ${projected_total:.4f}\n"
            f"Budget:      ${budget:.4f}\n"
        )
        if status == 'over_budget':
            body += "\n⚠️  WARNING: Projected cost exceeds budget. Worker count has been reduced.\n"

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"CI budget notification sent to {email} ({pct_int}%, {status})")
    except Exception as e:
        logger.warning(f"Failed to send CI budget notification: {e}")


def _send_ci_complete_notification(run_id: str):
    """Send run-complete email for a CI run."""
    try:
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return
        email = run['notification_email'] if 'notification_email' in run.keys() else ''
        if not email:
            return
        notify_on_complete = run['notify_on_complete'] if 'notify_on_complete' in run.keys() else 1
        if not notify_on_complete:
            return

        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info("SMTP not configured — skipping CI complete notification")
            return

        goal_text = run['goal_text'] if 'goal_text' in run.keys() else 'Unknown Case'
        cost = run['cost_so_far_usd'] or 0
        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        body = (
            f"Case Intelligence Run Complete\n"
            f"{'=' * 50}\n\n"
            f"Case:       {goal_text}\n"
            f"Run ID:     {run_id}\n"
            f"Total Cost: ${cost:.4f}\n\n"
            f"The analysis report is ready. Log in to view or download it.\n"
        )
        msg = EmailMessage()
        msg['Subject'] = f"CI Complete — {goal_text[:50]}"
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"CI complete notification sent to {email} for run {run_id}")
    except Exception as e:
        logger.warning(f"Failed to send CI complete notification: {e}")


@app.route('/api/trigger', methods=['POST'])
@login_required
def api_trigger():
    """Manually trigger analysis of a document."""
    data = request.json
    doc_id = data.get('doc_id')

    if not doc_id:
        return jsonify({'error': 'doc_id required'}), 400

    try:
        # Verify document exists
        doc = app.paperless_client.get_document(doc_id)

        # Queue for analysis (in a real implementation, you'd queue this)
        # For now, just return success
        return jsonify({
            'success': True,
            'message': f'Document {doc_id} queued for analysis',
            'document': {
                'id': doc['id'],
                'title': doc['title']
            }
        })
    except Exception as e:
        logger.error(f"Failed to trigger analysis for doc {doc_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs')
@login_required
def api_logs():
    """Get recent log entries from the running process."""
    try:
        limit = int(request.args.get('limit', '100'))

        # Get state info for header
        state_stats = app.state_manager.get_stats()

        logs = []

        # Add header with status
        logs.append(f"=== Analyzer Status (Updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}) ===")
        logs.append(f"Last run: {state_stats.get('last_run', 'Never')}")
        logs.append(f"Documents processed: {state_stats.get('total_documents_processed', 0)}")
        logs.append(f"Active profiles: {len(app.profile_loader.profiles)}")
        logs.append(f"LLM enabled: {os.environ.get('LLM_ENABLED', 'true')}")
        logs.append("")

        # Get logs from in-memory buffer
        if log_buffer:
            logs.append("=== Recent Activity (Live Updates) ===")
            # Get last N lines from buffer
            recent_logs = list(log_buffer)[-limit:]
            logs.extend(recent_logs)
        else:
            logs.append("=== No logs available yet ===")
            logs.append("Logs will appear here once the analyzer starts processing documents")

    except Exception as e:
        logger.error(f"Failed to generate logs: {e}")
        logs = [f"Error: {str(e)}"]

    return jsonify({
        'logs': logs
    })


@app.route('/api/reprocess', methods=['POST'])
@login_required
def api_reprocess():
    """Reset state and reprocess all documents."""
    try:
        # Delete state file to force reprocessing
        state_file = Path('/app/data/state.json')
        if state_file.exists():
            state_file.unlink()
            logger.info("State file deleted - will reprocess all documents on next poll")

        # Reset in-memory state using proper method
        app.state_manager.reset()

        return jsonify({
            'success': True,
            'message': 'State reset - all documents will be reprocessed on next poll cycle'
        })
    except Exception as e:
        logger.error(f"Failed to reset state: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reprocess/<int:doc_id>', methods=['POST'])
@login_required
def api_reprocess_document(doc_id):
    """Reprocess a specific document by removing it from state."""
    try:
        # Remove document from seen_ids so it will be reprocessed
        if hasattr(app.state_manager.state, 'last_seen_ids'):
            if doc_id in app.state_manager.state.get('last_seen_ids', []):
                app.state_manager.state['last_seen_ids'].remove(doc_id)
                app.state_manager.save_state()

        return jsonify({
            'success': True,
            'message': f'Document {doc_id} will be reprocessed on next poll cycle'
        })
    except Exception as e:
        logger.error(f"Failed to reprocess document {doc_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reconcile', methods=['POST'])
@login_required
@admin_required
def api_reconcile():
    """Reconcile processed_documents and vector store against current Paperless contents.
    Removes stale records for documents deleted from Paperless. Does NOT re-analyze anything."""
    try:
        if not hasattr(app, 'paperless_client') or not app.paperless_client:
            return jsonify({'error': 'Paperless client not available'}), 503

        project_slug = session.get('current_project', 'default')

        # 1. Fetch every doc ID currently in Paperless (paginated)
        paperless_ids = set()
        page = 1
        while True:
            resp = app.paperless_client.get_documents(ordering='id', page_size=100, page=page)
            for doc in resp.get('results', []):
                paperless_ids.add(doc['id'])
            if not resp.get('next'):
                break
            page += 1

        # 2. Get doc IDs from processed_documents for this project
        from analyzer.db import get_analyzed_doc_ids
        import sqlite3 as _sqlite3
        with _sqlite3.connect('/app/data/app.db') as _conn:
            _conn.row_factory = _sqlite3.Row
            db_rows = _conn.execute(
                "SELECT doc_id FROM processed_documents WHERE project_slug = ?", (project_slug,)
            ).fetchall()
        db_ids = {r['doc_id'] for r in db_rows}

        # 3. Get doc IDs from Chroma for this project
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=project_slug)
        chroma_ids = set()
        if vs.enabled:
            raw = vs.collection.get(include=['metadatas'])
            for meta in (raw.get('metadatas') or []):
                if meta and meta.get('document_id'):
                    chroma_ids.add(int(meta['document_id']))

        # 4. Find orphans (in our indexes but deleted from Paperless)
        db_orphans    = db_ids    - paperless_ids
        chroma_orphans = chroma_ids - paperless_ids

        # 5. Remove orphans from processed_documents
        db_removed = 0
        if db_orphans:
            with _sqlite3.connect('/app/data/app.db') as _conn:
                for oid in db_orphans:
                    _conn.execute("DELETE FROM processed_documents WHERE doc_id = ?", (oid,))
                    db_removed += 1
            logger.info(f"Reconcile: removed {db_removed} stale records from processed_documents")

        # 6. Remove orphans from Chroma
        chroma_removed = 0
        if chroma_orphans and vs.enabled:
            for oid in chroma_orphans:
                vs.delete_document(oid)
                chroma_removed += 1
            logger.info(f"Reconcile: removed {chroma_removed} stale embeddings from Chroma")

        # 7. Compute gaps (in Paperless but missing from our indexes)
        not_in_db     = len(paperless_ids - db_ids)
        not_in_chroma = len(paperless_ids - chroma_ids)

        return jsonify({
            'success': True,
            'paperless_total': len(paperless_ids),
            'db_orphans_removed': db_removed,
            'chroma_orphans_removed': chroma_removed,
            'not_analyzed': not_in_db,
            'not_embedded': not_in_chroma,
            'message': (
                f"Removed {db_removed} stale DB record(s) and {chroma_removed} stale embedding(s). "
                f"{not_in_db} doc(s) not yet analyzed, {not_in_chroma} doc(s) not yet embedded."
            )
        })

    except Exception as e:
        logger.error(f"Reconcile error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/me', methods=['GET'])
@login_required
def api_me_get():
    """Return the current user's own profile data."""
    row = get_user_by_id(current_user.id)
    if not row:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'id': row['id'],
        'username': row['username'],
        'display_name': row['display_name'] or '',
        'email': row['email'] or '',
        'phone': row['phone'] or '',
        'address': row['address'] or '',
        'job_title': row['job_title'] or '',
        'role': row['role'],
    })


@app.route('/api/me', methods=['PATCH'])
@login_required
def api_me_update():
    """Update current user's own editable profile fields. Role and username are not changeable here."""
    data = request.json or {}
    allowed = {k: v for k, v in data.items()
               if k in ('display_name', 'email', 'phone', 'address', 'job_title')}
    if not allowed:
        return jsonify({'error': 'No valid fields provided'}), 400
    if 'display_name' in allowed and not str(allowed['display_name']).strip():
        return jsonify({'error': 'Display name cannot be empty'}), 400
    try:
        db_update_user(current_user.id, **allowed)
        if 'display_name' in allowed:
            current_user.display_name = allowed['display_name'].strip()
        logger.info(f"Profile updated for user {current_user.username}: {list(allowed.keys())}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/change-password', methods=['POST'])
@login_required
def api_change_password():
    """Allow the current user to change their own password."""
    from analyzer.db import get_user_by_id
    from werkzeug.security import check_password_hash, generate_password_hash
    import sqlite3
    data = request.json or {}
    current_pw = data.get('current_password', '').strip()
    new_pw = data.get('new_password', '').strip()
    if not current_pw or not new_pw:
        return jsonify({'success': False, 'error': 'Both fields are required'}), 400
    if len(new_pw) < 6:
        return jsonify({'success': False, 'error': 'New password must be at least 6 characters'}), 400
    row = get_user_by_id(current_user.id)
    if not row or not check_password_hash(row['password_hash'], current_pw):
        return jsonify({'success': False, 'error': 'Current password is incorrect'}), 403
    try:
        from analyzer.db import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.execute('UPDATE users SET password_hash=? WHERE id=?',
                     (generate_password_hash(new_pw), current_user.id))
        conn.commit()
        conn.close()
        logger.info(f"Password changed for user {current_user.username}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/llm/status')
@login_required
def api_llm_status():
    """Get LLM configuration status."""
    import os
    enabled = os.environ.get('LLM_ENABLED', 'true').lower() == 'true'
    provider = os.environ.get('LLM_PROVIDER', 'anthropic')
    has_key = bool(os.environ.get('LLM_API_KEY'))

    return jsonify({
        'enabled': enabled,
        'provider': provider,
        'has_key': has_key,
        'setup_url': 'https://console.anthropic.com/settings/keys' if provider == 'anthropic' else 'https://platform.openai.com/api-keys'
    })


@app.route('/api/llm/test', methods=['POST'])
@login_required
def api_llm_test():
    """Test an LLM API key."""
    data = request.json
    provider = data.get('provider', 'anthropic')
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({'success': False, 'error': 'API key is required'}), 400

    try:
        if provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Try models in order of preference
            models_to_try = [
                'claude-3-5-sonnet-20241022',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307'
            ]

            last_error = None
            for model in models_to_try:
                try:
                    response = client.messages.create(
                        model=model,
                        max_tokens=10,
                        messages=[{'role': 'user', 'content': 'Hi'}]
                    )
                    return jsonify({
                        'success': True,
                        'message': f'✓ Claude API key is valid! Using model: {response.model}',
                        'model': response.model
                    })
                except Exception as e:
                    last_error = e
                    if '404' not in str(e):
                        raise
                    continue

            # If we got here, all models failed
            raise last_error or Exception("No models available")
        elif provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                messages=[{'role': 'user', 'content': 'Hi'}],
                max_tokens=10
            )
            return jsonify({
                'success': True,
                'message': f'✓ OpenAI API key is valid! Model: {response.model}'
            })
        else:
            return jsonify({'success': False, 'error': 'Unknown provider'}), 400

    except Exception as e:
        error_msg = str(e)
        if '401' in error_msg or 'authentication' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': '✗ Invalid API key - authentication failed'
            }), 400
        else:
            return jsonify({
                'success': False,
                'error': f'✗ Error: {error_msg}'
            }), 500


@app.route('/api/llm/save', methods=['POST'])
@login_required
def api_llm_save():
    """Save LLM configuration and restart container."""
    data = request.json
    provider = data.get('provider', 'anthropic')
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({'success': False, 'error': 'API key is required'}), 400

    try:
        # Update docker-compose.yml
        compose_file = Path('/docker-compose.yml')

        if compose_file.exists():
            with open(compose_file, 'r') as f:
                content = f.read()

            # Replace LLM settings
            import re
            content = re.sub(
                r'LLM_ENABLED: "[^"]*"',
                'LLM_ENABLED: "true"',
                content
            )
            content = re.sub(
                r'LLM_PROVIDER: \w+',
                f'LLM_PROVIDER: {provider}',
                content
            )
            content = re.sub(
                r'LLM_API_KEY: [^\n]+',
                f'LLM_API_KEY: {api_key}',
                content
            )

            with open(compose_file, 'w') as f:
                f.write(content)

            return jsonify({
                'success': True,
                'message': 'Configuration saved! Please restart the container:\ndocker compose up -d paperless-ai-analyzer'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not find docker-compose.yml'
            }), 500

    except Exception as e:
        logger.error(f"Failed to save LLM config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search')
@login_required
def api_search():
    """Search analysis results."""
    query = request.args.get('q', '').lower()
    risk_min = request.args.get('risk_min', type=int)
    risk_max = request.args.get('risk_max', type=int)
    has_anomalies = request.args.get('has_anomalies', type=bool)

    def _chroma_meta_to_result(m):
        anomalies_str = m.get('anomalies', '')
        anomalies_list = [a.strip() for a in anomalies_str.split(',') if a.strip()] if anomalies_str else []
        return {
            'doc_id': m.get('document_id'),
            'document_id': m.get('document_id'),
            'title': m.get('title', ''),
            'document_title': m.get('title', ''),
            'brief_summary': m.get('brief_summary', ''),
            'full_summary': m.get('full_summary', ''),
            'anomalies_found': anomalies_list,
            'risk_score': m.get('risk_score', 0),
            'timestamp': m.get('timestamp', ''),
        }

    # When any search criteria is provided, query Chroma directly — it has all
    # docs with full metadata and is scoped to the current project's collection.
    if query or has_anomalies or risk_min is not None or risk_max is not None:
        try:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=session.get('current_project', 'default'))
            if vs and vs.enabled:
                results = []
                seen_ids = set()

                if query:
                    # Exact doc_id match
                    try:
                        exact = vs.collection.get(ids=[query], include=['metadatas'])
                        if exact['ids']:
                            results.append(_chroma_meta_to_result(exact['metadatas'][0]))
                            seen_ids.add(str(exact['ids'][0]))
                    except Exception:
                        pass

                    # Semantic search across all embedded docs
                    semantic = vs.search(query, n_results=50)
                    for s in semantic:
                        doc_id = str(s['document_id'])
                        if doc_id not in seen_ids:
                            # Re-fetch full metadata (vs.search strips brief/full_summary)
                            try:
                                raw = vs.collection.get(ids=[doc_id], include=['metadatas'])
                                if raw['ids']:
                                    results.append(_chroma_meta_to_result(raw['metadatas'][0]))
                                    seen_ids.add(doc_id)
                            except Exception:
                                pass

                    # Also filter all docs by title/summary/anomaly text for exact substring matches
                    all_docs = vs.collection.get(include=['metadatas'])
                    for m in all_docs['metadatas']:
                        doc_id = str(m.get('document_id', ''))
                        if doc_id in seen_ids:
                            continue
                        if (query in m.get('title', '').lower() or
                                query in m.get('brief_summary', '').lower() or
                                query in m.get('full_summary', '').lower() or
                                query in m.get('anomalies', '').lower()):
                            results.append(_chroma_meta_to_result(m))
                            seen_ids.add(doc_id)
                else:
                    # No text query — fetch all and filter by metadata
                    all_docs = vs.collection.get(include=['metadatas'])
                    for m in all_docs['metadatas']:
                        results.append(_chroma_meta_to_result(m))

                # Apply risk and anomaly filters
                if has_anomalies:
                    results = [r for r in results if r.get('anomalies_found')]
                if risk_min is not None:
                    results = [r for r in results if r.get('risk_score', 0) >= risk_min]
                if risk_max is not None:
                    results = [r for r in results if r.get('risk_score', 0) <= risk_max]

                return jsonify({'results': results, 'count': len(results)})
        except Exception as _e:
            logger.warning(f"Chroma search failed, falling back to recent_analyses: {_e}")

    # Fallback: in-memory recent_analyses (last 100 from current session)
    with ui_state['lock']:
        results = ui_state['recent_analyses']

        if query:
            results = [r for r in results if
                      query in r.get('title', '').lower() or
                      query in str(r.get('doc_id', '')).lower() or
                      query in r.get('brief_summary', '').lower() or
                      query in r.get('full_summary', '').lower() or
                      any(query in a.lower() for a in r.get('anomalies_found', []))]
        if risk_min is not None:
            results = [r for r in results if r.get('risk_score', 0) >= risk_min]
        if risk_max is not None:
            results = [r for r in results if r.get('risk_score', 0) <= risk_max]
        if has_anomalies:
            results = [r for r in results if r.get('anomalies_found')]

        return jsonify({'results': results, 'count': len(results)})


@app.route('/api/tag-evidence/<int:doc_id>')
@login_required
def api_tag_evidence(doc_id):
    """
    Get enhanced tag evidence for a specific document.
    Returns detailed information about why each tag was flagged, including specific evidence from anomaly-detector.
    """
    import sqlite3
    import json

    # Define fallback explanations for anomaly types
    ANOMALY_EXPLANATIONS = {
        'balance_mismatch': {
            'category': 'Financial Integrity',
            'description': 'The running balance does not match the calculated balance based on debits and credits.',
            'severity': 'high'
        },
        'duplicate_lines': {
            'category': 'Data Quality',
            'description': 'Duplicate transaction entries were detected in the document.',
            'severity': 'medium'
        },
        'duplicate_transaction': {
            'category': 'Data Quality',
            'description': 'The same transaction appears multiple times with identical details.',
            'severity': 'medium'
        },
        'date_ordering': {
            'category': 'Data Quality',
            'description': 'Transaction dates are not in chronological order.',
            'severity': 'low'
        },
        'missing_data': {
            'category': 'Completeness',
            'description': 'Required fields or data are missing from the document.',
            'severity': 'medium'
        },
        'forensic_risk_high': {
            'category': 'Document Forensics',
            'description': 'High risk of document tampering detected through image analysis (risk score > 60%).',
            'severity': 'critical'
        },
        'forensic_risk_medium': {
            'category': 'Document Forensics',
            'description': 'Medium risk of document tampering detected through image analysis (risk score 30-60%).',
            'severity': 'medium'
        },
        'forensic_risk_low': {
            'category': 'Document Forensics',
            'description': 'Low risk indicators detected through image analysis (risk score < 30%).',
            'severity': 'low'
        }
    }

    # Try to find analysis in recent_analyses first for enhanced tags
    analysis = None
    with ui_state['lock']:
        for result in ui_state['recent_analyses']:
            if result.get('doc_id') == doc_id or result.get('document_id') == doc_id:
                analysis = result
                break

    # If not in recent_analyses, fetch from Paperless API
    if not analysis:
        try:
            doc = app.paperless_client.get_document(doc_id)

            # Get tags from Paperless
            tags = []
            for tag_id in doc.get('tags', []):
                try:
                    tag_response = app.paperless_client.session.get(
                        f'{app.paperless_client.base_url}/api/tags/{tag_id}/'
                    )
                    if tag_response.ok:
                        tags.append(tag_response.json().get('name', ''))
                except:
                    pass

            # Extract anomaly tags
            anomalies = [t.replace('anomaly:', '') for t in tags if t.startswith('anomaly:')]

            analysis = {
                'document_id': doc_id,
                'document_title': doc.get('title', 'Unknown'),
                'anomalies_found': anomalies,
                'enhanced_tags': []
            }
        except Exception as e:
            logger.error(f"Failed to fetch document {doc_id}: {e}")
            return jsonify({'error': 'Document not found'}), 404

    # Collect all tags with evidence
    all_tags = []

    # Add enhanced tags with detailed evidence (issue: type tags from LLM analysis)
    enhanced_tags = analysis.get('enhanced_tags', [])
    all_tags.extend(enhanced_tags)

    # Query anomaly-detector database for detailed evidence
    anomaly_detector_evidence = {}
    try:
        db_path = '/app/anomaly_data/anomaly_detector.db'
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query the processed_documents table
        cursor.execute("""
            SELECT
                has_anomalies,
                anomaly_types,
                balance_check_status,
                balance_diff_amount,
                beginning_balance,
                ending_balance,
                calculated_balance,
                pattern_flags,
                document_type
            FROM processed_documents
            WHERE paperless_doc_id = ?
        """, (doc_id,))

        row = cursor.fetchone()
        if row:
            # Parse JSON fields
            anomaly_types = json.loads(row['anomaly_types']) if row['anomaly_types'] else []
            pattern_flags = json.loads(row['pattern_flags']) if row['pattern_flags'] else []
            db_balance_status = row['balance_check_status']

            # Index pattern_flags by type for easy lookup (keeps all flags per type)
            flags_by_type = {}
            for flag in pattern_flags:
                t = flag.get('type', '')
                flags_by_type.setdefault(t, []).append(flag)

            # Build evidence for each confirmed anomaly type
            for anomaly_type in anomaly_types:
                matching_flags = flags_by_type.get(anomaly_type, [])
                evidence = {'db_confirmed': True, 'severity': 'medium'}

                if anomaly_type == 'balance_mismatch':
                    evidence['status'] = db_balance_status
                    if row['balance_diff_amount'] is not None:
                        evidence['difference'] = f"${abs(row['balance_diff_amount']):.2f}"
                    if row['beginning_balance'] is not None:
                        evidence['beginning_balance'] = f"${row['beginning_balance']:.2f}"
                    if row['ending_balance'] is not None:
                        evidence['ending_balance'] = f"${row['ending_balance']:.2f}"
                    if row['calculated_balance'] is not None:
                        evidence['calculated_balance'] = f"${row['calculated_balance']:.2f}"
                    # Include descriptions + details from each flag
                    issue_lines = []
                    for flag in matching_flags:
                        issue_lines.append(flag.get('description', ''))
                        for detail in flag.get('details', []):
                            issue_lines.append(f"  – {detail}")
                    if issue_lines:
                        evidence['issues'] = issue_lines

                elif anomaly_type == 'duplicate_lines':
                    duplicate_texts = []
                    for flag in matching_flags:
                        duplicate_texts.extend(flag.get('details', []))
                    evidence['duplicate_texts'] = duplicate_texts
                    evidence['count'] = len(duplicate_texts)
                    if matching_flags:
                        evidence['severity'] = matching_flags[0].get('severity', 'medium')

                elif anomaly_type == 'page_discontinuity':
                    for flag in matching_flags:
                        evidence['details'] = flag.get('details', [])
                        evidence['found_pages'] = flag.get('found_pages', [])
                        evidence['declared_max'] = flag.get('declared_max')
                        evidence['actual_count'] = flag.get('actual_count')
                        evidence['severity'] = flag.get('severity', 'medium')
                        break

                else:
                    # Generic: collect description + details from flags
                    issue_lines = []
                    for flag in matching_flags:
                        issue_lines.append(flag.get('description', ''))
                        for detail in flag.get('details', []):
                            issue_lines.append(f"  – {detail}")
                    if issue_lines:
                        evidence['issues'] = issue_lines
                    if matching_flags:
                        evidence['severity'] = matching_flags[0].get('severity', 'medium')

                anomaly_detector_evidence[anomaly_type] = evidence

            # Also store balance check result so we can detect false-positive tags
            anomaly_detector_evidence['_balance_status'] = db_balance_status

        conn.close()
    except Exception as e:
        logger.error(f"Failed to query anomaly-detector database: {e}")
        # Continue with fallback explanations

    def _build_description(anomaly, evidence):
        """Build a human-readable description from structured evidence."""
        if anomaly == 'balance_mismatch':
            # If DB says PASS, this is a false-positive tag
            if evidence.get('status') == 'PASS':
                return "Balance check passed — no arithmetic mismatch found in this document."
            lines = []
            if evidence.get('beginning_balance'):
                lines.append(f"Beginning Balance: {evidence['beginning_balance']}")
            if evidence.get('ending_balance'):
                lines.append(f"Ending Balance:    {evidence['ending_balance']}")
            if evidence.get('calculated_balance'):
                lines.append(f"Calculated Total:  {evidence['calculated_balance']}")
            if evidence.get('difference'):
                lines.append(f"Discrepancy:       {evidence['difference']}")
            header = "\n".join(f"• {l}" for l in lines) if lines else ""
            specific = "\n".join(f"• {i}" for i in evidence.get('issues', []))
            return (header + "\n\n" + specific).strip() if specific else (header or "Balance mismatch detected — specific amounts unavailable.")

        elif anomaly == 'duplicate_lines':
            texts = evidence.get('duplicate_texts', [])
            count = evidence.get('count', len(texts))
            if texts:
                quoted = "\n".join(f'  "{t}"' for t in texts[:10])
                return f"Found {count} duplicate line(s):\n{quoted}"
            return f"Found {count} duplicate transaction line(s)."

        elif anomaly == 'page_discontinuity':
            details = evidence.get('details', [])
            found = evidence.get('found_pages', [])
            declared = evidence.get('declared_max')
            actual = evidence.get('actual_count')
            lines = list(details)
            if found and declared:
                expected = set(range(1, declared + 1))
                missing = sorted(expected - set(found))
                if missing:
                    lines.append(f"Pages present in headers: {found}")
                    lines.append(f"Expected (1–{declared}): {missing} are missing")
            if actual and declared and actual != declared:
                lines.append(f"PDF has {actual} physical pages, headers say 1 of {declared}")
            return "\n".join(f"• {l}" for l in lines) if lines else "Page numbering inconsistencies detected."

        else:
            issues = evidence.get('issues', [])
            return "\n".join(f"• {i}" for i in issues) if issues else f"Flagged with: {anomaly.replace('_', ' ').title()}"

    # Add standard anomaly tags with real evidence from anomaly-detector
    anomalies = analysis.get('anomalies_found', [])
    db_balance_status = anomaly_detector_evidence.get('_balance_status')

    for anomaly in anomalies:
        # Skip if already covered by enhanced_tags (LLM findings)
        if any(tag.get('tag', '').endswith(anomaly) for tag in enhanced_tags):
            continue

        if anomaly in anomaly_detector_evidence:
            real_evidence = anomaly_detector_evidence[anomaly]
            description = _build_description(anomaly, real_evidence)
            explanation = ANOMALY_EXPLANATIONS.get(anomaly, {
                'category': 'Anomaly Detection',
                'severity': real_evidence.get('severity', 'medium')
            })
            # Downgrade severity for confirmed false-positives
            severity = explanation.get('severity', 'medium')
            if anomaly == 'balance_mismatch' and real_evidence.get('status') == 'PASS':
                severity = 'info'

            all_tags.append({
                'tag': f'anomaly:{anomaly}',
                'category': explanation.get('category', 'Anomaly Detection'),
                'description': description,
                'severity': severity,
                'evidence': real_evidence
            })

        elif anomaly == 'balance_mismatch' and db_balance_status == 'PASS':
            # Tag exists but DB confirms balance passed
            all_tags.append({
                'tag': f'anomaly:{anomaly}',
                'category': 'Financial Integrity',
                'description': "Balance check passed — no arithmetic mismatch found in this document.",
                'severity': 'info',
                'evidence': {'status': 'PASS'}
            })

        else:
            # Fallback: no DB record for this anomaly
            explanation = ANOMALY_EXPLANATIONS.get(anomaly, {
                'category': 'Anomaly Detection',
                'description': f'This document was flagged with: {anomaly.replace("_", " ").title()}',
                'severity': 'medium'
            })
            all_tags.append({
                'tag': f'anomaly:{anomaly}',
                'category': explanation['category'],
                'description': explanation['description'],
                'severity': explanation['severity'],
                'evidence': {}
            })

    return jsonify({
        'document_id': doc_id,
        'document_title': analysis.get('document_title') or analysis.get('title', 'Unknown'),
        'tags': all_tags,
        'integrity_summary': analysis.get('integrity_summary', ''),
        'issue_count': analysis.get('issue_count', 0) + len(anomalies),
        'critical_count': analysis.get('critical_count', 0)
    })


# ==================== Project Management API (v1.5.0) ====================

@app.route('/api/projects', methods=['GET'])
@login_required
def api_list_projects():
    """List all projects."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        projects = app.project_manager.list_projects(include_archived=True)

        # Add statistics and per-project Paperless config to each project
        from analyzer.court_db import get_court_doc_count
        for project in projects:
            try:
                stats = app.project_manager.get_statistics(project['slug'])
                project.update(stats)
            except Exception:
                pass  # Skip stats if unavailable
            try:
                project['court_doc_count'] = get_court_doc_count(project['slug'])
            except Exception:
                project['court_doc_count'] = 0
            try:
                cfg = app.project_manager.get_paperless_config(project['slug'])
                project['paperless_doc_base_url'] = cfg.get('doc_base_url') or ''
                project['paperless_url'] = cfg.get('url') or ''
                project['paperless_configured'] = bool(cfg.get('url') and cfg.get('token'))
            except Exception:
                project['paperless_doc_base_url'] = ''
                project['paperless_url'] = ''
                project['paperless_configured'] = False

        return jsonify({
            'projects': projects,
            'count': len(projects),
            'global_paperless_base_url': os.environ.get('PAPERLESS_PUBLIC_BASE_URL', ''),
        })

    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects', methods=['POST'])
@login_required
def api_create_project():
    """Create new project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        data = request.json

        if not data.get('name'):
            return jsonify({'error': 'Project name is required'}), 400

        # Generate slug if not provided
        slug = data.get('slug')
        if not slug:
            slug = app.project_manager.suggest_slug(data['name'])

        # Create project
        project = app.project_manager.create_project(
            slug=slug,
            name=data['name'],
            description=data.get('description', ''),
            color=data.get('color'),
            metadata=data.get('metadata')
        )

        # Create Paperless tag
        if app.paperless_client:
            app.paperless_client.get_or_create_tag(
                f"project:{slug}",
                color=project.get('color', '#3498db')
            )

        # Auto-provision a dedicated Paperless instance in the background
        import threading
        _provision_status[slug] = {'status': 'queued', 'phase': 'Queued', 'error': None}
        threading.Thread(
            target=_provision_project_paperless, args=(slug,), daemon=True
        ).start()

        logger.info(f"Created project: {slug}")
        return jsonify(project), 201

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>', methods=['GET'])
@login_required
def api_get_project(slug):
    """Get project details."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Add statistics
        stats = app.project_manager.get_statistics(slug)
        project.update(stats)
        from analyzer.court_db import get_court_doc_count
        try:
            project['court_doc_count'] = get_court_doc_count(slug)
        except Exception:
            project['court_doc_count'] = 0

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to get project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>', methods=['PUT'])
@login_required
def api_update_project(slug):
    """Update project metadata."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        data = request.json
        project = app.project_manager.update_project(slug, **data)

        logger.info(f"Updated project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to update project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>', methods=['DELETE'])
@login_required
def api_delete_project(slug):
    """Delete project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        delete_data = request.args.get('delete_data', type=bool, default=True)

        # Delete vector store collection if requested
        if delete_data:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=slug)
            if vs.enabled:
                vs.delete_collection()

        # Delete project
        success = app.project_manager.delete_project(slug, delete_data=delete_data)

        if success:
            logger.info(f"Deleted project: {slug}")
            return jsonify({'success': True, 'message': f'Project {slug} deleted'})
        else:
            return jsonify({'error': 'Failed to delete project'}), 500

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/archive', methods=['POST'])
@login_required
def api_archive_project(slug):
    """Archive project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = app.project_manager.archive_project(slug)
        logger.info(f"Archived project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to archive project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/unarchive', methods=['POST'])
@login_required
def api_unarchive_project(slug):
    """Unarchive project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = app.project_manager.unarchive_project(slug)
        logger.info(f"Unarchived project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to unarchive project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/paperless-config', methods=['POST'])
@login_required
def api_set_project_paperless_config(slug):
    """Save per-project Paperless-ngx connection config (URL, token, public base URL)."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        data = request.json or {}
        updates = {}
        if 'url' in data:
            updates['paperless_url'] = data['url'] or None
        if 'token' in data:
            updates['paperless_token'] = data['token'] or None
        if 'doc_base_url' in data:
            updates['paperless_doc_base_url'] = data['doc_base_url'] or None

        app.project_manager.update_project(slug, **updates)

        # Invalidate client cache so next request picks up new credentials
        _project_client_cache.pop(slug, None)

        logger.info(f"Updated Paperless config for project '{slug}'")
        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Failed to set Paperless config for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/paperless-config', methods=['GET'])
@login_required
def api_get_project_paperless_config(slug):
    """Get per-project Paperless-ngx connection config (token is masked)."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        cfg = app.project_manager.get_paperless_config(slug)
        return jsonify({
            'url': cfg.get('url') or '',
            'token_set': bool(cfg.get('token')),
            'doc_base_url': cfg.get('doc_base_url') or '',
        })
    except Exception as e:
        logger.error(f"Failed to get Paperless config for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/provision-snippets', methods=['GET'])
@login_required
def api_provision_snippets(slug):
    """Return ready-to-paste infrastructure snippets for a per-project Paperless instance."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Auto-assign next free Redis DB index (0 = shared default, 1+ = per-project)
        try:
            all_projects = app.project_manager.list_projects(include_archived=True)
            projects_with_instance = [
                p['slug'] for p in all_projects
                if app.project_manager.get_paperless_config(p['slug']).get('url')
            ]
            # Assign index based on position in the list (or next available)
            if slug in projects_with_instance:
                redis_db = projects_with_instance.index(slug) + 1
            else:
                redis_db = len(projects_with_instance) + 1
        except Exception:
            redis_db = 1

        db_name = f"paperless_{slug.replace('-', '_')}"
        db_user = f"paperless_{slug.replace('-', '_')}"
        web_svc = f"paperless-web-{slug}"
        consumer_svc = f"paperless-consumer-{slug}"

        compose_snippet = f"""  {web_svc}:
    image: ghcr.io/paperless-ngx/paperless-ngx:latest
    restart: unless-stopped
    depends_on:
      - paperless-postgres
      - paperless-redis
    environment:
      PAPERLESS_REDIS: redis://paperless-redis:6379/{redis_db}
      PAPERLESS_DBHOST: paperless-postgres
      PAPERLESS_DBNAME: {db_name}
      PAPERLESS_DBUSER: paperless
      PAPERLESS_DBPASS: ${{PAPERLESS_DBPASS}}
      PAPERLESS_SECRET_KEY: ${{PAPERLESS_{slug.upper().replace('-','_')}_SECRET_KEY}}
      PAPERLESS_URL: https://${{DOMAIN}}/paperless-{slug}
      PAPERLESS_FORCE_SCRIPT_NAME: /paperless-{slug}
      PAPERLESS_STATIC_URL: /paperless-{slug}/static/
      PAPERLESS_CONSUMPTION_DIR: /usr/src/paperless/consume
      PAPERLESS_DATA_DIR: /usr/src/paperless/data
      PAPERLESS_MEDIA_ROOT: /usr/src/paperless/media
      PAPERLESS_OCR_LANGUAGE: eng
    volumes:
      - /mnt/s/documents/paperless-{slug}/data:/usr/src/paperless/data
      - /mnt/s/documents/paperless-{slug}/media:/usr/src/paperless/media
      - /mnt/s/documents/paperless-{slug}/consume:/usr/src/paperless/consume
      - /mnt/s/documents/paperless-{slug}/export:/usr/src/paperless/export
    networks:
      - default

  {consumer_svc}:
    image: ghcr.io/paperless-ngx/paperless-ngx:latest
    restart: unless-stopped
    depends_on:
      - {web_svc}
    environment:
      PAPERLESS_REDIS: redis://paperless-redis:6379/{redis_db}
      PAPERLESS_DBHOST: paperless-postgres
      PAPERLESS_DBNAME: {db_name}
      PAPERLESS_DBUSER: paperless
      PAPERLESS_DBPASS: ${{PAPERLESS_DBPASS}}
      PAPERLESS_SECRET_KEY: ${{PAPERLESS_{slug.upper().replace('-','_')}_SECRET_KEY}}
      PAPERLESS_CONSUMPTION_DIR: /usr/src/paperless/consume
      PAPERLESS_DATA_DIR: /usr/src/paperless/data
      PAPERLESS_MEDIA_ROOT: /usr/src/paperless/media
    volumes:
      - /mnt/s/documents/paperless-{slug}/data:/usr/src/paperless/data
      - /mnt/s/documents/paperless-{slug}/media:/usr/src/paperless/media
      - /mnt/s/documents/paperless-{slug}/consume:/usr/src/paperless/consume
      - /mnt/s/documents/paperless-{slug}/export:/usr/src/paperless/export
    networks:
      - default
    command: document_consumer"""

        nginx_snippet = f"""location /paperless-{slug}/ {{
    proxy_pass http://paperless-web-{slug}:8000/paperless-{slug}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
    proxy_set_header X-Forwarded-Port $server_port;
    proxy_set_header X-Script-Name /paperless-{slug};
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    client_max_body_size 100M;
}}
location = /paperless-{slug} {{
    return 301 /paperless-{slug}/;
}}"""

        postgres_sql = f"""-- Run as the 'paperless' superuser (shared DB user):
-- CREATE DATABASE {db_name};  ← Auto-provisioning does this automatically
-- The shared 'paperless' DB user owns all project DBs."""

        return jsonify({
            'slug': slug,
            'web_service': web_svc,
            'consumer_service': consumer_svc,
            'redis_db_index': redis_db,
            'db_name': db_name,
            'db_user': db_user,
            'compose': compose_snippet,
            'nginx': nginx_snippet,
            'postgres_sql': postgres_sql,
        })

    except Exception as e:
        logger.error(f"Failed to generate provision snippets for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/migrate-to-own-paperless', methods=['POST'])
@login_required
def api_migrate_to_own_paperless(slug):
    """Start a background migration of all project docs from shared to per-project Paperless."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        existing = _migration_status.get(slug, {})
        if existing.get('status') == 'running':
            return jsonify({'error': 'Migration already in progress'}), 409

        _migration_status[slug] = {
            'status': 'running', 'total': 0, 'migrated': 0,
            'failed': 0, 'error': None, 'phase': 'starting'
        }
        from threading import Thread
        Thread(target=_migrate_project_to_own_paperless, args=(slug,), daemon=True).start()
        return jsonify({'success': True, 'message': f'Migration started for project {slug}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/migration-status', methods=['GET'])
@login_required
def api_migration_status(slug):
    """Return current migration status for a project."""
    status = _migration_status.get(slug, {'status': 'idle'})
    return jsonify(status)


@app.route('/api/projects/<slug>/provision-status', methods=['GET'])
@login_required
def api_provision_status(slug):
    """Return current auto-provisioning status for a project."""
    status = _provision_status.get(slug, {'status': 'idle'})
    return jsonify(status)


@app.route('/api/projects/<slug>/reprovision', methods=['POST'])
@login_required
def api_reprovision(slug):
    """Trigger automated Paperless provisioning for an existing project."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        existing = _provision_status.get(slug, {})
        if existing.get('status') == 'running':
            return jsonify({'error': 'Provisioning already in progress'}), 409

        _provision_status[slug] = {'status': 'queued', 'phase': 'Queued', 'error': None}
        from threading import Thread
        Thread(target=_provision_project_paperless, args=(slug,), daemon=True).start()
        return jsonify({'success': True, 'message': f'Provisioning started for project {slug}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# In-memory migration state: {slug: {status, total, migrated, failed, error, phase}}
_migration_status: dict = {}

# In-memory provisioning state: {slug: {status, phase, error}}
_provision_status: dict = {}


def _provision_log(slug: str, phase: str, status: str = None):
    s = _provision_status.setdefault(slug, {})
    s['phase'] = phase
    if status:
        s['status'] = status
    logger.info(f"[Provision:{slug}] {phase}")


def _provision_project_paperless(slug: str) -> None:
    """
    Daemon thread: spin up a dedicated Paperless-ngx instance for the given project.
    Creates Postgres DB, starts web+consumer containers, writes nginx conf, and wires
    the project to its new instance in the DB.
    """
    import secrets as _secrets
    import time as _time
    import os as _os

    _provision_status[slug] = {'status': 'running', 'phase': 'Starting', 'error': None}

    try:
        dc = _get_docker_client()
        if dc is None:
            raise RuntimeError("Docker SDK not available — cannot auto-provision")

        # ── 1. Read shared config from existing paperless-web container ─────────────
        _provision_log(slug, 'Reading shared Paperless config')
        try:
            pw_container = dc.containers.get('paperless-web')
            pw_env = {
                e.split('=', 1)[0]: e.split('=', 1)[1]
                for e in pw_container.attrs['Config']['Env']
                if '=' in e
            }
        except Exception as e:
            raise RuntimeError(f"Cannot reach paperless-web container: {e}")

        db_pass = pw_env.get('PAPERLESS_DBPASS', '')
        if not db_pass:
            raise RuntimeError("PAPERLESS_DBPASS not found in paperless-web env")
        time_zone = pw_env.get('PAPERLESS_TIME_ZONE', 'America/Toronto')

        # ── 2. Pick next free Redis DB index ────────────────────────────────────────
        _provision_log(slug, 'Assigning Redis DB index')
        used_redis_dbs = set()
        try:
            for c in dc.containers.list():
                if 'paperless-web-' in c.name:
                    c_env = {
                        e.split('=', 1)[0]: e.split('=', 1)[1]
                        for e in c.attrs['Config']['Env']
                        if '=' in e
                    }
                    redis_val = c_env.get('PAPERLESS_REDIS', '')
                    # format: redis://paperless-redis:6379/N
                    if '/' in redis_val:
                        try:
                            used_redis_dbs.add(int(redis_val.rsplit('/', 1)[1]))
                        except ValueError:
                            pass
        except Exception:
            pass
        redis_db = 1
        while redis_db in used_redis_dbs:
            redis_db += 1

        # ── 3. Generate credentials ──────────────────────────────────────────────────
        _provision_log(slug, 'Generating credentials')
        secret_key = _secrets.token_hex(32)
        admin_password = _secrets.token_urlsafe(16)

        # ── 4. Create Postgres database (idempotent) ─────────────────────────────────
        _provision_log(slug, 'Creating Postgres database')
        db_name = f"paperless_{slug.replace('-', '_')}"
        try:
            pg = dc.containers.get('paperless-postgres')
            # Check if DB exists
            check_result = pg.exec_run(
                ['psql', '-U', 'paperless', '-tAc',
                 f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"]
            )
            if check_result.output.decode().strip() != '1':
                create_result = pg.exec_run(
                    ['psql', '-U', 'paperless', '-c', f'CREATE DATABASE {db_name}']
                )
                if create_result.exit_code != 0:
                    raise RuntimeError(
                        f"Failed to create DB {db_name}: "
                        f"{create_result.output.decode()[:200]}"
                    )
        except Exception as e:
            if 'No such container' in str(e):
                raise RuntimeError("paperless-postgres container not found") from e
            raise

        # ── 5. Create host directories ───────────────────────────────────────────────
        _provision_log(slug, 'Creating host directories')
        base_dir = f"/mnt/s/documents/paperless-{slug}"
        for subdir in ['data', 'media', 'consume', 'export', 'tmp']:
            _os.makedirs(f"{base_dir}/{subdir}", exist_ok=True)

        # ── 6. Start web container ────────────────────────────────────────────────────
        web_name = f"paperless-web-{slug}"
        consumer_name = f"paperless-consumer-{slug}"
        image = 'ghcr.io/paperless-ngx/paperless-ngx:latest'

        _provision_log(slug, f'Starting {web_name}')
        # Remove existing containers if present (idempotent)
        for cname in [web_name, consumer_name]:
            try:
                existing = dc.containers.get(cname)
                existing.stop(timeout=10)
                existing.remove()
            except Exception:
                pass

        # PAPERLESS_URL must be origin-only (no path) — Paperless passes it to CORS_ALLOWED_ORIGINS
        # and Django CORS will reject it if it contains a path.  Copy directly from existing container.
        existing_url = pw_env.get('PAPERLESS_URL', 'https://voipguru.org').rstrip('/')
        existing_csrf = pw_env.get('PAPERLESS_CSRF_TRUSTED_ORIGINS', existing_url)
        existing_allowed = pw_env.get('PAPERLESS_ALLOWED_HOSTS', existing_url.split('/')[2])
        domain = existing_url.split('/')[2]  # e.g. "www.voipguru.org"
        web_env = {
            'PAPERLESS_REDIS': f'redis://paperless-redis:6379/{redis_db}',
            'PAPERLESS_DBHOST': 'paperless-postgres',
            'PAPERLESS_DBNAME': db_name,
            'PAPERLESS_DBUSER': 'paperless',
            'PAPERLESS_DBPASS': db_pass,
            'PAPERLESS_SECRET_KEY': secret_key,
            'PAPERLESS_URL': existing_url,                           # origin only, no sub-path
            'PAPERLESS_FORCE_SCRIPT_NAME': f'/paperless-{slug}',    # sub-path goes here
            'PAPERLESS_STATIC_URL': f'/paperless-{slug}/static/',
            'PAPERLESS_CSRF_TRUSTED_ORIGINS': existing_csrf,
            'PAPERLESS_ALLOWED_HOSTS': existing_allowed + f',paperless-web-{slug}',
            'PAPERLESS_TIME_ZONE': time_zone,
            'PAPERLESS_CONSUMPTION_DIR': '/usr/src/paperless/consume',
            'PAPERLESS_DATA_DIR': '/usr/src/paperless/data',
            'PAPERLESS_MEDIA_ROOT': '/usr/src/paperless/media',
            'PAPERLESS_OCR_LANGUAGE': 'eng',
            'PAPERLESS_ADMIN_USER': 'admin',
            'PAPERLESS_ADMIN_PASSWORD': admin_password,
            'PAPERLESS_ADMIN_MAIL': 'admin@localhost',
        }
        web_vols = {
            f'{base_dir}/data': {'bind': '/usr/src/paperless/data', 'mode': 'rw'},
            f'{base_dir}/media': {'bind': '/usr/src/paperless/media', 'mode': 'rw'},
            f'{base_dir}/consume': {'bind': '/usr/src/paperless/consume', 'mode': 'rw'},
            f'{base_dir}/export': {'bind': '/usr/src/paperless/export', 'mode': 'rw'},
            f'{base_dir}/tmp': {'bind': '/tmp/paperless', 'mode': 'rw'},
        }

        dc.containers.run(
            image,
            name=web_name,
            detach=True,
            environment=web_env,
            volumes=web_vols,
            network='docker_default',
            restart_policy={'Name': 'unless-stopped'},
            labels={'managed-by': 'paperless-ai-analyzer', 'project': slug},
        )

        # ── 7. Start consumer container ───────────────────────────────────────────────
        _provision_log(slug, f'Starting {consumer_name}')
        consumer_env = {k: v for k, v in web_env.items()}
        dc.containers.run(
            image,
            name=consumer_name,
            command='document_consumer',
            detach=True,
            environment=consumer_env,
            volumes=web_vols,
            network='docker_default',
            restart_policy={'Name': 'unless-stopped'},
            labels={'managed-by': 'paperless-ai-analyzer', 'project': slug},
        )

        # ── 8. Wait for web container to be ready ─────────────────────────────────────
        # First run: Django runs migrations + collects static files — allow up to 8 minutes.
        # urllib raises HTTPError for 4xx/5xx — those still mean Django is responding.
        # Only a connection error (refused/timeout) means the container isn't ready yet.
        import urllib.request as _urlreq
        import urllib.error as _urlerr
        deadline = _time.monotonic() + 480
        ready = False
        while _time.monotonic() < deadline:
            elapsed = int(_time.monotonic() - (deadline - 480))
            _provision_log(slug, f'Waiting for Paperless web to start ({elapsed}s elapsed, up to 8 min)…')
            try:
                _urlreq.urlopen(f'http://{web_name}:8000/api/', timeout=5)
                ready = True
                break
            except _urlerr.HTTPError:
                # Any HTTP response — Django is up and serving
                ready = True
                break
            except Exception:
                # Connection refused / timeout — not up yet
                pass
            _time.sleep(5)

        if not ready:
            raise RuntimeError(
                f"{web_name} did not become ready within 8 minutes. "
                f"Check logs: docker logs {web_name}"
            )

        # ── 9. Force-set admin password via docker exec ───────────────────────────────
        # Paperless skips PAPERLESS_ADMIN_PASSWORD if the admin user already exists
        # (e.g. on reprovision). We exec a Django shell command to always set it.
        _provision_log(slug, 'Setting admin password')
        py_cmd = (
            "from django.contrib.auth import get_user_model; "
            "U=get_user_model(); "
            "u,_=U.objects.get_or_create(username='admin'); "
            f"u.set_password('{admin_password}'); "
            "u.is_superuser=True; u.is_staff=True; u.save()"
        )
        web_c = dc.containers.get(web_name)
        reset = web_c.exec_run(
            ['python', '/usr/src/paperless/src/manage.py', 'shell', '-c', py_cmd],
        )
        if reset.exit_code != 0:
            raise RuntimeError(
                f"Password reset exec failed (exit {reset.exit_code}): "
                f"{reset.output.decode()[:400]}"
            )

        # ── 10. Obtain API token ────────────────────────────────────────────────────────
        _provision_log(slug, 'Obtaining Paperless API token')
        import json as _json
        import urllib.parse as _urlparse
        # Retry a few times in case the password reset needs a moment to commit
        api_token = None
        for _attempt in range(6):
            token_data = _urlparse.urlencode({
                'username': 'admin',
                'password': admin_password,
            }).encode()
            token_req = _urlreq.Request(
                f'http://{web_name}:8000/api/token/',
                data=token_data,
                method='POST',
            )
            try:
                with _urlreq.urlopen(token_req, timeout=10) as resp:
                    token_body = _json.loads(resp.read())
                    api_token = token_body['token']
                    break
            except _urlerr.HTTPError as he:
                logger.warning(f"[Provision:{slug}] Token attempt {_attempt+1}: HTTP {he.code}")
                _time.sleep(3)
            except Exception as e:
                logger.warning(f"[Provision:{slug}] Token attempt {_attempt+1}: {e}")
                _time.sleep(3)
        if not api_token:
            raise RuntimeError(f"Failed to get API token from {web_name} after 6 attempts")

        # ── 10. Write nginx location conf ─────────────────────────────────────────────
        _provision_log(slug, 'Writing nginx location conf')
        nginx_conf = f"""# Auto-generated for project: {slug}
location /paperless-{slug}/ {{
    proxy_pass http://paperless-web-{slug}:8000/paperless-{slug}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
    proxy_set_header X-Forwarded-Port $server_port;
    proxy_set_header X-Script-Name /paperless-{slug};
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    client_max_body_size 100M;
}}
location = /paperless-{slug} {{
    return 301 /paperless-{slug}/;
}}
"""
        nginx_conf_path = f'/app/nginx-projects-locations.d/paperless-{slug}.conf'
        with open(nginx_conf_path, 'w') as f:
            f.write(nginx_conf)

        # ── 11. Reload nginx ──────────────────────────────────────────────────────────
        _provision_log(slug, 'Reloading nginx')
        try:
            nginx_c = dc.containers.get('nginx')
            nginx_c.exec_run(['nginx', '-s', 'reload'])
        except Exception as e:
            logger.warning(f"[Provision:{slug}] nginx reload failed (non-fatal): {e}")

        # ── 12. Save config to DB ─────────────────────────────────────────────────────
        _provision_log(slug, 'Saving project config to database')
        paperless_url = f'http://{web_name}:8000'
        doc_base_url = f'https://{domain}/paperless-{slug}'
        app.project_manager.update_project(slug,
            paperless_url=paperless_url,
            paperless_token=api_token,
            paperless_doc_base_url=doc_base_url,
            paperless_secret_key=secret_key,
            paperless_admin_password=admin_password,
        )
        _project_client_cache.pop(slug, None)

        _provision_status[slug] = {
            'status': 'complete',
            'phase': 'Provisioning complete',
            'error': None,
            'paperless_url': paperless_url,
            'doc_base_url': doc_base_url,
        }
        logger.info(f"[Provision:{slug}] Complete — accessible at {doc_base_url}")

    except Exception as exc:
        logger.error(f"[Provision:{slug}] Failed: {exc}", exc_info=True)
        _provision_status[slug] = {
            'status': 'error',
            'phase': 'Failed',
            'error': str(exc),
        }


def _migration_log(slug: str, phase: str, migrated: int = None, total: int = None,
                   failed: int = None):
    s = _migration_status.setdefault(slug, {})
    s['phase'] = phase
    if migrated is not None:
        s['migrated'] = migrated
    if total is not None:
        s['total'] = total
    if failed is not None:
        s['failed'] = failed
    logger.info(f"[Migration:{slug}] {phase} — {s.get('migrated',0)}/{s.get('total',0)}")


def _migrate_project_to_own_paperless(slug: str) -> None:
    """
    Daemon thread: migrate all project documents from the shared Paperless instance
    to the project's dedicated Paperless instance.
    """
    import sqlite3 as _sqlite3
    import re as _re
    status = _migration_status.setdefault(slug, {})
    status['status'] = 'running'
    status['error'] = None

    try:
        # 1. Preflight
        _migration_log(slug, 'preflight')
        new_client = _get_project_client(slug)
        if new_client is app.paperless_client:
            raise ValueError("No dedicated Paperless instance configured. "
                             "Save URL + token on the Connect tab first.")
        if not new_client.health_check():
            raise RuntimeError(f"New Paperless instance is not reachable at {new_client.base_url}")
        old_client = app.paperless_client

        # 2. Enumerate docs from shared instance
        _migration_log(slug, 'enumerating')
        all_docs = []
        page = 1
        while True:
            resp = old_client.get_documents_by_project(
                slug, ordering='-modified', page_size=100, page=page
            )
            all_docs.extend(resp.get('results', []))
            if not resp.get('next'):
                break
            page += 1

        total = len(all_docs)
        _migration_log(slug, 'enumerating', total=total, migrated=0, failed=0)
        logger.info(f"[Migration:{slug}] Found {total} documents to migrate")

        if total == 0:
            status['status'] = 'done'
            status['phase'] = 'complete — 0 documents found'
            return

        # 3. Per-document migration
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=slug)
        doc_id_map = {}  # old_id -> new_id
        migrated = 0
        failed = 0

        for doc in all_docs:
            old_id = doc['id']
            try:
                _migration_log(slug, f'migrating doc {old_id} ({migrated}/{total})',
                               migrated=migrated, total=total, failed=failed)

                content_bytes = old_client.download_document(old_id, archived=True)
                filename = (doc.get('original_file_name') or
                            f"{doc.get('title','document').replace(' ','-')}.pdf")

                task_id = new_client.upload_document_bytes(
                    filename=filename,
                    content=content_bytes,
                    title=doc.get('title'),
                    created=doc.get('created'),
                )
                if not task_id:
                    raise RuntimeError(f"upload returned no task_id for old doc {old_id}")

                new_id = new_client.resolve_task_to_doc_id(task_id, timeout=180)
                if not new_id:
                    raise RuntimeError(f"OCR timed out for task {task_id} (old doc {old_id})")

                doc_id_map[old_id] = new_id

                # Re-key ChromaDB
                if vs.enabled:
                    try:
                        old_str = str(old_id)
                        existing = vs.collection.get(
                            ids=[old_str],
                            include=['embeddings', 'documents', 'metadatas']
                        )
                        if existing['ids']:
                            emb = existing['embeddings'][0] if existing.get('embeddings') else None
                            doc_text = existing['documents'][0] if existing.get('documents') else ''
                            meta = existing['metadatas'][0] if existing.get('metadatas') else {}
                            meta['document_id'] = new_id
                            vs.collection.delete(ids=[old_str])
                            add_kw = {'ids': [str(new_id)], 'documents': [doc_text], 'metadatas': [meta]}
                            if emb:
                                add_kw['embeddings'] = [emb]
                            vs.collection.add(**add_kw)
                    except Exception as _ce:
                        logger.warning(f"[Migration:{slug}] ChromaDB re-key failed for {old_id}: {_ce}")

                # Update processed_documents
                try:
                    with _sqlite3.connect('/app/data/app.db') as conn:
                        conn.execute(
                            "UPDATE processed_documents SET doc_id = ? WHERE doc_id = ? AND project_slug = ?",
                            (new_id, old_id, slug)
                        )
                        conn.commit()
                except Exception as _dbe:
                    logger.warning(f"[Migration:{slug}] DB update failed for {old_id}: {_dbe}")

                migrated += 1
                _migration_log(slug, f'migrated {migrated}/{total}',
                               migrated=migrated, total=total, failed=failed)

            except Exception as e:
                failed += 1
                logger.error(f"[Migration:{slug}] Failed to migrate doc {old_id}: {e}")
                _migration_log(slug, f'error on doc {old_id}: {e}',
                               migrated=migrated, total=total, failed=failed)

        # 4. Patch chat history
        if doc_id_map:
            _migration_log(slug, 'patching chat history', migrated=migrated, total=total)
            try:
                with _sqlite3.connect('/app/data/app.db') as conn:
                    sess_ids = [r[0] for r in conn.execute(
                        "SELECT id FROM chat_sessions WHERE project_slug = ?", (slug,)
                    ).fetchall()]
                    if sess_ids:
                        ph = ','.join('?' * len(sess_ids))
                        rows = conn.execute(
                            f"SELECT id, content FROM chat_messages WHERE session_id IN ({ph})",
                            sess_ids
                        ).fetchall()
                        updates = []
                        for msg_id, content in rows:
                            nc = content
                            for oid, nid in doc_id_map.items():
                                nc = _re.sub(
                                    rf'(?<=#){oid}(?!\d)',
                                    str(nid), nc
                                )
                            if nc != content:
                                updates.append((nc, msg_id))
                        if updates:
                            conn.executemany("UPDATE chat_messages SET content = ? WHERE id = ?", updates)
                            conn.commit()
                            logger.info(f"[Migration:{slug}] Patched {len(updates)} chat messages")
            except Exception as _che:
                logger.warning(f"[Migration:{slug}] Chat history patch failed: {_che}")

        # 5. Update court_imported_docs (tables live in projects.db)
        if doc_id_map:
            _migration_log(slug, 'updating court_imported_docs')
            try:
                with _sqlite3.connect('/app/data/projects.db') as conn:
                    for oid, nid in doc_id_map.items():
                        conn.execute(
                            "UPDATE court_imported_docs SET paperless_doc_id = ? "
                            "WHERE paperless_doc_id = ? AND project_slug = ?",
                            (nid, oid, slug)
                        )
                    conn.commit()
            except Exception as _cde:
                logger.warning(f"[Migration:{slug}] court_imported_docs update failed: {_cde}")

        status['status'] = 'done'
        status['doc_id_map'] = {str(k): v for k, v in doc_id_map.items()}
        status['phase'] = f'complete — {migrated}/{total} migrated, {failed} failed'
        logger.info(f"[Migration:{slug}] Complete: {migrated}/{total} migrated, {failed} failed")

    except Exception as e:
        status['status'] = 'error'
        status['error'] = str(e)
        status['phase'] = f'error: {e}'
        logger.error(f"[Migration:{slug}] Migration failed: {e}", exc_info=True)


@app.route('/api/projects/<slug>/paperless-health-check', methods=['POST'])
@login_required
def api_paperless_health_check(slug):
    """Test-connect a Paperless URL + token without saving. Returns {ok, message/error}."""
    try:
        data = request.json or {}
        url = (data.get('url') or '').strip().rstrip('/')
        token = (data.get('token') or '').strip()
        if not url:
            return jsonify({'ok': False, 'error': 'url is required'}), 400
        # Fall back to saved token when none supplied (allows testing without re-entering)
        if not token and app.project_manager:
            cfg = app.project_manager.get_paperless_config(slug)
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


@app.route('/api/projects/<slug>/doc-link/<int:doc_id>', methods=['GET'])
@login_required
def api_project_doc_link(slug, doc_id):
    """Return the public Paperless URL for a specific document in a project."""
    if not app.project_manager:
        return jsonify({'url': None})
    try:
        cfg = app.project_manager.get_paperless_config(slug)
        base = (cfg.get('doc_base_url') or '').rstrip('/')
        url = f"{base}/documents/{doc_id}/details" if base else None
        return jsonify({'url': url})
    except Exception as e:
        logger.error(f"doc-link error for {slug}/{doc_id}: {e}")
        return jsonify({'url': None})


@app.route('/api/projects/<slug>/documents', methods=['GET'])
@login_required
def api_list_project_documents(slug):
    """List all documents in a project's Chroma collection (lazy-loaded for expand panel)."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=slug)
        if not vs.enabled:
            return jsonify({'documents': [], 'count': 0})
        raw = vs.collection.get(include=['metadatas'])
        # Fetch per-project Paperless doc base URL for "View in Paperless" links.
        # Fall back to PAPERLESS_PUBLIC_BASE_URL env var for projects without a
        # dedicated Paperless instance (all existing projects until configured).
        try:
            _pcfg = app.project_manager.get_paperless_config(slug) if app.project_manager else {}
            _base_url = (_pcfg.get('doc_base_url') or '').rstrip('/')
        except Exception:
            _base_url = ''
        if not _base_url:
            _base_url = os.environ.get('PAPERLESS_PUBLIC_BASE_URL', '').rstrip('/')

        docs = []
        for i, doc_id in enumerate(raw.get('ids', [])):
            m = raw['metadatas'][i]
            anomalies_str = m.get('anomalies', '')
            anomalies_list = [a.strip() for a in anomalies_str.split(',') if a.strip()] if anomalies_str else []
            _did = int(m.get('document_id', 0))
            _link = f"{_base_url}/documents/{_did}/details" if _base_url and _did else None
            docs.append({
                'doc_id':        _did,
                'title':         m.get('title', 'Untitled'),
                'timestamp':     m.get('timestamp', ''),
                'brief_summary': m.get('brief_summary', ''),
                'full_summary':  m.get('full_summary', ''),
                'anomalies':     anomalies_list,
                'risk_score':    float(m.get('risk_score') or 0),
                'paperless_link': _link,
            })
        docs.sort(key=lambda x: x['doc_id'])
        return jsonify({'documents': docs, 'count': len(docs)})
    except Exception as e:
        logger.error(f"Failed to list project documents for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/documents/<int:doc_id>', methods=['DELETE'])
@login_required
def api_delete_project_document(slug, doc_id):
    """Delete a document from Paperless-ngx, Chroma, and processed_documents."""
    import sqlite3 as _sqlite3
    warnings = []

    # 1. Delete from Paperless-ngx (use per-project client if configured)
    try:
        pc = _get_project_client(slug)
        if pc:
            url = f"{pc.base_url}/api/documents/{doc_id}/"
            resp = pc.session.delete(url)
            if resp.status_code not in (204, 404):
                resp.raise_for_status()
    except Exception as e:
        warnings.append(f"Paperless: {e}")
        logger.warning(f"Could not delete doc {doc_id} from Paperless: {e}")

    # 2. Delete from Chroma vector store
    try:
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=slug)
        if vs.enabled:
            vs.delete_document(doc_id)
    except Exception as e:
        warnings.append(f"Chroma: {e}")
        logger.warning(f"Could not delete doc {doc_id} from Chroma (slug={slug}): {e}")

    # 3. Delete from processed_documents
    try:
        with _sqlite3.connect('/app/data/app.db') as conn:
            conn.execute("DELETE FROM processed_documents WHERE doc_id = ?", (doc_id,))
    except Exception as e:
        warnings.append(f"DB: {e}")
        logger.warning(f"Could not delete doc {doc_id} from processed_documents: {e}")

    if warnings:
        return jsonify({'success': True, 'warnings': warnings,
                        'message': f'Document {doc_id} removed with warnings: ' + '; '.join(warnings)})

    logger.info(f"Deleted document {doc_id} from project {slug} (Paperless + Chroma + DB)")
    return jsonify({'success': True, 'message': f'Document {doc_id} deleted'})


@app.route('/api/current-project', methods=['GET'])
@login_required
def api_get_current_project():
    """Get currently selected project from session."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        from flask import session
        project_slug = session.get('current_project', 'default')
        project = app.project_manager.get_project(project_slug)

        if not project:
            # Fallback to default
            project = app.project_manager.get_project('default')
            session['current_project'] = 'default'

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to get current project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/current-project', methods=['POST'])
@login_required
def api_set_current_project():
    """Set current project in session."""
    if not app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        from flask import session
        data = request.json
        project_slug = data.get('project_slug')

        if not project_slug:
            return jsonify({'error': 'project_slug is required'}), 400

        # Validate project exists
        project = app.project_manager.get_project(project_slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        session['current_project'] = project_slug
        logger.info(f"Switched to project: {project_slug}")

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to set current project: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orphan-documents', methods=['GET'])
@login_required
def api_list_orphan_documents():
    """List documents without project tag."""
    if not app.paperless_client:
        return jsonify({'error': 'Paperless client not available'}), 503

    try:
        orphans = app.paperless_client.get_documents_without_project()

        # Format for UI
        orphan_list = []
        for doc in orphans[:100]:  # Limit to 100 for performance
            orphan_list.append({
                'id': doc['id'],
                'title': doc['title'],
                'created': doc.get('created'),
                'correspondent': doc.get('correspondent'),
                'tags': [t['name'] for t in doc.get('tags', [])]
            })

        return jsonify({'orphans': orphan_list, 'count': len(orphan_list)})

    except Exception as e:
        logger.error(f"Failed to get orphan documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/assign-project', methods=['POST'])
@login_required
def api_assign_project_to_documents():
    """Assign project to one or more documents."""
    if not app.paperless_client or not app.project_manager:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = request.json
        document_ids = data.get('document_ids', [])
        project_slug = data.get('project_slug')

        if not document_ids or not project_slug:
            return jsonify({'error': 'document_ids and project_slug are required'}), 400

        # Validate project
        project = app.project_manager.get_project(project_slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Add project tag to each document
        success_count = 0
        failed = []

        for doc_id in document_ids:
            if app.paperless_client.add_project_tag(doc_id, project_slug, color=project.get('color')):
                success_count += 1
            else:
                failed.append(doc_id)

        # Update project document count
        if success_count > 0:
            app.project_manager.increment_document_count(project_slug, delta=success_count)

        result = {
            'success': True,
            'assigned': success_count,
            'total': len(document_ids)
        }

        if failed:
            result['failed'] = failed

        logger.info(f"Assigned {success_count}/{len(document_ids)} documents to project '{project_slug}'")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to assign documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<slug>/reanalyze', methods=['POST'])
@login_required
def api_reanalyze_project(slug):
    """Manually trigger re-analysis of all documents in a project with full context."""
    if not app.project_manager or not app.document_analyzer:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        # Validate project exists
        project = app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        logger.info(f"Manual re-analysis requested for project '{slug}'")

        # Trigger re-analysis in background thread to avoid blocking
        from threading import Thread
        def run_reanalysis():
            try:
                app.document_analyzer.re_analyze_project(slug)
            except Exception as e:
                logger.error(f"Background re-analysis failed: {e}")

        thread = Thread(target=run_reanalysis, daemon=True)
        thread.start()

        # Return immediately
        return jsonify({
            'success': True,
            'message': f'Re-analysis started for project: {slug}',
            'note': 'This runs in background and may take several minutes. Check logs for progress.'
        })

    except Exception as e:
        logger.error(f"Failed to trigger re-analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/migrate-documents', methods=['POST'])
@login_required
@admin_required
def api_migrate_documents():
    """Migrate documents from one project to another."""
    if not app.project_manager or not app.paperless_client:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = request.json
        source_slug = data.get('source_project')
        dest_slug = data.get('destination_project')
        document_ids = data.get('document_ids', [])  # Optional: specific documents, or all if empty

        if not source_slug or not dest_slug:
            return jsonify({'error': 'Both source_project and destination_project are required'}), 400

        if source_slug == dest_slug:
            return jsonify({'error': 'Source and destination projects must be different'}), 400

        # Validate both projects exist
        source_project = app.project_manager.get_project(source_slug)
        dest_project = app.project_manager.get_project(dest_slug)

        if not source_project:
            return jsonify({'error': f'Source project not found: {source_slug}'}), 404
        if not dest_project:
            return jsonify({'error': f'Destination project not found: {dest_slug}'}), 404

        logger.info(f"Migration requested: {source_slug} → {dest_slug} ({len(document_ids) if document_ids else 'all'} docs)")

        # Run migration in background thread
        from threading import Thread
        migration_result = {'status': 'running', 'migrated': 0, 'errors': 0}

        def run_migration():
            try:
                from analyzer.vector_store import VectorStore
                import sqlite3

                source_tag = f"project:{source_slug}"
                dest_tag = f"project:{dest_slug}"

                source_tag_id = app.paperless_client.get_or_create_tag(source_tag, color='#95a5a6')
                dest_tag_id = app.paperless_client.get_or_create_tag(dest_tag, color='#e74c3c')

                if not source_tag_id or not dest_tag_id:
                    raise Exception("Failed to get/create project tags")

                # Use Chroma as the source of truth for which docs belong to the source
                # project. Paperless tag lookup misses docs that predate the projects
                # feature (they live in the default Chroma collection but have no
                # project:default tag in Paperless).
                source_vs = VectorStore(project_slug=source_slug)
                dest_vs = VectorStore(project_slug=dest_slug)

                if document_ids:
                    docs_to_migrate = [int(d) for d in document_ids]
                else:
                    chroma_all = source_vs.collection.get(include=['metadatas'])
                    docs_to_migrate = [int(i) for i in chroma_all['ids']]

                logger.info(f"Migration: {len(docs_to_migrate)} docs from {source_slug} → {dest_slug}")

                migrated_count = 0
                error_count = 0

                for doc_id in docs_to_migrate:
                    try:
                        # ── 1. Move Chroma embedding ──────────────────────────────
                        result = source_vs.collection.get(
                            ids=[str(doc_id)],
                            include=['embeddings', 'metadatas', 'documents']
                        )
                        if result['ids']:
                            meta = dict(result['metadatas'][0])
                            dest_vs.collection.upsert(
                                ids=[str(doc_id)],
                                embeddings=result['embeddings'],
                                metadatas=[meta],
                                documents=result['documents'],
                            )
                            source_vs.collection.delete(ids=[str(doc_id)])

                        # ── 2. Update Paperless tags ──────────────────────────────
                        doc = app.paperless_client.get_document(doc_id)
                        if doc:
                            current_tags = doc.get('tags', [])
                            updated_tags = [t for t in current_tags if t != source_tag_id]
                            if dest_tag_id not in updated_tags:
                                updated_tags.append(dest_tag_id)
                            app.paperless_client.update_document(doc_id, {'tags': updated_tags})

                        migrated_count += 1

                    except Exception as e:
                        logger.error(f"Failed to migrate document {doc_id}: {e}")
                        error_count += 1

                # ── 3. Update processed_documents.project_slug ───────────────────
                try:
                    db_path = '/app/data/app.db'
                    con = sqlite3.connect(db_path)
                    if document_ids:
                        # Partial migration — update only the specific docs
                        for doc_id in docs_to_migrate:
                            con.execute(
                                'UPDATE processed_documents SET project_slug=? WHERE doc_id=?',
                                (dest_slug, doc_id)
                            )
                    else:
                        # Full migration — batch update all docs in source project
                        con.execute(
                            'UPDATE processed_documents SET project_slug=? WHERE project_slug=?',
                            (dest_slug, source_slug)
                        )
                    con.commit()
                    con.close()
                    logger.info(f"Updated processed_documents: {source_slug} → {dest_slug}")
                except Exception as e:
                    logger.warning(f"Could not update processed_documents after migration: {e}")

                # ── 4. Migrate chat sessions (and their messages via CASCADE) ────
                try:
                    db_path = '/app/data/app.db'
                    con = sqlite3.connect(db_path)
                    session_rows = con.execute(
                        'UPDATE chat_sessions SET project_slug=? WHERE project_slug=?',
                        (dest_slug, source_slug)
                    )
                    con.commit()
                    chat_count = session_rows.rowcount
                    con.close()
                    logger.info(f"Migrated {chat_count} chat session(s) from {source_slug} → {dest_slug}")
                except Exception as e:
                    logger.warning(f"Could not migrate chat sessions: {e}")

                # ── 5. Migrate Case Intelligence runs ─────────────────────────────
                # ci_runs.project_slug is the only field that needs updating;
                # all child tables (entities, events, contradictions, theories,
                # reports, etc.) reference run_id and follow automatically.
                try:
                    ci_db_path = '/app/data/case_intelligence.db'
                    con = sqlite3.connect(ci_db_path)
                    r = con.execute(
                        'UPDATE ci_runs SET project_slug=? WHERE project_slug=?',
                        (dest_slug, source_slug)
                    )
                    con.commit()
                    ci_count = r.rowcount
                    con.close()
                    logger.info(
                        f"Migrated {ci_count} Case Intelligence run(s) "
                        f"from {source_slug} → {dest_slug}"
                    )
                except Exception as e:
                    logger.warning(f"Could not migrate CI runs: {e}")

                # ── 6. Migrate court import jobs and imported-doc records ──────────
                try:
                    proj_db_path = '/app/data/projects.db'
                    con = sqlite3.connect(proj_db_path)
                    r_jobs = con.execute(
                        'UPDATE court_import_jobs SET project_slug=? WHERE project_slug=?',
                        (dest_slug, source_slug)
                    )
                    r_docs = con.execute(
                        'UPDATE court_imported_docs SET project_slug=? WHERE project_slug=?',
                        (dest_slug, source_slug)
                    )
                    con.commit()
                    con.close()
                    logger.info(
                        f"Migrated {r_jobs.rowcount} court import job(s) and "
                        f"{r_docs.rowcount} court imported doc record(s) "
                        f"from {source_slug} → {dest_slug}"
                    )
                except Exception as e:
                    logger.warning(f"Could not migrate court import data: {e}")

                # ── 8. Refresh project document count cache ───────────────────────
                try:
                    app.project_manager.update_document_count(source_slug, source_vs.collection.count())
                    app.project_manager.update_document_count(dest_slug, dest_vs.collection.count())
                    logger.info(f"Updated project doc counts after migration")
                except Exception as e:
                    logger.warning(f"Could not refresh project doc counts: {e}")

                migration_result['status'] = 'completed'
                migration_result['migrated'] = migrated_count
                migration_result['errors'] = error_count

                logger.info(f"Migration completed: {migrated_count} docs moved to {dest_slug}, {error_count} errors")

            except Exception as e:
                logger.error(f"Migration failed: {e}")
                migration_result['status'] = 'failed'
                migration_result['error'] = str(e)

        thread = Thread(target=run_migration, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'message': f'Migration started: {source_slug} → {dest_slug}',
            'note': 'Migration runs in background. Check logs for progress.'
        })

    except Exception as e:
        logger.error(f"Failed to start migration: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/transform-url', methods=['POST'])
@login_required
def api_transform_upload_url():
    """Transform a cloud share link into a direct download URL.

    Supports Google Drive, Google Docs/Sheets/Slides, Dropbox, and OneDrive.
    For all other URLs the input is passed through unchanged.
    """
    import re as _re
    try:
        data = request.json or {}
        raw_url = data.get('url', '').strip()
        if not raw_url:
            return jsonify({'error': 'url is required'}), 400

        service = 'generic'
        direct_url = raw_url
        filename_hint = None

        # ── Google Drive file ──────────────────────────────────────────
        m = _re.match(
            r'https://drive\.google\.com/file/d/([^/?#]+)',
            raw_url, _re.IGNORECASE
        )
        if m:
            file_id = m.group(1)
            direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            service = 'google_drive'

        # ── Google Docs / Sheets / Slides ──────────────────────────────
        if service == 'generic':
            m = _re.match(
                r'https://docs\.google\.com/(document|spreadsheets|presentation)/d/([^/?#]+)',
                raw_url, _re.IGNORECASE
            )
            if m:
                doc_type = m.group(1)
                file_id = m.group(2)
                fmt_map = {
                    'document': 'pdf',
                    'spreadsheets': 'pdf',
                    'presentation': 'pdf',
                }
                fmt = fmt_map.get(doc_type, 'pdf')
                direct_url = (
                    f'https://docs.google.com/{doc_type}/d/{file_id}/export?format={fmt}'
                )
                service = 'google_drive'
                filename_hint = f'document.{fmt}'

        # ── Dropbox ────────────────────────────────────────────────────
        if service == 'generic' and 'dropbox.com' in raw_url.lower():
            direct_url = _re.sub(r'[?&]dl=0', lambda m2: m2.group(0).replace('dl=0', 'dl=1'), raw_url)
            if 'dl=1' not in direct_url:
                sep = '&' if '?' in direct_url else '?'
                direct_url = direct_url + sep + 'dl=1'
            service = 'dropbox'

        # ── OneDrive 1drv.ms ──────────────────────────────────────────
        if service == 'generic' and '1drv.ms' in raw_url.lower():
            service = 'onedrive'
            # RemoteFileDownloader follows the redirect; pass through unchanged

        logger.info(f"transform-url: {service} → {direct_url[:80]}")
        return jsonify({
            'direct_url': direct_url,
            'service': service,
            'filename_hint': filename_hint,
        })

    except Exception as e:
        logger.error(f"transform-url error: {e}")
        return jsonify({'error': str(e)}), 500

# Allowed file extensions for directory-scan (OCR-able or text-bearing documents)
_ALLOWED_SCAN_EXTS = {
    '.pdf',
    '.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif', '.bmp', '.webp', '.heic', '.heif',
    '.docx', '.doc', '.odt', '.rtf', '.txt', '.md', '.rst',
    '.xlsx', '.xls', '.ods', '.csv',
    '.pptx', '.ppt', '.odp',
    '.eml', '.msg',
    '.djvu', '.epub',
}


@app.route('/api/upload/scan-url', methods=['POST'])
@login_required
def api_scan_url():
    """Probe a URL: return single-file info OR a list of compatible file links found on an
    HTML directory-listing page.  Used by the Upload tab to support folder-level ingestion.
    """
    import re as _re
    from urllib.parse import urljoin, urlparse as _urlparse, unquote as _unquote
    try:
        data = request.json or {}
        url = data.get('url', '').strip()
        auth_type = data.get('auth_type', 'none')
        username = data.get('username')
        password = data.get('password')
        token = data.get('token')

        if not url:
            return jsonify({'error': 'url is required'}), 400

        import requests as _req

        session_auth = None
        req_headers = {'User-Agent': 'Paperless-AI-Analyzer/2.0'}
        if auth_type == 'basic' and username and password:
            session_auth = (username, password)
        elif auth_type == 'token' and token:
            req_headers['Authorization'] = f'Bearer {token}'

        # ── Step 1: HEAD the URL ────────────────────────────────────────────────
        try:
            head = _req.head(url, auth=session_auth, headers=req_headers,
                             allow_redirects=True, timeout=15)
            content_type = head.headers.get('content-type', '').lower().split(';')[0].strip()
        except Exception as e:
            return jsonify({'error': f'Could not reach URL: {e}'}), 400

        # ── Step 2: Check file extension ────────────────────────────────────────
        path = _urlparse(url).path
        m = _re.search(r'(\.[a-z0-9]{2,6})(?:[?#]|$)', path.lower())
        file_ext = m.group(1) if m else ''

        if file_ext in _ALLOWED_SCAN_EXTS:
            size = int(head.headers.get('content-length', 0))
            return jsonify({
                'type': 'single',
                'url': url,
                'filename': _unquote(path.split('/')[-1]) or 'document',
                'size_bytes': size,
                'ext': file_ext,
            })

        # ── Step 3: Not a known file extension — parse as HTML listing ──────────
        if 'text/html' not in content_type:
            # Non-HTML, no recognised extension: treat as opaque single file
            size = int(head.headers.get('content-length', 0))
            fname = _unquote(path.split('/')[-1]) or 'document'
            return jsonify({'type': 'single', 'url': url,
                            'filename': fname, 'size_bytes': size, 'ext': file_ext})

        # Fetch the HTML page
        try:
            resp = _req.get(url, auth=session_auth, headers=req_headers,
                            allow_redirects=True, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            return jsonify({'error': f'Failed to fetch page: {e}'}), 400

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')
        base_url = resp.url  # after any redirects

        files = []
        seen = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith('#') or href.startswith('?'):
                continue
            full_url = urljoin(base_url, href)
            link_path = _urlparse(full_url).path
            lm = _re.search(r'(\.[a-z0-9]{2,6})(?:[?#]|$)', link_path.lower())
            lext = lm.group(1) if lm else ''
            if lext not in _ALLOWED_SCAN_EXTS:
                continue
            if full_url in seen:
                continue
            seen.add(full_url)
            filename = _unquote(link_path.split('/')[-1]) or 'document'
            # Best-effort file size (short timeout, non-fatal)
            size = 0
            try:
                fh = _req.head(full_url, auth=session_auth, headers=req_headers,
                               allow_redirects=True, timeout=4)
                size = int(fh.headers.get('content-length', 0))
            except Exception:
                pass
            files.append({'filename': filename, 'url': full_url,
                          'size_bytes': size, 'ext': lext})

        if not files:
            return jsonify({
                'error': 'No compatible files found at this URL. '
                         'Supported types: PDF, images, Word/Excel/ODT, TXT, EML and more.'
            }), 404

        logger.info(f"scan-url: found {len(files)} compatible files at {url[:80]}")
        return jsonify({'type': 'directory', 'base_url': base_url, 'files': files})

    except Exception as e:
        logger.error(f"scan-url error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/history', methods=['GET'])
@login_required
def api_upload_history():
    """Return the last 20 import history rows for the current user."""
    try:
        rows = get_import_history(current_user.id)
        return jsonify({'history': [dict(r) for r in rows]})
    except Exception as e:
        logger.error(f"Failed to fetch upload history: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/from-url', methods=['POST'])
@login_required
def api_upload_from_url():
    """Download file from URL and upload to Paperless."""
    try:
        data = request.json
        url = data.get('url')
        source = data.get('source', 'url')          # e.g. 'google_drive', 'dropbox', 'url'
        project_slug = data.get('project_slug')     # optional: set when smart metadata confirmed
        metadata_in = data.get('metadata', {})      # optional: AI metadata from prior analyze step
        auth_type = data.get('auth_type', 'none')
        username = data.get('username')
        password = data.get('password')
        token = data.get('token')
        custom_headers = data.get('custom_headers', {})

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # URL-based dedup: skip if already successfully imported from this exact URL
        try:
            from analyzer.db import _get_conn as _db_conn_dedup
            with _db_conn_dedup() as _dc:
                _existing = _dc.execute(
                    "SELECT id, filename, paperless_doc_id FROM import_history"
                    " WHERE original_url = ? AND status = 'uploaded' LIMIT 1",
                    (url,)
                ).fetchone()
            if _existing:
                logger.info(f"URL dedup: skipping already-imported URL {url[:80]}")
                return jsonify({
                    'success': True,
                    'duplicate': True,
                    'document_id': _existing['paperless_doc_id'],
                    'title': _existing['filename'],
                    'detail': 'Already imported from this URL',
                }), 200
        except Exception as _dedup_err:
            logger.warning(f"URL dedup check failed (non-fatal): {_dedup_err}")

        # Download file
        from analyzer.remote_downloader import RemoteFileDownloader
        downloader = RemoteFileDownloader()
        try:
            file_path, download_metadata = downloader.download_from_url(
                url=url,
                auth_type=auth_type,
                username=username,
                password=password,
                token=token,
                custom_headers=custom_headers
            )
        except Exception as e:
            log_import(current_user.id, source, url, url=url,
                       status='error', error=f'Download failed: {str(e)}')
            return jsonify({'error': f'Download failed: {str(e)}'}), 400

        filename = download_metadata.get('filename', 'document')

        try:
            # If smart metadata confirmed (project_slug + metadata provided), use SmartUploader
            if project_slug and metadata_in and app.smart_uploader:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        app.smart_uploader.upload_to_paperless(file_path, project_slug, metadata_in)
                    )
                finally:
                    loop.close()
                title = metadata_in.get('suggested_title') or filename
            else:
                # Direct upload without AI metadata — still apply project tag if selected
                tag_ids = []
                if project_slug and app.paperless_client:
                    proj_tag = app.paperless_client.get_or_create_tag(f"project:{project_slug}")
                    if proj_tag is not None:
                        tag_ids.append(proj_tag)
                result = app.paperless_client.upload_document(file_path, title=filename,
                                                              tags=tag_ids or None)
                title = filename
        finally:
            downloader.cleanup(file_path)

        if result:
            doc_id = result.get('id')
            if project_slug and not metadata_in and app.project_manager:
                app.project_manager.increment_document_count(project_slug, delta=1)
            logger.info(f"Uploaded from URL ({source}): {filename}")
            log_import(current_user.id, source, title, url=url,
                       doc_id=doc_id, status='uploaded')
            return jsonify({'success': True, 'document_id': doc_id, 'title': title})
        else:
            log_import(current_user.id, source, filename, url=url,
                       status='error', error='Upload to Paperless failed')
            return jsonify({'error': 'Upload failed'}), 500

    except Exception as e:
        logger.error(f"Failed to upload from URL: {e}")
        try:
            _lc = locals()
            log_import(current_user.id,
                       _lc.get('source', 'url'),
                       _lc.get('url', 'unknown'),
                       url=_lc.get('url'),
                       status='error', error=str(e))
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/analyze', methods=['POST'])
@login_required
def api_analyze_upload():
    """Analyze uploaded file and extract metadata."""
    if not app.smart_uploader:
        return jsonify({'error': 'Smart upload not available (LLM disabled)'}), 503

    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file temporarily
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Analyze document
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metadata = loop.run_until_complete(app.smart_uploader.analyze_document(file_path))
        finally:
            loop.close()

        # Clean up temp file
        os.remove(file_path)
        os.rmdir(temp_dir)

        logger.info(f"Analyzed upload: {file.filename}")
        return jsonify(metadata)

    except Exception as e:
        logger.error(f"Failed to analyze upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/submit', methods=['POST'])
@login_required
def api_submit_upload():
    """Submit file to Paperless. Supports direct upload and smart-metadata upload."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get optional metadata from form data
        project_slug = request.form.get('project_slug')
        import json as _json
        metadata_json = request.form.get('metadata', '{}')
        metadata = _json.loads(metadata_json)

        # Save file temporarily
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        try:
            # Smart metadata path: project_slug + metadata provided → use SmartUploader
            if project_slug and metadata and app.smart_uploader:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        app.smart_uploader.upload_to_paperless(file_path, project_slug, metadata)
                    )
                finally:
                    loop.close()
                title = metadata.get('suggested_title') or file.filename
            else:
                # Direct upload without AI metadata — still apply project tag if selected
                tag_ids = []
                if project_slug and app.paperless_client:
                    proj_tag = app.paperless_client.get_or_create_tag(f"project:{project_slug}")
                    if proj_tag is not None:
                        tag_ids.append(proj_tag)
                result = app.paperless_client.upload_document(
                    file_path, title=file.filename.rsplit('.', 1)[0],
                    tags=tag_ids or None
                )
                title = file.filename
        finally:
            os.remove(file_path)
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

        if result:
            doc_id = result.get('id')
            if project_slug and not metadata and app.project_manager:
                app.project_manager.increment_document_count(project_slug, delta=1)
            logger.info(f"Uploaded {file.filename} (project={project_slug or 'none'})")
            log_import(current_user.id, 'file', title,
                       doc_id=doc_id, status='uploaded')
            return jsonify({'success': True, 'document_id': doc_id, 'title': title})
        else:
            log_import(current_user.id, 'file', file.filename,
                       status='error', error='Upload to Paperless failed')
            return jsonify({'error': 'Upload failed'}), 500

    except Exception as e:
        logger.error(f"Failed to submit upload: {e}")
        try:
            fname = request.files.get('file', None)
            log_import(current_user.id, 'file',
                       fname.filename if fname else 'unknown',
                       status='error', error=str(e))
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm-usage/stats', methods=['GET'])
@login_required
def api_get_llm_usage_stats():
    """Get LLM usage statistics."""
    try:
        days = request.args.get('days', 30, type=int)

        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(app.document_analyzer, 'usage_tracker') or not app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        stats = app.document_analyzer.usage_tracker.get_usage_stats(days=days)
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Failed to get usage stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm-usage/recent', methods=['GET'])
@login_required
def api_get_recent_llm_calls():
    """Get recent LLM API calls."""
    try:
        limit = request.args.get('limit', 50, type=int)

        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(app.document_analyzer, 'usage_tracker') or not app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        calls = app.document_analyzer.usage_tracker.get_recent_calls(limit=limit)
        return jsonify({'calls': calls})

    except Exception as e:
        logger.error(f"Failed to get recent calls: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm-usage/pricing', methods=['GET'])
@login_required
def api_get_llm_pricing():
    """Get current LLM pricing information."""
    try:
        if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
            return jsonify({'error': 'Document analyzer not available'}), 503

        if not hasattr(app.document_analyzer, 'usage_tracker') or not app.document_analyzer.usage_tracker:
            return jsonify({'error': 'Usage tracker not available'}), 503

        pricing = app.document_analyzer.usage_tracker.get_pricing()
        return jsonify({'pricing': pricing})

    except Exception as e:
        logger.error(f"Failed to get pricing: {e}")
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# SMTP utilities
# ---------------------------------------------------------------------------
_SMTP_SETTINGS_FILE = Path('/app/data/smtp_settings.json')
_SMTP_DEFAULTS = {
    'host': '', 'port': 587, 'starttls': True,
    'user': '', 'pass': '', 'from': '', 'helo': '',
    'bug_report_to': 'dblagbro@voipguru.org',
}

def _load_smtp_settings() -> dict:
    try:
        if _SMTP_SETTINGS_FILE.exists():
            return {**_SMTP_DEFAULTS, **json.loads(_SMTP_SETTINGS_FILE.read_text())}
    except Exception:
        pass
    return dict(_SMTP_DEFAULTS)

def _save_smtp_settings(settings: dict):
    _SMTP_SETTINGS_FILE.write_text(json.dumps(settings, indent=2))

def _smtp_send(smtp_cfg: dict, msg: EmailMessage):
    host = smtp_cfg.get('host', '')
    port = int(smtp_cfg.get('port', 587))
    starttls = bool(smtp_cfg.get('starttls', True))
    user = smtp_cfg.get('user', '')
    pwd = smtp_cfg.get('pass', '')
    helo = smtp_cfg.get('helo') or None
    if not host:
        raise RuntimeError('SMTP host is not configured')
    with smtplib.SMTP(host, port, local_hostname=helo) as s:
        s.ehlo()
        if starttls:
            s.starttls(context=ssl.create_default_context())
            s.ehlo()
        if user:
            s.login(user, pwd)
        s.send_message(msg)


def _send_welcome_email(email: str, display_name: str, username: str, role: str, app_base_url: str, job_title: str = ''):
    """Send a welcome notification to a newly created user.  Does not include the password."""
    try:
        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info(f"SMTP not configured — skipping welcome email for {username}")
            return

        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        version = _APP_VERSION
        base = app_base_url.rstrip('/')
        docs_url = f"{base}/docs"
        github_url = 'https://github.com/dblagbro/paperless-ai-analyzer'
        job_title_line = f"\n  Job Title: {job_title}" if job_title else ""

        body = f"""Hi {display_name},

Your Paperless AI Analyzer account has been created and is ready to use.

Account Details
───────────────
  Username : {username}
  Role     : {role.capitalize()}{job_title_line}

Access the Application
──────────────────────
  {base}/

User Manual
───────────
  The full user manual is available at:
  {docs_url}

  Key sections:
  • Quick Start          {docs_url}/getting-started
  • Projects             {docs_url}/projects
  • Smart Upload         {docs_url}/upload
  • AI Chat              {docs_url}/chat
  • Search & Analysis    {docs_url}/search
  • Anomaly Detection    {docs_url}/anomaly-detection
  • Configuration        {docs_url}/configuration

Resources
─────────
  GitHub / README : {github_url}#readme

If you have any questions, please contact your system administrator.

—
Paperless AI Analyzer v{version}
"""
        msg = EmailMessage()
        msg['Subject'] = 'Welcome to Paperless AI Analyzer — Your Account is Ready'
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"Welcome email sent to {email} for user '{username}'")
    except Exception as e:
        logger.warning(f"Failed to send welcome email to {email}: {e}")


def _send_manual_email(email: str, display_name: str, app_base_url: str):
    """Send the user manual link to a user."""
    try:
        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            raise RuntimeError("SMTP is not configured")

        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        version = _APP_VERSION
        base = app_base_url.rstrip('/')
        docs_url = f"{base}/docs"

        body = f"""Hi {display_name},

Here is a link to the Paperless AI Analyzer user manual:

  {docs_url}

Manual Sections
───────────────
  Overview & Features    {docs_url}/overview
  Quick Start Guide      {docs_url}/getting-started
  Projects               {docs_url}/projects
  Smart Upload           {docs_url}/upload
  AI Chat                {docs_url}/chat
  Search & Analysis      {docs_url}/search
  Anomaly Detection      {docs_url}/anomaly-detection
  Debug & Tools          {docs_url}/tools
  Configuration          {docs_url}/configuration
  User Management        {docs_url}/users
  LLM Usage & Cost       {docs_url}/llm-usage
  API Reference          {docs_url}/api

—
Paperless AI Analyzer v{version}
"""
        msg = EmailMessage()
        msg['Subject'] = 'Paperless AI Analyzer — User Manual'
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"Manual email sent to {email}")
    except Exception as e:
        logger.warning(f"Failed to send manual email to {email}: {e}")
        raise


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
@app.route('/api/about')
@login_required
def api_about():
    return jsonify({
        'name': 'Paperless AI Analyzer',
        'version': _APP_VERSION,
        'description': (
            'Intelligent document analysis, anomaly detection, and AI-powered '
            'chat interface for Paperless-ngx.'
        ),
        'components': {
            'paperless-ai-analyzer': _APP_VERSION,
            'anomaly-detector': '1.5.1',
        },
        'github': 'https://github.com/dblagbro/paperless-ai-analyzer',
    })


# ---------------------------------------------------------------------------
# User Manual / Docs
# ---------------------------------------------------------------------------

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

@app.route('/docs')
@app.route('/docs/')
@app.route('/docs/<path:page>')
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


# ---------------------------------------------------------------------------
# Docs AI help endpoint
# ---------------------------------------------------------------------------
@app.route('/api/docs/ask', methods=['POST'])
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
        provider = ai_cfg.get('provider', 'openai')
        model = ai_cfg.get('model', 'gpt-4o')
        api_key = ai_cfg.get('api_key', '')

        # Try primary, then fallback
        answer = None
        for _p, _m, _k in [
            (provider, model, api_key),
            (ai_cfg.get('fallback_provider', ''), ai_cfg.get('fallback_model', ''),
             load_ai_config().get('global', {}).get(ai_cfg.get('fallback_provider', ''), {}).get('api_key', '')),
        ]:
            if not _p or not _k:
                continue
            try:
                if _p == 'openai':
                    import openai as _oai
                    resp = _oai.OpenAI(api_key=_k).chat.completions.create(
                        model=_m,
                        messages=[{'role': 'system', 'content': system_prompt}] + messages,
                        max_tokens=1024,
                    )
                    answer = resp.choices[0].message.content
                elif _p == 'anthropic':
                    import anthropic as _ant
                    resp = _ant.Anthropic(api_key=_k).messages.create(
                        model=_m, max_tokens=1024,
                        system=system_prompt, messages=messages,
                    )
                    answer = resp.content[0].text
                if answer:
                    break
            except Exception as e:
                logger.warning(f"docs/ask {_p}: {e}")
                continue

        if not answer:
            return jsonify({'error': 'No AI provider available — check Configuration → AI Settings.'}), 503
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"docs/ask error: {e}")
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# SMTP / Notifications settings (admin only)
# ---------------------------------------------------------------------------
@app.route('/api/smtp-settings', methods=['GET'])
@login_required
def api_get_smtp_settings():
    if not current_user.role == 'admin':
        return jsonify({'error': 'Admin only'}), 403
    s = _load_smtp_settings()
    # Mask password for display
    s_safe = {**s, 'pass': '••••••••' if s.get('pass') else ''}
    return jsonify(s_safe)


@app.route('/api/smtp-settings', methods=['POST'])
@login_required
def api_save_smtp_settings():
    if not current_user.role == 'admin':
        return jsonify({'error': 'Admin only'}), 403
    data = request.get_json(force=True) or {}
    current = _load_smtp_settings()
    updated = {
        'host': str(data.get('host', current.get('host', ''))).strip(),
        'port': int(data.get('port', current.get('port', 587))),
        'starttls': bool(data.get('starttls', current.get('starttls', True))),
        'user': str(data.get('user', current.get('user', ''))).strip(),
        'from': str(data.get('from', current.get('from', ''))).strip(),
        'helo': str(data.get('helo', current.get('helo', ''))).strip(),
        'bug_report_to': str(data.get('bug_report_to', current.get('bug_report_to', 'dblagbro@voipguru.org'))).strip(),
    }
    # Only update password if a real value is provided (not the mask)
    raw_pass = str(data.get('pass', ''))
    if raw_pass and raw_pass != '••••••••':
        updated['pass'] = raw_pass
    else:
        updated['pass'] = current.get('pass', '')
    _save_smtp_settings(updated)
    return jsonify({'ok': True, 'message': 'SMTP settings saved'})


@app.route('/api/smtp-settings/test', methods=['POST'])
@login_required
def api_test_smtp():
    if not current_user.role == 'admin':
        return jsonify({'error': 'Admin only'}), 403
    smtp_cfg = _load_smtp_settings()
    dest = smtp_cfg.get('bug_report_to') or smtp_cfg.get('user', '')
    if not dest:
        return jsonify({'error': 'No destination email configured'}), 400
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Paperless AI Analyzer — SMTP Test'
        msg['From'] = smtp_cfg.get('from') or smtp_cfg.get('user', 'noreply@localhost')
        msg['To'] = dest
        msg.set_content('This is a test email from Paperless AI Analyzer.\n\nSMTP is configured correctly!')
        _smtp_send(smtp_cfg, msg)
        return jsonify({'ok': True, 'message': f'Test email sent to {dest}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Bug Report
# ---------------------------------------------------------------------------
@app.route('/api/bug-report', methods=['POST'])
@login_required
def api_bug_report():
    description = (request.form.get('description') or '').strip()
    severity = (request.form.get('severity') or 'Medium').strip()
    contact_email = (request.form.get('contact_email') or '').strip()
    include_logs = request.form.get('include_logs', 'true').lower() != 'false'
    browser_info = request.headers.get('User-Agent', 'Unknown')

    if not description:
        return jsonify({'error': 'Please describe the problem'}), 400

    smtp_cfg = _load_smtp_settings()
    dest = smtp_cfg.get('bug_report_to', 'dblagbro@voipguru.org')

    # Collect recent log lines from the in-memory buffer
    log_snippet = ''
    if include_logs:
        try:
            buf = list(app.log_buffer)[-60:]
            log_snippet = '\n'.join(buf)
        except Exception:
            log_snippet = '(logs unavailable)'

    # Build email body
    lines = [
        f'Paperless AI Analyzer — Bug Report',
        f'=' * 50,
        f'',
        f'Severity:    {severity}',
        f'Reported by: {current_user.display_name} (user: {current_user.username})',
        f'Version:     {_APP_VERSION}',
        f'Timestamp:   {datetime.utcnow().isoformat()}Z',
        f'Browser:     {browser_info}',
    ]
    if contact_email:
        lines.append(f'Contact:     {contact_email}')
    lines += [
        f'',
        f'DESCRIPTION',
        f'-' * 40,
        description,
        f'',
    ]
    if log_snippet:
        lines += [
            f'RECENT LOGS (last 60 lines)',
            f'-' * 40,
            log_snippet,
            f'',
        ]

    body_text = '\n'.join(lines)

    try:
        msg = EmailMessage()
        msg['Subject'] = f'[{severity}] Paperless AI Analyzer Bug Report — v{_APP_VERSION}'
        msg['From'] = smtp_cfg.get('from') or smtp_cfg.get('user', 'noreply@localhost')
        msg['To'] = dest
        if contact_email:
            msg['Reply-To'] = contact_email
        msg.set_content(body_text)

        # Attach HAR file if provided
        har_file = request.files.get('har_file')
        if har_file and har_file.filename:
            har_data = har_file.read()
            msg.add_attachment(
                har_data,
                maintype='application',
                subtype='json',
                filename=har_file.filename or 'browser.har',
            )

        _smtp_send(smtp_cfg, msg)
        return jsonify({'ok': True, 'message': f'Bug report sent to {dest}. Thank you!'})
    except Exception as e:
        logger.error(f'Bug report email failed: {e}')
        return jsonify({'error': f'Failed to send email: {e}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        # Check Paperless API
        healthy = app.paperless_client.health_check()

        if healthy:
            return jsonify({'status': 'healthy'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'paperless_api_unreachable'}), 503
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503


# ---------------------------------------------------------------------------
# System Health & Container Manager (v2.5.0)
# ---------------------------------------------------------------------------

PAPERLESS_CONTAINERS = [
    'paperless-web', 'paperless-consumer', 'paperless-redis', 'paperless-postgres',
]
HEALTH_TIMEOUT = 1.8  # seconds per component

_own_container_name_cache = None


def _get_own_container_name() -> str:
    """Return this container's Docker name. Cached after first successful lookup.

    Strategy:
    1. Docker sets the container hostname to the short container ID. Look it up
       via the Docker socket (requires the socket mount to be present).
    2. Fall back to URL_PREFIX env var (e.g. /paperless-ai-analyzer-dev →
       paperless-ai-analyzer-dev) — reliable because every instance has a
       unique prefix configured in docker-compose.yml.
    3. Last resort: hard-coded default name.
    """
    global _own_container_name_cache
    if _own_container_name_cache is not None:
        return _own_container_name_cache
    import socket as _socket
    hostname = _socket.gethostname()
    dc = _get_docker_client()
    if dc:
        try:
            container = dc.containers.get(hostname)
            _own_container_name_cache = container.name
            return _own_container_name_cache
        except Exception:
            pass
    # Fallback: derive from URL_PREFIX (strip leading slash)
    prefix = os.environ.get('URL_PREFIX', '').strip('/')
    _own_container_name_cache = prefix or 'paperless-ai-analyzer'
    return _own_container_name_cache


def _get_managed_containers() -> list:
    """Return the 4 core Paperless containers, per-project containers, plus this analyzer instance.

    Per-project containers are discovered dynamically from projects.db: any project whose
    paperless_url references 'paperless-web-{slug}' has a dedicated instance and gets both
    paperless-web-{slug} and paperless-consumer-{slug} added to the managed list.
    """
    managed = list(PAPERLESS_CONTAINERS)
    try:
        import sqlite3 as _sqlite3
        db_path = '/app/data/projects.db'
        with _sqlite3.connect(db_path) as _conn:
            rows = _conn.execute(
                "SELECT slug, paperless_url FROM projects WHERE paperless_url IS NOT NULL AND paperless_url != ''"
            ).fetchall()
            for (slug, url) in rows:
                if slug and f'paperless-web-{slug}' in (url or ''):
                    managed.append(f'paperless-web-{slug}')
                    managed.append(f'paperless-consumer-{slug}')
    except Exception:
        pass
    managed.append(_get_own_container_name())
    return managed


def _get_docker_client():
    """Return docker.DockerClient or None — never raises."""
    try:
        import docker
        return docker.from_env(timeout=3)
    except Exception:
        return None


def _check_paperless_api():
    """Check Paperless API health with timing."""
    import time
    start = time.monotonic()
    try:
        healthy = app.paperless_client.health_check()
        latency_ms = int((time.monotonic() - start) * 1000)
        if healthy:
            return {'status': 'ok', 'latency_ms': latency_ms, 'detail': f'Responded in {latency_ms}ms'}
        return {'status': 'error', 'latency_ms': latency_ms, 'detail': 'Health check returned false'}
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        return {'status': 'error', 'latency_ms': latency_ms, 'detail': str(e)[:120]}


def _check_chromadb():
    """Check ChromaDB by counting documents in the default vector store."""
    import time
    start = time.monotonic()
    try:
        da = getattr(app, 'document_analyzer', None)
        if not da or not getattr(da, 'vector_store', None) or not da.vector_store.enabled:
            return {'status': 'warning', 'latency_ms': 0, 'detail': 'Vector store not enabled'}
        count = da.vector_store.collection.count()
        latency_ms = int((time.monotonic() - start) * 1000)
        return {'status': 'ok', 'latency_ms': latency_ms, 'detail': f'{count} documents'}
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        return {'status': 'error', 'latency_ms': latency_ms, 'detail': str(e)[:120]}


def _check_llm():
    """Check LLM configuration — reads stored AI config to report all providers."""
    try:
        cfg = load_ai_config()
        global_keys = cfg.get('global', {})
        configured = [p for p, v in global_keys.items() if v.get('api_key', '').strip()]
    except Exception:
        configured = []

    # Fallback: also honour env-var key in case config file is missing
    env_provider = os.environ.get('LLM_PROVIDER', 'anthropic')
    env_key = os.environ.get('LLM_API_KEY', '').strip()
    if env_key and env_provider not in configured:
        configured.append(env_provider)

    if not configured:
        return {'status': 'error', 'latency_ms': 0, 'detail': 'No AI provider keys configured'}

    detail = ', '.join(f'{p} ✓' for p in configured)
    return {'status': 'ok', 'latency_ms': 0, 'detail': detail}


def _check_analyzer_loop():
    """Check whether the analyzer loop has run recently."""
    try:
        stats = app.state_manager.get_stats()
        last_run = stats.get('last_run')
        if not last_run:
            return {'status': 'warning', 'latency_ms': 0, 'detail': 'No runs recorded yet'}
        from datetime import datetime, timezone
        if isinstance(last_run, str):
            last_run = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_s = int((now - last_run).total_seconds())
        poll_interval = int(os.environ.get('POLL_INTERVAL_SECONDS', '30'))
        threshold = poll_interval * 10  # 10× gives headroom for long LLM processing cycles
        if age_s > threshold:
            return {'status': 'warning', 'latency_ms': 0, 'detail': f'Last run {age_s}s ago (threshold {threshold}s)'}
        return {'status': 'ok', 'latency_ms': 0, 'detail': f'Last run {age_s}s ago'}
    except Exception as e:
        return {'status': 'warning', 'latency_ms': 0, 'detail': f'Could not check: {str(e)[:80]}'}


def _check_docker_container(name, dc):
    """Check a Docker container's running status."""
    try:
        container = dc.containers.get(name)
        status = container.status
        if status == 'running':
            return {'status': 'ok', 'latency_ms': 0, 'detail': 'running'}
        return {'status': 'error', 'latency_ms': 0, 'detail': f'status: {status}'}
    except Exception as e:
        err = str(e)
        if 'Not Found' in err or '404' in err:
            return {'status': 'warning', 'latency_ms': 0, 'detail': 'not found'}
        return {'status': 'error', 'latency_ms': 0, 'detail': err[:80]}


def _check_project_containers(dc) -> dict:
    """Aggregate health check for all per-project Paperless containers.

    Returns ok if all are running, warning if any are missing/stopped,
    error if Docker is unavailable. Detail lists each project slug + status.
    """
    if not dc:
        return {'status': 'warning', 'latency_ms': 0, 'detail': 'Docker unavailable'}
    try:
        import sqlite3 as _sqlite3
        db_path = '/app/data/projects.db'
        with _sqlite3.connect(db_path) as _conn:
            rows = _conn.execute(
                "SELECT slug, paperless_url FROM projects WHERE paperless_url IS NOT NULL AND paperless_url != ''"
            ).fetchall()
        project_slugs = [slug for (slug, url) in rows if slug and f'paperless-web-{slug}' in (url or '')]
        if not project_slugs:
            return {'status': 'ok', 'latency_ms': 0, 'detail': 'No per-project instances provisioned'}
        parts = []
        worst = 'ok'
        for slug in project_slugs:
            web_name = f'paperless-web-{slug}'
            r = _check_docker_container(web_name, dc)
            parts.append(f'{slug}:{r["status"]}')
            if r['status'] == 'error' and worst != 'error':
                worst = 'error'
            elif r['status'] == 'warning' and worst == 'ok':
                worst = 'warning'
        return {'status': worst, 'latency_ms': 0, 'detail': ', '.join(parts)}
    except Exception as e:
        return {'status': 'warning', 'latency_ms': 0, 'detail': str(e)[:120]}


@app.route('/api/system-health')
@login_required
def api_system_health():
    """Parallel health check for all system components."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone
    dc = _get_docker_client()
    docker_available = dc is not None

    checks = {
        'paperless_api': _check_paperless_api,
        'chromadb': _check_chromadb,
        'llm': _check_llm,
        'analyzer_loop': _check_analyzer_loop,
        'postgres': (lambda: _check_docker_container('paperless-postgres', dc))
                    if docker_available else
                    (lambda: {'status': 'warning', 'latency_ms': 0, 'detail': 'Docker unavailable'}),
        'redis': (lambda: _check_docker_container('paperless-redis', dc))
                 if docker_available else
                 (lambda: {'status': 'warning', 'latency_ms': 0, 'detail': 'Docker unavailable'}),
        'projects_containers': (lambda: _check_project_containers(dc)),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {executor.submit(fn): name for name, fn in checks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result(timeout=HEALTH_TIMEOUT)
            except Exception as e:
                results[name] = {'status': 'error', 'latency_ms': 0, 'detail': f'Check failed: {str(e)[:60]}'}

    statuses = [r['status'] for r in results.values()]
    if 'error' in statuses:
        overall = 'error'
    elif 'warning' in statuses:
        overall = 'warning'
    else:
        overall = 'ok'

    return jsonify({
        'overall': overall,
        'checked_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'components': results,
    })


@app.route('/api/containers')
@login_required
def api_containers_list():
    """List managed containers with status info."""
    dc = _get_docker_client()
    if not dc:
        return jsonify({'available': False, 'containers': []})

    from datetime import datetime, timezone
    containers = []
    for name in _get_managed_containers():
        try:
            c = dc.containers.get(name)
            attrs = c.attrs
            started_at = attrs.get('State', {}).get('StartedAt', '')
            uptime_seconds = None
            if started_at and started_at != '0001-01-01T00:00:00Z':
                try:
                    started = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    uptime_seconds = int((datetime.now(timezone.utc) - started).total_seconds())
                except Exception:
                    pass
            health_status = None
            health = attrs.get('State', {}).get('Health')
            if health:
                health_status = health.get('Status')
            image_tag = ''
            image_tags = c.image.tags
            if image_tags:
                image_tag = image_tags[0]
            containers.append({
                'name': name,
                'status': c.status,
                'image': image_tag,
                'uptime_seconds': uptime_seconds,
                'health_status': health_status,
            })
        except Exception as e:
            err = str(e)
            if 'Not Found' in err or '404' in err:
                containers.append({'name': name, 'status': 'not_found', 'image': '',
                                   'uptime_seconds': None, 'health_status': None})
            else:
                containers.append({'name': name, 'status': 'error', 'image': '',
                                   'uptime_seconds': None, 'health_status': None})

    return jsonify({'available': True, 'containers': containers})


@app.route('/api/containers/<name>/restart', methods=['POST'])
@login_required
@admin_required
def api_container_restart(name):
    """Restart a managed container (admin only)."""
    if name not in _get_managed_containers():
        return jsonify({'error': 'Container not in managed list'}), 403
    dc = _get_docker_client()
    if not dc:
        return jsonify({'error': 'Docker unavailable'}), 503
    try:
        container = dc.containers.get(name)
        container.restart(timeout=15)
        logger.warning(f'Admin {current_user.username} restarted container {name}')
        return jsonify({'ok': True, 'message': f'Container {name} restarted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/containers/<name>/logs')
@login_required
@admin_required
def api_container_logs(name):
    """Get recent logs from a managed container (admin only)."""
    if name not in _get_managed_containers():
        return jsonify({'error': 'Container not in managed list'}), 403
    dc = _get_docker_client()
    if not dc:
        return jsonify({'error': 'Docker unavailable'}), 503
    lines = min(int(request.args.get('lines', 100)), 500)
    try:
        container = dc.containers.get(name)
        raw = container.logs(tail=lines, timestamps=True)
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
        log_lines = [ln for ln in raw.splitlines() if ln]
        return jsonify({'name': name, 'lines': log_lines})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Chat Session Routes
# ---------------------------------------------------------------------------

@app.route('/api/chat/sessions', methods=['GET'])
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
            sessions = [{
                'id': r['id'],
                'title': r['title'],
                'document_type': r['document_type'],
                'created_at': r['created_at'],
                'updated_at': r['updated_at'],
                'owner_username': r['owner_username'],
                'is_shared': bool(r['is_shared']),
                'is_own': r['user_id'] == current_user.id,
            } for r in rows]
            return jsonify({'sessions': sessions, 'is_admin': False})
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions', methods=['POST'])
@login_required
def api_chat_sessions_create():
    """Create a new chat session."""
    try:
        data = request.json or {}
        title = data.get('title', 'New Chat')
        document_type = data.get('document_type')
        project_slug = session.get('current_project', 'default')
        session_id = create_session(current_user.id, title=title, document_type=document_type,
                                    project_slug=project_slug)
        return jsonify({'session_id': session_id, 'title': title})
    except Exception as e:
        logger.error(f"Create session error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>', methods=['GET'])
@login_required
def api_chat_session_get(session_id):
    """Get session details + messages."""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        # Check access
        if not current_user.is_admin and not can_access_session(session_id, current_user.id):
            return jsonify({'error': 'Access denied'}), 403
        msgs = get_messages(session_id)
        shares = get_session_shares(session_id)
        return jsonify({
            'session': {
                'id': session['id'],
                'title': session['title'],
                'document_type': session['document_type'],
                'created_at': session['created_at'],
                'updated_at': session['updated_at'],
                'user_id': session['user_id'],
                'is_owner': session['user_id'] == current_user.id,
            },
            'messages': [{'id': m['id'], 'role': m['role'], 'content': m['content'], 'created_at': m['created_at']} for m in msgs],
            'shared_with': [{'id': s['id'], 'username': s['username']} for s in shares],
        })
    except Exception as e:
        logger.error(f"Get session error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
@login_required
def api_chat_session_delete(session_id):
    """Delete a session (owner or admin only)."""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and session['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        delete_session(session_id)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>', methods=['PATCH'])
@login_required
def api_chat_session_rename(session_id):
    """Rename a session title."""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and session['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        data = request.json or {}
        title = data.get('title', '').strip()
        if not title:
            return jsonify({'error': 'Title required'}), 400
        update_session_title(session_id, title)
        return jsonify({'success': True, 'title': title})
    except Exception as e:
        logger.error(f"Rename session error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/messages/<int:message_id>/edit', methods=['PATCH'])
@login_required
def api_chat_message_edit(session_id, message_id):
    """Edit a user message and delete subsequent messages so the conversation can be resent."""
    try:
        sess = get_session(session_id)
        if not sess:
            return jsonify({'error': 'Session not found'}), 404
        if not can_access_session(session_id, current_user.id):
            return jsonify({'error': 'Access denied'}), 403
        data = request.json or {}
        new_content = data.get('content', '').strip()
        if not new_content:
            return jsonify({'error': 'Content required'}), 400
        update_message_content(message_id, session_id, new_content)
        delete_messages_from(session_id, message_id + 1)
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Edit message error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/share', methods=['POST'])
@login_required
def api_chat_session_share(session_id):
    """Share session with a user by username."""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and session['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        data = request.json or {}
        target_username = data.get('username', '').strip()
        if not target_username:
            return jsonify({'error': 'Username required'}), 400
        target_user = get_user_by_username(target_username)
        if not target_user:
            return jsonify({'error': f"User '{target_username}' not found"}), 404
        if target_user['id'] == session['user_id']:
            return jsonify({'error': 'Cannot share with the owner'}), 400
        share_session(session_id, target_user['id'], current_user.id)
        return jsonify({'success': True, 'shared_with': target_username})
    except Exception as e:
        logger.error(f"Share session error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/share/<int:uid>', methods=['DELETE'])
@login_required
def api_chat_session_unshare(session_id, uid):
    """Remove share from a user."""
    try:
        session = get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        if not current_user.is_admin and session['user_id'] != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
        unshare_session(session_id, uid)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Unshare session error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/export', methods=['GET'])
@login_required
def api_chat_session_export(session_id):
    """Export chat session as PDF."""
    try:
        import mistune
        import weasyprint

        session = get_session(session_id)
        if not session:
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
            session_title=session['title'],
            username=current_user.display_name,
            export_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
            document_type=session['document_type'],
            messages=messages_with_html,
        )

        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        safe_title = ''.join(c for c in session['title'] if c.isalnum() or c in ' -_')[:40]
        response.headers['Content-Disposition'] = (
            f'attachment; filename="chat-{session_id[:8]}-{safe_title}.pdf"'
        )
        return response
    except Exception as e:
        logger.error(f"Export session error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# User Management Routes (admin only)
# ---------------------------------------------------------------------------

@app.route('/api/users', methods=['GET'])
@login_required
@admin_required
def api_users_list():
    """List all users (admin only)."""
    try:
        rows = list_users()
        users = [{
            'id': r['id'],
            'username': r['username'],
            'display_name': r['display_name'],
            'email': r['email'] or '',
            'role': r['role'],
            'created_at': r['created_at'],
            'last_login': r['last_login'],
            'is_active': bool(r['is_active']),
            'phone': r['phone'] or '',
            'address': r['address'] or '',
            'github': r['github'] or '',
            'linkedin': r['linkedin'] or '',
            'facebook': r['facebook'] or '',
            'instagram': r['instagram'] or '',
            'other_handles': r['other_handles'] or '',
            'timezone': r['timezone'] or '',
            'job_title': r['job_title'] or '',
        } for r in rows]
        return jsonify({'users': users})
    except Exception as e:
        logger.error(f"List users error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users', methods=['POST'])
@login_required
@admin_required
def api_users_create():
    """Create a new user (admin only), then send a welcome email if SMTP is configured."""
    try:
        data = request.json or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'basic')
        display_name = data.get('display_name', '').strip() or username
        email = data.get('email', '').strip()
        job_title = data.get('job_title', '').strip()
        if not username or not password:
            return jsonify({'error': 'username and password required'}), 400
        if role not in ('basic', 'advanced', 'admin'):
            return jsonify({'error': 'role must be basic, advanced, or admin'}), 400
        if get_user_by_username(username):
            return jsonify({'error': f"User '{username}' already exists"}), 409
        db_create_user(username, password, role=role, display_name=display_name, email=email)

        # Send welcome notification if an email was supplied
        if email:
            app_url = request.host_url.rstrip('/') + request.script_root
            _send_welcome_email(email, display_name, username, role, app_url, job_title=job_title)

        return jsonify({'success': True, 'username': username, 'email_sent': bool(email)}), 201
    except Exception as e:
        logger.error(f"Create user error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:uid>', methods=['PATCH'])
@login_required
@admin_required
def api_users_update(uid):
    """Update role / display_name / email / password / is_active (admin only)."""
    try:
        data = request.json or {}
        allowed = {k: v for k, v in data.items() if k in (
            'role', 'display_name', 'email', 'password', 'is_active',
            'phone', 'address', 'github', 'linkedin', 'facebook',
            'instagram', 'other_handles', 'timezone', 'job_title',
        )}
        if 'role' in allowed and allowed['role'] not in ('basic', 'advanced', 'admin'):
            return jsonify({'error': 'role must be basic, advanced, or admin'}), 400
        db_update_user(uid, **allowed)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Update user error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:uid>', methods=['DELETE'])
@login_required
@admin_required
def api_users_deactivate(uid):
    """Soft-delete a user by setting is_active=0 (admin only)."""
    try:
        if uid == current_user.id:
            return jsonify({'error': 'Cannot deactivate yourself'}), 400
        db_update_user(uid, is_active=0)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:uid>/send-manual', methods=['POST'])
@login_required
@admin_required
def api_users_send_manual(uid):
    """Send the user manual link to a user via email (admin only)."""
    try:
        row = get_user_by_id(uid)
        if not row:
            return jsonify({'error': 'User not found'}), 404
        user = dict(row)
        if not user.get('email'):
            return jsonify({'error': 'User has no email address configured'}), 400
        base_url = request.host_url.rstrip('/') + request.script_root
        _send_manual_email(user['email'], user.get('display_name') or user['username'], base_url)
        return jsonify({'success': True, 'message': f"Manual sent to {user['email']}"})
    except Exception as e:
        logger.error(f"Send manual error for uid {uid}: {e}")
        return jsonify({'error': str(e)}), 500


def update_ui_stats(analysis_result: Dict[str, Any]) -> None:
    """
    Update UI statistics with analysis result.

    Args:
        analysis_result: Result from document analysis
    """
    with ui_state['lock']:
        # Add to recent analyses
        ui_state['recent_analyses'].append(analysis_result)

        # Keep only last 100
        if len(ui_state['recent_analyses']) > 100:
            ui_state['recent_analyses'] = ui_state['recent_analyses'][-100:]

        # Update stats
        ui_state['stats']['total_analyzed'] += 1

        # Persist to app.db so the count survives container restarts
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
    """Get analyzer uptime in seconds."""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
            return int(uptime_seconds)
    except:
        return 0


# =============================================================================
# CASE INTELLIGENCE AI ROUTES — v3.0.0
# All routes check _ci_gate() before executing.
# =============================================================================

@app.route('/api/ci/status')
@login_required
def ci_status():
    """Feature status + authority corpus stats."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_authority_corpus_stats
        from analyzer.case_intelligence.job_manager import get_job_manager
        corpus_stats = get_authority_corpus_stats()
        active_runs = get_job_manager().list_active_runs()
        return jsonify({
            'enabled': True,
            'authority_corpus': corpus_stats,
            'active_runs': len(active_runs),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/jurisdictions')
@login_required
def ci_jurisdictions():
    """List pre-built jurisdiction profiles."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.jurisdiction import list_jurisdiction_profiles
        return jsonify({'jurisdictions': list_jurisdiction_profiles()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _match_jurisdiction_profile(extracted: dict) -> str:
    """Map LLM-extracted jurisdiction details to a JURISDICTION_PROFILES key."""
    court = (extracted.get('court_name') or '').lower()
    is_bk = extracted.get('is_bankruptcy', False) or 'bankruptcy' in court
    bk_ch = extracted.get('bankruptcy_chapter')
    is_fam = extracted.get('is_family_court', False) or 'family' in court
    is_sur = extracted.get('is_surrogate', False) or 'surrogate' in court
    is_fed = extracted.get('is_federal', False)
    is_commercial = 'commercial' in court

    if is_bk:
        if 'eastern' in court or 'edny' in court or 'e.d.n.y' in court:
            return 'edny-bankruptcy'
        ch = int(bk_ch) if bk_ch and str(bk_ch).isdigit() else 7
        return f'sdny-bankruptcy-ch{ch}' if ch in (7, 11) else 'sdny-bankruptcy-ch7'
    if is_fam:
        return 'nys-family-court'
    if is_sur:
        return 'nys-surrogate'
    if 'appellate' in court:
        return 'nys-appellate-div-1' if ('first' in court or '1st' in court) else 'nys-appellate-div-2'
    if is_fed or 'district' in court or 'sdny' in court or 's.d.n.y' in court:
        if 'eastern' in court or 'edny' in court or 'e.d.n.y' in court:
            return 'edny-civil'
        return 'sdny-civil'
    if is_commercial:
        return 'nys-commercial-division'
    # Default: NYS Supreme Court (most common for NY cases)
    state = (extracted.get('state') or 'NY').upper()
    if state == 'NY' or 'supreme' in court:
        return 'nys-supreme-civil'
    return 'custom'


@app.route('/api/ci/detect-jurisdiction', methods=['POST'])
@login_required
def ci_detect_jurisdiction():
    """Detect court jurisdiction from project documents using LLM."""
    ok, err = _ci_gate()
    if not ok:
        return err
    import json as _json
    try:
        project_slug = session.get('current_project', 'default')
        from analyzer.vector_store import VectorStore
        from analyzer.case_intelligence.jurisdiction import JURISDICTION_PROFILES

        vs = VectorStore(project_slug=project_slug)
        if not vs.enabled:
            return jsonify({'detected': False, 'reason': 'Vector store not enabled for this project'})

        # Search for jurisdiction-relevant document excerpts
        results = vs.search(
            "court supreme district county filed plaintiff defendant address jurisdiction caption index number",
            n_results=8
        )
        if not results:
            return jsonify({'detected': False, 'reason': 'No documents found in project'})

        # Build document excerpts (first 500 chars each)
        excerpts = []
        for r in results[:6]:
            title = r.get('title', 'Untitled')
            content = (r.get('content') or r.get('document') or '')[:500].strip()
            if content:
                excerpts.append(f"[{title}]\n{content}")
        if not excerpts:
            return jsonify({'detected': False, 'reason': 'Documents found but content unavailable'})
        excerpts_text = "\n\n---\n\n".join(excerpts)

        # Use the app's existing LLM client
        llm = getattr(app, 'llm_client', None)
        if not llm or not llm.client:
            return jsonify({'detected': False, 'reason': 'LLM client not available'})

        prompt = (
            "You are analyzing legal documents to identify their court jurisdiction.\n\n"
            "Examine these document excerpts and identify:\n"
            "- The specific court (e.g. 'NYS Supreme Court', 'S.D.N.Y.', 'E.D.N.Y. Bankruptcy Court')\n"
            "- The county (for NY state cases, e.g. 'Kings', 'New York', 'Queens')\n"
            "- The state (two-letter code, e.g. 'NY')\n"
            "- Whether it is a federal case (SDNY, EDNY, etc.)\n"
            "- Whether it is a bankruptcy case, and if so, chapter number (7 or 11)\n"
            "- Whether it is a Family Court or Surrogate's Court case\n"
            "- Addresses of parties (to confirm location)\n\n"
            "Document excerpts:\n"
            f"{excerpts_text[:3500]}\n\n"
            "Respond with ONLY valid JSON, no other text:\n"
            '{"court_name": "string or null", "county": "string or null", '
            '"state": "NY", "is_federal": false, "is_bankruptcy": false, '
            '"bankruptcy_chapter": null, "is_family_court": false, '
            '"is_surrogate": false, "confidence": 0.85, '
            '"reasoning": "one sentence explanation"}'
        )

        if llm.provider == 'openai':
            resp = llm.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_tokens=350,
                response_format={'type': 'json_object'},
            )
            raw = resp.choices[0].message.content
        else:
            resp = llm.client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=350,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = resp.content[0].text

        if not raw or not raw.strip():
            return jsonify({'detected': False, 'reason': 'LLM returned an empty response'})

        # Strip markdown code fences if the model wrapped the JSON
        raw_stripped = raw.strip()
        if raw_stripped.startswith('```'):
            raw_stripped = raw_stripped.split('\n', 1)[-1]
            raw_stripped = raw_stripped.rsplit('```', 1)[0].strip()

        extracted = _json.loads(raw_stripped)
        profile_id = _match_jurisdiction_profile(extracted)
        profile = JURISDICTION_PROFILES.get(profile_id)

        return jsonify({
            'detected': True,
            'jurisdiction_id': profile_id,
            'display_name': profile.display_name if profile else 'Custom',
            'court': profile.court if profile else (extracted.get('court_name') or ''),
            'county': extracted.get('county'),
            'confidence': extracted.get('confidence', 0.5),
            'reason': extracted.get('reasoning', ''),
        })

    except Exception as e:
        logger.error(f"CI detect-jurisdiction error: {e}", exc_info=True)
        return jsonify({'detected': False, 'reason': f'Detection failed: {str(e)}'})


@app.route('/api/ci/runs', methods=['GET'])
@login_required
def ci_list_runs():
    """List CI runs for current project.
    Admins see all runs; others see own runs + runs shared with them.
    """
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import (
            list_ci_runs, get_run_ids_shared_with, get_ci_run,
        )
        project_slug = session.get('current_project', 'default')

        if current_user.is_admin:
            runs = [dict(r) for r in list_ci_runs(project_slug)]
        else:
            own = [dict(r) for r in list_ci_runs(project_slug, user_id=current_user.id)]
            shared_ids = get_run_ids_shared_with(current_user.id)
            shared = []
            for rid in shared_ids:
                r = get_ci_run(rid)
                if r and r['project_slug'] == project_slug:
                    d = dict(r)
                    d['_shared'] = True
                    shared.append(d)
            own_ids = {r['id'] for r in own}
            runs = own + [r for r in shared if r['id'] not in own_ids]
            runs.sort(key=lambda r: r.get('created_at', ''), reverse=True)

        # Annotate with owner display_name
        for r in runs:
            u = get_user_by_id(r['user_id'])
            r['owner_name'] = u['display_name'] if u else f"user#{r['user_id']}"

        return jsonify({'runs': runs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/goal-assistant', methods=['POST'])
@login_required
@advanced_required
def ci_goal_assistant():
    """Lightweight AI chat that helps the user write a focused CI goal statement.
    Stateless — caller maintains conversation history in the browser.
    """
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        data = request.json or {}
        messages = data.get('messages', [])          # [{role, content}, ...]
        ctx = data.get('context', {})
        role = ctx.get('role', 'neutral')
        jurisdiction = ctx.get('jurisdiction', 'Not specified')
        draft_goal = ctx.get('draft_goal', '').strip()

        system_prompt = (
            "You are a legal case strategy advisor helping an attorney write a clear, focused "
            "goal statement for a document-analysis AI called Case Intelligence.\n\n"
            "Case Intelligence will analyze legal documents to:\n"
            "- Extract people, organizations, accounts, and key properties\n"
            "- Build a chronological timeline of events\n"
            "- Detect financial flows and amounts\n"
            "- Find contradictions between documents\n"
            "- Generate factual and legal theories\n"
            "- Identify relevant legal authorities\n\n"
            "A great goal statement tells Case Intelligence EXACTLY what to find:\n"
            "  • Who the client represents and the core legal dispute\n"
            "  • What specific evidence or patterns to surface\n"
            "  • What outcome the attorney needs (support damages, build defense, find smoking-gun)\n\n"
            f"CURRENT SETUP:\n"
            f"  Role: {role}\n"
            f"  Jurisdiction: {jurisdiction}\n"
        )
        if draft_goal:
            system_prompt += f"  Draft goal: \"{draft_goal}\"\n"
        system_prompt += (
            "\nINSTRUCTIONS:\n"
            "1. On the first turn, briefly acknowledge the context and ask 2-3 targeted "
            "questions to understand the case better. Be concise.\n"
            "2. After you have enough information, produce a polished goal statement.\n"
            "3. When you are ready to suggest a goal, end your message with exactly:\n"
            "📋 Suggested Goal:\n[goal text — 2-4 sentences, specific and actionable]\n"
            "4. Only produce ONE suggested goal block per message. Keep the rest of your "
            "message brief.\n"
            "5. If the user wants changes, revise the suggested goal in the same format."
        )

        project_slug = session.get('current_project', 'default')
        chat_cfg = get_project_ai_config(project_slug, 'chat')
        _full_cfg = load_ai_config()

        def _global_key(p):
            return _full_cfg.get('global', {}).get(p, {}).get('api_key', '').strip()

        provider = chat_cfg.get('provider', 'openai')
        api_key = (chat_cfg.get('api_key') or '').strip() or _global_key(provider)
        model = chat_cfg.get('model', 'gpt-4o')

        # Fallback if primary has no key
        if not api_key:
            fb_prov = chat_cfg.get('fallback_provider')
            api_key = _global_key(fb_prov) if fb_prov else ''
            if api_key:
                provider = fb_prov
                model = chat_cfg.get('fallback_model', model)

        if not api_key:
            return jsonify({'error': 'No AI API key configured.'}), 503

        if provider == 'openai':
            import openai as _oai
            client = _oai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=600,
                messages=[{'role': 'system', 'content': system_prompt}] + messages,
            )
            reply = resp.choices[0].message.content
        elif provider == 'anthropic':
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            # Anthropic requires at least one message; on the first greeting call
            # the browser sends an empty list, so inject a synthetic opener.
            ant_messages = messages if messages else [
                {'role': 'user', 'content': 'Hello, I need help writing a focused goal statement for my Case Intelligence analysis.'}
            ]
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=system_prompt,
                messages=ant_messages,
            )
            reply = resp.content[0].text
        else:
            return jsonify({'error': f'Unsupported provider: {provider}'}), 400

        # Extract suggested goal if present
        suggested_goal = None
        marker = '📋 Suggested Goal:'
        if marker in reply:
            suggested_goal = reply.split(marker, 1)[1].strip()

        return jsonify({'response': reply, 'suggested_goal': suggested_goal})
    except Exception as e:
        logger.error(f"CI goal assistant error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs', methods=['POST'])
@login_required
@advanced_required
def ci_create_run():
    """Create a new CI run (draft status)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import json as _json
        from analyzer.case_intelligence.db import create_ci_run
        data = request.json or {}
        project_slug = session.get('current_project', 'default')

        jurisdiction_json = '{}'
        if 'jurisdiction' in data:
            jurisdiction_json = _json.dumps(data['jurisdiction'])

        notification_email = data.get('notification_email', '') or ''
        notify_on_complete = 1 if data.get('notify_on_complete', True) else 0
        notify_on_budget   = 1 if data.get('notify_on_budget',   True) else 0

        run_id = create_ci_run(
            project_slug=project_slug,
            user_id=current_user.id,
            role=data.get('role', 'neutral'),
            goal_text=data.get('goal_text', ''),
            budget_per_run_usd=float(data.get('budget_per_run_usd', 10.0)),
            jurisdiction_json=jurisdiction_json,
            objectives=_json.dumps(data.get('objectives', [])),
            max_tier=int(data.get('max_tier', 3)),
            notification_email=notification_email,
            notify_on_complete=notify_on_complete,
            notify_on_budget=notify_on_budget,
        )
        # Store web research config if provided
        if 'web_research_config' in data:
            from analyzer.case_intelligence.db import update_ci_run as _ucr
            wrc = data['web_research_config']
            if isinstance(wrc, dict):
                _ucr(run_id, web_research_config=_json.dumps(wrc))
        return jsonify({'run_id': run_id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>', methods=['GET'])
@login_required
def ci_get_run(run_id):
    """Get run config + status."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Not authorized'}), 403
        return jsonify(dict(run))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>', methods=['PUT'])
@login_required
@advanced_required
def ci_update_run(run_id):
    """Update run config (draft only)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import json as _json
        from analyzer.case_intelligence.db import get_ci_run, update_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if run['status'] not in ('draft',):
            return jsonify({'error': 'Can only edit draft runs'}), 400
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        data = request.json or {}
        allowed_fields = {
            'role', 'goal_text', 'budget_per_run_usd', 'max_tier', 'auto_routing',
        }
        kwargs = {k: v for k, v in data.items() if k in allowed_fields}
        if 'jurisdiction' in data:
            kwargs['jurisdiction_json'] = _json.dumps(data['jurisdiction'])
        if 'objectives' in data:
            kwargs['objectives'] = _json.dumps(data['objectives'])
        if 'web_research_config' in data:
            wrc = data['web_research_config']
            kwargs['web_research_config'] = _json.dumps(wrc) if isinstance(wrc, dict) else wrc
        update_ci_run(run_id, **kwargs)
        return jsonify({'status': 'updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/start', methods=['POST'])
@login_required
@advanced_required
def ci_start_run(run_id):
    """Launch a CI run as a background job."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import os as _os
        from analyzer.case_intelligence.db import get_ci_run, update_ci_run
        from analyzer.case_intelligence.orchestrator import CIOrchestrator
        from analyzer.case_intelligence.job_manager import get_job_manager
        from analyzer.case_intelligence.db import init_ci_db

        # Ensure DB is initialized
        init_ci_db()

        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        if run['status'] not in ('draft', 'failed', 'cancelled'):
            return jsonify({'error': f"Cannot start run in status '{run['status']}'"}), 400

        # Reset run for fresh start
        update_ci_run(run_id, status='queued', progress_pct=0,
                      cost_so_far_usd=0, budget_blocked=0,
                      budget_blocked_note=None, error_message=None,
                      started_at=datetime.utcnow().isoformat())

        # Build LLM clients from app config
        llm_clients = _build_ci_llm_clients()

        orchestrator = CIOrchestrator(
            llm_clients=llm_clients,
            paperless_client=getattr(app, 'paperless_client', None),
            usage_tracker=getattr(app, 'usage_tracker', None),
            cohere_api_key=_os.environ.get('COHERE_API_KEY'),
            budget_notification_cb=_send_ci_budget_notification,
            completion_notification_cb=_send_ci_complete_notification,
        )

        job_manager = get_job_manager()
        started = job_manager.start_run(run_id, orchestrator.execute_run, run_id)

        if not started:
            return jsonify({'error': 'Run is already active'}), 409

        return jsonify({'status': 'started', 'run_id': run_id})
    except Exception as e:
        logger.error(f"CI start_run failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/cancel', methods=['POST'])
@login_required
@advanced_required
def ci_cancel_run(run_id):
    """Send cancellation signal to a running CI job."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run
        from analyzer.case_intelligence.job_manager import get_job_manager

        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        sent = get_job_manager().cancel_run(run_id)
        return jsonify({'cancelled': sent, 'run_id': run_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _ci_elapsed_seconds(run):
    """Return seconds since run started_at, or 0."""
    if not run.get('started_at'):
        return 0
    try:
        from datetime import datetime, timezone
        start = datetime.fromisoformat(run['started_at'].replace('Z', '+00:00'))
        return int((datetime.now(timezone.utc) - start).total_seconds())
    except Exception:
        return 0


@app.route('/api/ci/runs/<run_id>/status')
@login_required
def ci_run_status(run_id):
    """Live progress polling endpoint (every 3s from UI)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        return jsonify({
            'run_id': run_id,
            'status': run['status'],
            'current_stage': run['current_stage'],
            'progress_pct': run['progress_pct'],
            'cost_so_far_usd': run['cost_so_far_usd'],
            'budget_per_run_usd': run['budget_per_run_usd'],
            'docs_processed': run['docs_processed'],
            'docs_total': run['docs_total'],
            'error_message': run['error_message'],
            'budget_blocked': bool(run['budget_blocked']),
            'budget_blocked_note': run['budget_blocked_note'],
            'tokens_in':       run.get('tokens_in', 0) or 0,
            'tokens_out':      run.get('tokens_out', 0) or 0,
            'active_managers': run.get('active_managers', 0) or 0,
            'active_workers':  run.get('active_workers', 0) or 0,
            'elapsed_seconds': _ci_elapsed_seconds(run),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/findings')
@login_required
def ci_run_findings(run_id):
    """Full findings: entities, timeline, contradictions, theories, authorities."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import (
            get_ci_run, get_ci_entities, get_ci_timeline, get_ci_contradictions,
            get_ci_theories, get_ci_authorities, get_ci_disputed_facts,
        )
        import json as _json
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Not authorized'}), 403

        findings_summary = None
        if run['findings_summary']:
            try:
                findings_summary = _json.loads(run['findings_summary'])
            except Exception:
                findings_summary = {'raw': run['findings_summary']}

        # ── Run metadata ──────────────────────────────────────────────────────
        # Compute duration
        duration_str = None
        try:
            if run.get('started_at') and run.get('completed_at'):
                from datetime import datetime as _dt
                fmt = '%Y-%m-%dT%H:%M:%S'
                t0 = _dt.fromisoformat(run['started_at'].split('.')[0].replace('Z', ''))
                t1 = _dt.fromisoformat(run['completed_at'].split('.')[0].replace('Z', ''))
                secs = int((t1 - t0).total_seconds())
                if secs >= 3600:
                    duration_str = f"{secs//3600}h {(secs%3600)//60}m"
                elif secs >= 60:
                    duration_str = f"{secs//60}m {secs%60}s"
                else:
                    duration_str = f"{secs}s"
        except Exception:
            pass

        # Look up user display name
        run_user = None
        try:
            import sqlite3 as _sq3
            with _sq3.connect('/app/data/app.db') as _uc:
                _uc.row_factory = _sq3.Row
                _ur = _uc.execute(
                    'SELECT display_name, username FROM users WHERE id=?',
                    (run.get('user_id'),)
                ).fetchone()
                if _ur:
                    run_user = _ur['display_name'] or _ur['username']
        except Exception:
            pass

        run_meta = {
            'created_at':      run.get('created_at'),
            'completed_at':    run.get('completed_at'),
            'duration':        duration_str,
            'run_by':          run_user,
            'role':            run.get('role'),
            'goal_text':       run.get('goal_text'),
            'project_slug':    run.get('project_slug'),
            'docs_total':      run.get('docs_total'),
            'docs_processed':  run.get('docs_processed'),
            'cost_usd':        run.get('cost_so_far_usd'),
            'budget_usd':      run.get('budget_per_run_usd'),
            'status':          run.get('status'),
        }

        # ── Build doc_map from theory evidence fields ─────────────────────────
        theories_raw = [dict(t) for t in get_ci_theories(run_id)]
        doc_ids_needed = set()
        for t in theories_raw:
            for field in ('supporting_evidence', 'counter_evidence'):
                val = t.get(field)
                if not val:
                    continue
                try:
                    items = _json.loads(val) if isinstance(val, str) else val
                    for item in (items or []):
                        did = item.get('paperless_doc_id')
                        if did:
                            doc_ids_needed.add(int(did))
                except Exception:
                    pass

        doc_map = {}
        if doc_ids_needed and hasattr(app, 'paperless_client'):
            for did in doc_ids_needed:
                try:
                    doc = app.paperless_client.get_document(did)
                    if doc:
                        doc_map[did] = {
                            'id':      did,
                            'title':   doc.get('title', f'Document {did}'),
                            'summary': (doc.get('content') or '')[:300].strip(),
                        }
                except Exception:
                    doc_map[did] = {'id': did, 'title': f'Document {did}', 'summary': ''}

        # Fetch web research results
        from analyzer.case_intelligence.db import get_ci_web_research as _gcwr
        web_research_raw = _gcwr(run_id)

        return jsonify({
            'run_id':           run_id,
            'status':           run['status'],
            'run_meta':         run_meta,
            'doc_map':          doc_map,
            'findings_summary': findings_summary,
            'entities':         [dict(e) for e in get_ci_entities(run_id)],
            'timeline':         [dict(ev) for ev in get_ci_timeline(run_id)],
            'contradictions':   [dict(c) for c in get_ci_contradictions(run_id)],
            'disputed_facts':   [dict(f) for f in get_ci_disputed_facts(run_id)],
            'theories':         theories_raw,
            'authorities':      [dict(a) for a in get_ci_authorities(run_id)],
            'web_research':     web_research_raw,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>', methods=['DELETE'])
@login_required
@advanced_required
def ci_delete_run(run_id):
    """Delete a CI run and all its associated findings."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import sqlite3 as _sq3
        from analyzer.case_intelligence.db import get_ci_run as _get_run
        run = _get_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        with _sq3.connect('/app/data/case_intelligence.db') as conn:
            row = conn.execute('SELECT id, status FROM ci_runs WHERE id=?', (run_id,)).fetchone()
            if not row:
                return jsonify({'error': 'Run not found'}), 404
            if row[1] == 'running':
                return jsonify({'error': 'Cannot delete a run that is currently running. Cancel it first.'}), 409
            # Cascade delete (FK ON DELETE CASCADE covers child tables if enabled,
            # but delete explicitly to be safe)
            for tbl in ('ci_entities', 'ci_timeline_events', 'ci_contradictions',
                        'ci_disputed_facts', 'ci_theory_ledger', 'ci_authorities',
                        'ci_reports', 'ci_manager_reports', 'ci_run_questions'):
                conn.execute(f'DELETE FROM {tbl} WHERE run_id=?', (run_id,))
            conn.execute('DELETE FROM ci_runs WHERE id=?', (run_id,))
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/shares', methods=['GET'])
@login_required
@advanced_required
def ci_get_run_shares(run_id):
    """List users this run is shared with. Owner or admin only."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, list_ci_run_shares
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        shares = list_ci_run_shares(run_id)
        result = []
        for s in shares:
            u = get_user_by_id(s['shared_with'])
            result.append({
                'user_id': s['shared_with'],
                'display_name': u['display_name'] if u else f"user#{s['shared_with']}",
                'username': u['username'] if u else '',
                'shared_by': s['shared_by'],
                'shared_at': s['shared_at'],
            })
        return jsonify({'shares': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/shares', methods=['POST'])
@login_required
@advanced_required
def ci_add_run_share(run_id):
    """Share a run with another user by username or user_id. Owner or admin only."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, add_ci_run_share
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        data = request.json or {}
        target = None
        if 'username' in data:
            target = get_user_by_username(data['username'])
        elif 'user_id' in data:
            target = get_user_by_id(int(data['user_id']))
        if not target:
            return jsonify({'error': 'User not found'}), 404
        if not target['is_active']:
            return jsonify({'error': 'Cannot share with an inactive user'}), 400
        if target['role'] not in ('advanced', 'admin'):
            return jsonify({'error': 'Target user must have Advanced or Admin role to access CI runs'}), 400
        if target['id'] == run['user_id']:
            return jsonify({'error': 'Run already belongs to this user'}), 400

        add_ci_run_share(run_id, shared_with=target['id'], shared_by=current_user.id)
        return jsonify({'ok': True, 'shared_with': target['display_name']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/shares/<int:uid>', methods=['DELETE'])
@login_required
@advanced_required
def ci_remove_run_share(run_id, uid):
    """Remove a user's access to a shared run. Owner or admin only."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, remove_ci_run_share
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        remove_ci_run_share(run_id, uid)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/questions', methods=['GET'])
@login_required
def ci_get_questions(run_id):
    """Get clarifying questions for a run."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_questions
        questions = get_ci_questions(run_id)
        return jsonify({'questions': [dict(q) for q in questions]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/answers', methods=['POST'])
@login_required
@advanced_required
def ci_submit_answers(run_id):
    """Submit answers to clarifying questions."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, answer_ci_question, update_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        data = request.json or {}
        answers = data.get('answers', {})  # {question_id: answer_text}
        proceed_with_assumptions = data.get('proceed_with_assumptions', False)

        for qid_str, answer_text in answers.items():
            try:
                answer_ci_question(int(qid_str), answer_text)
            except Exception as e:
                logger.warning(f"CI answer question {qid_str} failed: {e}")

        if proceed_with_assumptions:
            update_ci_run(run_id, proceed_with_assumptions=1)

        return jsonify({'status': 'answers_saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/reports', methods=['POST'])
@login_required
@advanced_required
def ci_create_report(run_id):
    """Generate a report for a CI run."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, create_ci_report
        from analyzer.case_intelligence.report_generator import ReportGenerator
        import threading as _threading

        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if run['status'] != 'completed':
            return jsonify({'error': 'Run must be completed before generating reports'}), 400

        data = request.json or {}
        instructions = data.get('instructions', 'Generate a comprehensive case summary.')
        template = data.get('template', 'custom')

        report_id = create_ci_report(
            run_id=run_id,
            user_id=current_user.id,
            instructions=instructions,
            template=template,
        )

        # Generate in background thread
        llm_clients = _build_ci_llm_clients()
        generator = ReportGenerator(
            llm_clients=llm_clients,
            usage_tracker=getattr(app, 'usage_tracker', None),
        )

        def _generate():
            from analyzer.case_intelligence.db import update_ci_report
            update_ci_report(report_id, content='', status='generating')
            generator.generate(run_id, report_id, instructions, template)

        _threading.Thread(target=_generate, daemon=True,
                           name=f'ci-report-{report_id[:8]}').start()

        return jsonify({'report_id': report_id, 'status': 'generating'}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/reports/<report_id>', methods=['GET'])
@login_required
def ci_get_report(run_id, report_id):
    """Get report content."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_report
        report = get_ci_report(report_id)
        if not report or report['run_id'] != run_id:
            return jsonify({'error': 'Report not found'}), 404
        return jsonify(dict(report))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/runs/<run_id>/reports/<report_id>/pdf', methods=['GET'])
@login_required
def ci_download_report_pdf(run_id, report_id):
    """Download report as PDF."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_report
        from analyzer.case_intelligence.report_generator import ReportGenerator

        report = get_ci_report(report_id)
        if not report or report['run_id'] != run_id:
            return jsonify({'error': 'Report not found'}), 404
        if report['status'] != 'complete' or not report['content']:
            return jsonify({'error': 'Report not yet complete'}), 400

        generator = ReportGenerator(llm_clients={}, usage_tracker=None)
        pdf_bytes = generator.generate_pdf(report['content'], title=f'CI Report {run_id[:8]}')

        if not pdf_bytes:
            # Fallback: return markdown as text file
            response = make_response(report['content'])
            response.headers['Content-Type'] = 'text/markdown; charset=utf-8'
            response.headers['Content-Disposition'] = f'attachment; filename="ci_report_{report_id[:8]}.md"'
            return response

        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="ci_report_{report_id[:8]}.pdf"'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/authority/ingest', methods=['POST'])
@login_required
@admin_required
def ci_ingest_authorities():
    """Trigger authority corpus ingestion (admin only)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import os as _os
        import threading as _threading
        from analyzer.case_intelligence.authority_ingester import AuthorityIngester
        from analyzer.case_intelligence.authority_retriever import AuthorityRetriever
        from analyzer.case_intelligence.db import init_ci_db

        init_ci_db()
        data = request.json or {}
        sources = data.get('sources', ['nysenate', 'ecfr', 'courtlistener'])

        def _ingest():
            ingester = AuthorityIngester(
                courtlistener_token=_os.environ.get('COURTLISTENER_API_TOKEN'),
                nysenate_token=_os.environ.get('NYSENATE_API_TOKEN'),
            )
            results = ingester.ingest_all(sources=sources)
            logger.info(f"Authority ingestion complete: {results}")

            # Embed newly added authorities
            retriever = AuthorityRetriever(
                cohere_api_key=_os.environ.get('COHERE_API_KEY'),
            )
            if retriever.enabled:
                embedded = retriever.embed_pending_authorities(batch_size=200)
                logger.info(f"Embedded {embedded} new authorities")

        _threading.Thread(target=_ingest, daemon=True, name='ci-authority-ingest').start()
        return jsonify({'status': 'ingestion_started', 'sources': sources})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ci/authority/status')
@login_required
def ci_authority_status():
    """Authority corpus statistics."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_authority_corpus_stats
        from analyzer.case_intelligence.authority_retriever import AuthorityRetriever
        import os as _os

        db_stats = get_authority_corpus_stats()
        retriever = AuthorityRetriever(cohere_api_key=_os.environ.get('COHERE_API_KEY'))
        chroma_stats = retriever.get_corpus_stats()
        return jsonify({
            'db': db_stats,
            'chroma': chroma_stats,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _build_ci_llm_clients() -> dict:
    """
    Build LLM client dict for CI components.
    Returns {'openai': client_or_None, 'anthropic': client_or_None}
    """
    import os as _os
    clients = {}

    # Get usage tracker from document_analyzer if available
    _usage_tracker = None
    if hasattr(app, 'document_analyzer') and hasattr(app.document_analyzer, 'usage_tracker'):
        _usage_tracker = app.document_analyzer.usage_tracker

    # Check provider from env
    lm_provider = _os.environ.get('LLM_PROVIDER', 'anthropic').lower()

    # LLM_API_KEY belongs to the configured provider only — never cross-assign
    _generic_key = _os.environ.get('LLM_API_KEY', '')
    openai_key = _os.environ.get('OPENAI_API_KEY') or (
        _generic_key if lm_provider == 'openai' else ''
    )
    anthropic_key = _os.environ.get('ANTHROPIC_API_KEY') or (
        _generic_key if lm_provider == 'anthropic' else ''
    )

    # Try to reuse the existing app llm_client
    existing_client = getattr(app, 'llm_client', None)
    if existing_client:
        if lm_provider == 'openai':
            clients['openai'] = existing_client
            # Also build an anthropic client if key exists
            if anthropic_key and anthropic_key != openai_key:
                try:
                    from analyzer.llm.llm_client import LLMClient
                    clients['anthropic'] = LLMClient(
                        provider='anthropic',
                        api_key=anthropic_key,
                        usage_tracker=_usage_tracker,
                    )
                except Exception:
                    pass
        else:
            clients['anthropic'] = existing_client
            # Also build an openai client if key exists
            if openai_key:
                try:
                    from analyzer.llm.llm_client import LLMClient
                    clients['openai'] = LLMClient(
                        provider='openai',
                        api_key=openai_key,
                        usage_tracker=_usage_tracker,
                    )
                except Exception:
                    pass

    return clients


# =============================================================================
# END CASE INTELLIGENCE AI ROUTES
# =============================================================================


# =============================================================================
# COURT DOCUMENT IMPORTER ROUTES  (v3.5.0)
# Gated by COURT_IMPORT_ENABLED=true environment variable.
# =============================================================================

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
                app.document_analyzer.analyze_document(full_doc)
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
                app.document_analyzer.analyze_document(full_doc)
                ok += 1
            except Exception as e:
                logger.warning(f"_analyze_missing_for_project: doc {doc_id} failed: {e}")

        logger.info(f"_analyze_missing_for_project({project_slug}): {ok}/{len(missing)} analyzed")
        return ok

    except Exception as e:
        logger.error(f"_analyze_missing_for_project({project_slug}) error: {e}", exc_info=True)
        return 0


@app.route('/api/projects/<slug>/analyze-missing', methods=['POST'])
@login_required
def api_project_analyze_missing(slug):
    """Trigger background AI analysis for all Paperless docs in <slug> not yet in ChromaDB."""
    if not hasattr(app, 'document_analyzer') or not app.document_analyzer:
        return jsonify({'error': 'Analyzer not running'}), 503
    from threading import Thread
    Thread(target=_analyze_missing_for_project, args=(slug,), daemon=True).start()
    return jsonify({'success': True, 'message': f'Scanning project {slug} for unanalyzed docs'})


def _run_court_import(job_id: str, project_slug: str, court_system: str,
                      case_id: str, cancel_event=None):
    """
    Background worker: download all docket entries for a case and upload to Paperless.
    Called from CourtImportJobManager in a daemon thread.
    """
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

        paperless_client = getattr(app, 'paperless_client', None)
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
            title = f"{case_id} \u2014 {entry.seq}: {entry.title[:80]}"

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
        if imported > 0 and hasattr(app, 'document_analyzer') and app.document_analyzer:
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


# ── Credential routes ────────────────────────────────────────────────────────

@app.route('/api/court/credentials', methods=['POST'])
@login_required
@advanced_required
def court_save_credentials():
    """Save (upsert) court credentials for the current project."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = request.get_json(force=True) or {}
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


@app.route('/api/court/credentials', methods=['GET'])
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


@app.route('/api/court/credentials/test', methods=['POST'])
@login_required
@advanced_required
def court_test_credentials():
    """Test court credentials and return account info."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = request.get_json(force=True) or {}
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


@app.route('/api/court/credentials/<court_system>', methods=['DELETE'])
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


# ── Search / docket routes ───────────────────────────────────────────────────

@app.route('/api/court/search', methods=['POST'])
@login_required
def court_search():
    """Search for cases by case number or party name."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = request.get_json(force=True) or {}
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


@app.route('/api/court/docket/<court_system>/<path:case_id>', methods=['GET'])
@login_required
def court_get_docket(court_system, case_id):
    """Return the full docket entry list for a case."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')

    try:
        connector = _build_court_connector(court_system, project_slug)
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


# ── Import job routes ────────────────────────────────────────────────────────

@app.route('/api/court/import/start', methods=['POST'])
@login_required
def court_import_start():
    """Create and start a background import job."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = request.get_json(force=True) or {}
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


@app.route('/api/court/import/status/<job_id>', methods=['GET'])
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


@app.route('/api/court/import/cancel/<job_id>', methods=['POST'])
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


@app.route('/api/court/import/history', methods=['GET'])
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


@app.route('/api/court/credentials/parse', methods=['POST'])
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
    ok, err = _court_gate()
    if not ok:
        return err

    data = request.get_json(force=True) or {}
    raw_text     = (data.get('raw_text') or '').strip()
    conversation = data.get('conversation') or []

    if not raw_text and not conversation:
        return jsonify({'error': 'raw_text or conversation required'}), 400

    llm = getattr(app, 'llm_client', None)
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


# =============================================================================
# END COURT DOCUMENT IMPORTER ROUTES
# =============================================================================


# =============================================================================
# AI FORM FILLER ROUTES
# =============================================================================

@app.route('/api/ai-form/parse', methods=['POST'])
@login_required
def ai_form_parse():
    """
    Generic AI form-field extractor.  Accepts free-form text (or a multi-turn
    conversation) plus a field schema, and returns structured field values.

    Request JSON:
        schema       (list)  — [{name, label, description, secret, required}]
        conversation (list)  — [{role, content}, ...] full history including
                               the latest user message to send
        project_slug (str)   — optional, default 'default'

    Response JSON:
        fields       {fieldName: value|null, ...}
        summary      str
        follow_up    str|null
        complete     bool
        notes        str|null
    """
    import json as _json

    data         = request.get_json(force=True) or {}
    schema       = data.get('schema') or []
    conversation = data.get('conversation') or []
    project_slug = (data.get('project_slug') or
                    session.get('current_project', 'default') or 'default')

    if not conversation:
        return jsonify({'error': 'conversation required'}), 400
    if not schema:
        return jsonify({'error': 'schema required'}), 400

    # ── Build system prompt from schema ──────────────────────────────────────
    field_lines = []
    field_names = []
    for f in schema:
        name   = f.get('name', '')
        label  = f.get('label', name)
        desc   = f.get('description', '')
        req    = f.get('required', False)
        secret = f.get('secret', False)
        line   = f'  - "{name}" ({label})'
        if desc:   line += f': {desc}'
        if req:    line += ' [REQUIRED]'
        if secret: line += ' [sensitive]'
        field_lines.append(line)
        field_names.append(name)

    fields_template  = ', '.join(f'"{n}": null' for n in field_names)
    response_template = (
        '{"fields": {' + fields_template + '}, '
        '"summary": "plain English of what was found", '
        '"follow_up": "single clarifying question or null", '
        '"complete": false, '
        '"notes": "any other important observations or null"}'
    )

    system_prompt = (
        "You are an expert at extracting structured data from unstructured text "
        "(emails, Slack messages, notes, etc.).\n\n"
        "Fields to extract:\n"
        + '\n'.join(field_lines) + "\n\n"
        "RULES:\n"
        "  - Extract as many fields as possible from the provided text.\n"
        "  - If a required field is missing or ambiguous, ask ONE clarifying question.\n"
        "  - Ask follow-up questions ONE AT A TIME — never ask multiple at once.\n"
        "  - Set complete:true when you have enough information to fill the form "
        "(all required fields are present or can be reasonably inferred).\n"
        "  - Leave optional fields as null if not found.\n\n"
        "Respond with ONLY valid JSON — no markdown fences, no extra text:\n"
        + response_template
    )

    # ── Resolve LLM via project AI config ────────────────────────────────────
    chat_cfg = get_project_ai_config(project_slug, 'chat')
    full_cfg = load_ai_config()

    def _global_key(p):
        return full_cfg.get('global', {}).get(p, {}).get('api_key', '').strip()

    provider = chat_cfg.get('provider', 'openai')
    api_key  = (chat_cfg.get('api_key') or '').strip() or _global_key(provider)
    model    = chat_cfg.get('model', 'gpt-4o-mini')

    # Fallback provider
    if not api_key:
        fb_prov = chat_cfg.get('fallback_provider')
        api_key  = _global_key(fb_prov) if fb_prov else ''
        if api_key:
            provider = fb_prov
            model    = chat_cfg.get('fallback_model', model)

    if not api_key:
        return jsonify({
            'error': 'No AI API key configured — set up an AI provider in Settings first.'
        }), 503

    try:
        raw_response = ''

        if provider == 'openai':
            import openai as _oai
            client = _oai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{'role': 'system', 'content': system_prompt}] + conversation,
                temperature=0,
                max_tokens=800,
            )
            raw_response = resp.choices[0].message.content or ''

        elif provider == 'anthropic':
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=800,
                system=system_prompt,
                messages=conversation,
            )
            raw_response = resp.content[0].text if resp.content else ''

        else:
            return jsonify({'error': f'Unsupported AI provider: {provider}'}), 400

        # Strip markdown fences if present
        stripped = raw_response.strip()
        if stripped.startswith('```'):
            stripped = stripped.split('\n', 1)[1]
            stripped = stripped.rsplit('```', 1)[0]

        parsed = _json.loads(stripped)
        return jsonify(parsed)

    except Exception as e:
        logger.error(f'ai_form_parse error: {e}')
        return jsonify({'error': str(e)}), 500


# =============================================================================
# END AI FORM FILLER ROUTES
# =============================================================================


def run_web_server(state_manager, profile_loader, paperless_client, host='0.0.0.0', port=8051,
                  project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    """
    Run the Flask web server in a separate thread using Waitress (production WSGI server).

    Args:
        state_manager: StateManager instance
        profile_loader: ProfileLoader instance
        paperless_client: PaperlessClient instance
        host: Host to bind to
        port: Port to bind to
        project_manager: ProjectManager instance (v1.5.0)
        llm_client: LLM client instance (v1.5.0)
        smart_uploader: SmartUploader instance (v1.5.0)
        document_analyzer: DocumentAnalyzer instance (v1.5.0 - for re-analysis)
    """
    create_app(state_manager, profile_loader, paperless_client,
              project_manager=project_manager, llm_client=llm_client, smart_uploader=smart_uploader,
              document_analyzer=document_analyzer)

    logger.info(f"Starting production web UI on {host}:{port}")

    # Use Waitress for production-grade serving
    from waitress import serve
    serve(
        app,
        host=host,
        port=port,
        threads=4,  # Handle 4 concurrent requests
        channel_timeout=300,  # 5 minute timeout
        cleanup_interval=10,  # Clean up connections every 10s
        _quiet=False  # Show startup message
    )


def start_web_server_thread(state_manager, profile_loader, paperless_client, host='0.0.0.0', port=8051,
                           project_manager=None, llm_client=None, smart_uploader=None, document_analyzer=None):
    """
    Start web server in background thread.

    Args:
        state_manager: StateManager instance
        profile_loader: ProfileLoader instance
        paperless_client: PaperlessClient instance
        host: Host to bind to
        port: Port to bind to
        project_manager: ProjectManager instance (v1.5.0)
        llm_client: LLM client instance (v1.5.0)
        smart_uploader: SmartUploader instance (v1.5.0)
        document_analyzer: DocumentAnalyzer instance (v1.5.0 - for re-analysis)
    """
    # Set up log buffer handler
    root_logger = logging.getLogger()
    buffer_handler = LogBufferHandler()
    buffer_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(buffer_handler)
    logger.info("Log buffer handler registered - live logs enabled")

    thread = Thread(
        target=run_web_server,
        args=(state_manager, profile_loader, paperless_client, host, port,
              project_manager, llm_client, smart_uploader, document_analyzer),
        daemon=True
    )
    thread.start()
    logger.info(f"Web UI thread started on {host}:{port}")
    return thread
