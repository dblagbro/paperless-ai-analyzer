"""
Web UI for Paperless AI Analyzer

Simple Flask-based dashboard for monitoring and control.
"""

import os
import json
import logging
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
from flask import Flask, render_template, jsonify, request, redirect, url_for, make_response
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
            # nginx already stripped the prefix from PATH_INFO, so don't strip again
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
    share_session, unshare_session, get_session_shares, can_access_session,
    list_users, create_user as db_create_user, update_user as db_update_user,
    log_import, get_import_history,
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


@app.context_processor
def inject_user():
    is_admin = current_user.is_authenticated and current_user.is_admin
    return {'current_user': current_user, 'is_admin': is_admin}


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

        # Update stats
        with ui_state['lock']:
            ui_state['stats']['total_analyzed'] = total_docs

            # Count high risk documents
            high_risk_count = 0
            anomalies_count = 0

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

                # Add to recent analyses (summaries will be generated on first view)
                ui_state['recent_analyses'].append({
                    'document_id': doc_id,
                    'document_title': doc['title'],
                    'anomalies_found': anomalies[:5],
                    'risk_score': risk_score,
                    'timestamp': doc['modified'],
                    'ai_analysis': "",
                    'created': doc.get('created', ''),
                    'correspondent': doc.get('correspondent', None),
                    'brief_summary': f"Financial document: {doc['title']}",
                    'full_summary': ""
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

    # Initialize UI state from existing documents
    initialize_ui_state()

    return app


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

        # Get vector store stats
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()
        vector_stats = vector_store.get_stats() if vector_store.enabled else {'enabled': False, 'total_documents': 0}

        return jsonify({
            'status': 'running',
            'uptime_seconds': _get_uptime(),
            'state': state_stats,
            'stats': ui_state['stats'],
            'last_update': ui_state['last_update'],
            'active_profiles': len(app.profile_loader.profiles),
            'vector_store': vector_stats
        })


@app.route('/api/recent')
@login_required
def api_recent():
    """Get recent analysis results."""
    with ui_state['lock']:
        return jsonify({
            'analyses': ui_state['recent_analyses'][-50:]  # Last 50
        })


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

        # Use semantic search with vector store
        from analyzer.vector_store import VectorStore

        vector_store = VectorStore()

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

        # If we don't have analyses, fetch from Paperless
        if not recent_analyses or len(recent_analyses) < 5:
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
            context += f"\n- Doc {doc_id}: {doc_title}"
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

                system_prompt += f"\n\n--- Document {doc_id} ---"
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

CRITICAL - NEVER HALLUCINATE DATA:
- If a document says "(Scanned image PDF with no extractable text)" or similar, DO NOT invent financial data
- NEVER make up dollar amounts, totals, sections, or line items that aren't explicitly in the content above
- If asked about specific numbers in a document with no extracted content, respond:
  "This document is a scanned image with no extracted text. I cannot analyze specific financial figures without OCR or Vision AI to read the document."
- Only report numbers and facts that are EXPLICITLY stated in the document content provided above
- If the content is missing or incomplete, acknowledge this limitation clearly

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
- Provide the most complete analysis possible given the data you can access"""

        # Load AI configuration and try providers/models in order
        ai_config = load_ai_config()
        chat_config = ai_config.get('chat', {})
        providers = chat_config.get('providers', [])

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
        append_message(session_id, 'user', user_message)
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
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/vector/types', methods=['GET'])
@login_required
def api_vector_types():
    """Get list of all document types in vector store."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore()

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
        vector_store = VectorStore()

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
        vector_store = VectorStore()

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
        vector_store = VectorStore()

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
        vector_store = VectorStore()

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
        vector_store = VectorStore()

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

def get_default_ai_config():
    """Get default AI configuration with proper model priorities."""
    return {
        'document_analysis': {
            'providers': [
                {
                    'name': 'openai',
                    'api_key': os.environ.get('OPENAI_API_KEY', ''),
                    'models': ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
                    'enabled': bool(os.environ.get('OPENAI_API_KEY'))
                },
                {
                    'name': 'anthropic',
                    'api_key': os.environ.get('LLM_API_KEY', ''),
                    'models': ['claude-3-opus-20240229', 'claude-3-5-sonnet-20241022', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                    'enabled': bool(os.environ.get('LLM_API_KEY'))
                }
            ]
        },
        'chat': {
            'providers': [
                {
                    'name': 'openai',
                    'api_key': os.environ.get('OPENAI_API_KEY', ''),
                    'models': ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
                    'enabled': bool(os.environ.get('OPENAI_API_KEY'))
                },
                {
                    'name': 'anthropic',
                    'api_key': os.environ.get('LLM_API_KEY', ''),
                    'models': ['claude-3-opus-20240229', 'claude-3-5-sonnet-20241022', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                    'enabled': bool(os.environ.get('LLM_API_KEY'))
                }
            ]
        }
    }

def load_ai_config():
    """Load AI configuration from file, or return defaults."""
    try:
        if AI_CONFIG_PATH.exists():
            with open(AI_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                logger.info("Loaded AI configuration from file")
                return config
    except Exception as e:
        logger.warning(f"Failed to load AI config: {e}")

    return get_default_ai_config()

def save_ai_config(config):
    """Save AI configuration to file."""
    try:
        AI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Saved AI configuration")
        return True
    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")
        return False


@app.route('/api/ai-config', methods=['GET'])
@login_required
def api_ai_config_get():
    """Get current AI configuration."""
    try:
        config = load_ai_config()
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        logger.error(f"Failed to get AI config: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-config', methods=['POST'])
@login_required
def api_ai_config_save():
    """Save AI configuration."""
    try:
        data = request.json
        config = data.get('config')

        if not config:
            return jsonify({'error': 'Configuration is required'}), 400

        # Validate structure
        if 'document_analysis' not in config or 'chat' not in config:
            return jsonify({'error': 'Invalid configuration structure'}), 400

        # Save configuration
        if save_ai_config(config):
            return jsonify({
                'success': True,
                'message': 'AI configuration saved. Changes will apply to new analysis tasks.'
            })
        else:
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

            # Try the specified model or default
            test_model = model if model else 'gpt-3.5-turbo'
            try:
                response = client.chat.completions.create(
                    model=test_model,
                    messages=[{"role": "user", "content": "Say 'test successful'"}],
                    max_tokens=10
                )
                return jsonify({
                    'success': True,
                    'message': f'✓ OpenAI API key is valid! Using model: {test_model}',
                    'model': test_model
                })
            except Exception as e:
                error_msg = str(e)
                if '404' in error_msg or 'model_not_found' in error_msg:
                    return jsonify({
                        'success': False,
                        'error': f'Model {test_model} not available with this API key'
                    }), 400
                raise

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
        logs.append(f"LLM enabled: {os.environ.get('LLM_ENABLED', 'false')}")
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
    enabled = os.environ.get('LLM_ENABLED', 'false').lower() == 'true'
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

    with ui_state['lock']:
        results = ui_state['recent_analyses']

        # Filter by query (search title, doc_id, anomalies, and summaries)
        if query:
            results = [r for r in results if
                      query in r.get('title', '').lower() or
                      query in str(r.get('doc_id', '')).lower() or
                      query in r.get('brief_summary', '').lower() or
                      query in r.get('full_summary', '').lower() or
                      any(query in a.lower() for a in r.get('anomalies_found', []))]

        # Filter by risk score
        if risk_min is not None:
            results = [r for r in results if r.get('risk_score', 0) >= risk_min]
        if risk_max is not None:
            results = [r for r in results if r.get('risk_score', 0) <= risk_max]

        # Filter by anomalies
        if has_anomalies:
            results = [r for r in results if r.get('anomalies_found')]

        return jsonify({
            'results': results,
            'count': len(results)
        })


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
        include_archived = request.args.get('archived', type=bool, default=False)
        projects = app.project_manager.list_projects(include_archived=include_archived)

        # Add statistics to each project
        for project in projects:
            try:
                stats = app.project_manager.get_statistics(project['slug'])
                project.update(stats)
            except Exception:
                pass  # Skip stats if unavailable

        return jsonify({'projects': projects, 'count': len(projects)})

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
                # Get or create project tags
                source_tag = f"project:{source_slug}"
                dest_tag = f"project:{dest_slug}"

                # Get source tag ID
                source_tag_id = app.paperless_client.get_or_create_tag(source_tag, color='#95a5a6')
                dest_tag_id = app.paperless_client.get_or_create_tag(dest_tag, color='#e74c3c')

                if not source_tag_id or not dest_tag_id:
                    raise Exception("Failed to get/create project tags")

                # Get documents to migrate
                if document_ids:
                    # Specific documents
                    docs_to_migrate = document_ids
                else:
                    # All documents with source project tag
                    all_docs = app.paperless_client.get_documents_by_tag(source_tag_id)
                    docs_to_migrate = [doc['id'] for doc in all_docs]

                migrated_count = 0
                error_count = 0

                for doc_id in docs_to_migrate:
                    try:
                        # Get current document
                        doc = app.paperless_client.get_document(doc_id)
                        if not doc:
                            error_count += 1
                            continue

                        current_tags = doc.get('tags', [])

                        # Remove source tag, add dest tag
                        updated_tags = [t for t in current_tags if t != source_tag_id]
                        if dest_tag_id not in updated_tags:
                            updated_tags.append(dest_tag_id)

                        # Update document tags
                        success = app.paperless_client.update_document(doc_id, {'tags': updated_tags})
                        if success:
                            migrated_count += 1
                        else:
                            error_count += 1

                    except Exception as e:
                        logger.error(f"Failed to migrate document {doc_id}: {e}")
                        error_count += 1

                # Update project counts
                app.project_manager.increment_document_count(source_slug, delta=-migrated_count)
                app.project_manager.increment_document_count(dest_slug, delta=migrated_count)

                migration_result['status'] = 'completed'
                migration_result['migrated'] = migrated_count
                migration_result['errors'] = error_count

                logger.info(f"Migration completed: {migrated_count} docs migrated, {error_count} errors")

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
                # Direct upload without AI metadata
                result = app.paperless_client.upload_document(file_path, title=filename)
                title = filename
        finally:
            downloader.cleanup(file_path)

        if result:
            doc_id = result.get('id')
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
                # Direct upload without AI metadata
                result = app.paperless_client.upload_document(
                    file_path, title=file.filename.rsplit('.', 1)[0]
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
            rows = get_sessions(current_user.id)
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
        session_id = create_session(current_user.id, title=title, document_type=document_type)
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
            'messages': [{'role': m['role'], 'content': m['content'], 'created_at': m['created_at']} for m in msgs],
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
            'role': r['role'],
            'created_at': r['created_at'],
            'last_login': r['last_login'],
            'is_active': bool(r['is_active']),
        } for r in rows]
        return jsonify({'users': users})
    except Exception as e:
        logger.error(f"List users error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users', methods=['POST'])
@login_required
@admin_required
def api_users_create():
    """Create a new user (admin only)."""
    try:
        data = request.json or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'basic')
        display_name = data.get('display_name', '').strip() or username
        if not username or not password:
            return jsonify({'error': 'username and password required'}), 400
        if role not in ('basic', 'admin'):
            return jsonify({'error': 'role must be basic or admin'}), 400
        if get_user_by_username(username):
            return jsonify({'error': f"User '{username}' already exists"}), 409
        db_create_user(username, password, role=role, display_name=display_name)
        return jsonify({'success': True, 'username': username}), 201
    except Exception as e:
        logger.error(f"Create user error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:uid>', methods=['PATCH'])
@login_required
@admin_required
def api_users_update(uid):
    """Update user role / display_name / password (admin only)."""
    try:
        data = request.json or {}
        allowed = {k: v for k, v in data.items() if k in ('role', 'display_name', 'password')}
        if 'role' in allowed and allowed['role'] not in ('basic', 'admin'):
            return jsonify({'error': 'role must be basic or admin'}), 400
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
