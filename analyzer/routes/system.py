import logging
import os
from datetime import datetime
from email.message import EmailMessage
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user

from analyzer.app import admin_required, ui_state, log_buffer, safe_json_body
from analyzer.services.smtp_service import (
    load_smtp_settings as _load_smtp_settings,
    save_smtp_settings as _save_smtp_settings,
    smtp_send as _smtp_send,
)

logger = logging.getLogger(__name__)

bp = Blueprint('system', __name__)

try:
    from analyzer.app import _APP_VERSION
except ImportError:
    _APP_VERSION = '0.0.0'

# ---------------------------------------------------------------------------
# Health check constants
# ---------------------------------------------------------------------------

PAPERLESS_CONTAINERS = [
    'paperless-web', 'paperless-consumer', 'paperless-redis', 'paperless-postgres',
]
HEALTH_TIMEOUT = 1.8  # seconds per component

_own_container_name_cache = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_own_container_name() -> str:
    """Return this container's Docker name. Cached after first successful lookup."""
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
    prefix = os.environ.get('URL_PREFIX', '').strip('/')
    _own_container_name_cache = prefix or 'paperless-ai-analyzer'
    return _own_container_name_cache


def _get_managed_containers() -> list:
    """Return the 4 core Paperless containers, per-project containers, plus this analyzer instance."""
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


def _check_paperless_api(paperless_client=None):
    """Check Paperless API health with timing.

    Takes the client as an argument so the caller (api_system_health) can
    capture `current_app.paperless_client` on the request thread and hand it
    to ThreadPoolExecutor workers — which otherwise fail with "Working outside
    of application context" when they try to dereference current_app.
    """
    import time
    start = time.monotonic()
    try:
        if paperless_client is None:
            paperless_client = current_app.paperless_client
        healthy = paperless_client.health_check()
        latency_ms = int((time.monotonic() - start) * 1000)
        if healthy:
            return {'status': 'ok', 'latency_ms': latency_ms, 'detail': f'Responded in {latency_ms}ms'}
        return {'status': 'error', 'latency_ms': latency_ms, 'detail': 'Health check returned false'}
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        return {'status': 'error', 'latency_ms': latency_ms, 'detail': str(e)[:120]}


def _check_chromadb(document_analyzer=None):
    """Check ChromaDB by counting documents in the default vector store.
    See note on _check_paperless_api for why dependencies are passed in."""
    import time
    start = time.monotonic()
    try:
        da = document_analyzer if document_analyzer is not None else getattr(current_app, 'document_analyzer', None)
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
    from analyzer.services.ai_config_service import load_ai_config
    try:
        cfg = load_ai_config()
        global_keys = cfg.get('global', {})
        configured = [p for p, v in global_keys.items() if v.get('api_key', '').strip()]
    except Exception:
        configured = []

    env_provider = os.environ.get('LLM_PROVIDER', 'anthropic')
    env_key = os.environ.get('LLM_API_KEY', '').strip()
    if env_key and env_provider not in configured:
        configured.append(env_provider)

    if not configured:
        return {'status': 'error', 'latency_ms': 0, 'detail': 'No AI provider keys configured'}

    detail = ', '.join(f'{p} ✓' for p in configured)
    return {'status': 'ok', 'latency_ms': 0, 'detail': detail}


def _check_analyzer_loop(state_manager=None):
    """Check whether the analyzer loop has run recently.
    See note on _check_paperless_api for why dependencies are passed in."""
    try:
        if state_manager is None:
            state_manager = current_app.state_manager
        stats = state_manager.get_stats()
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
        threshold = poll_interval * 10
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
    """Aggregate health check for all per-project Paperless containers."""
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route('/api/smtp-settings', methods=['GET'])
@login_required
def api_get_smtp_settings():
    if not current_user.role == 'admin':
        return jsonify({'error': 'Admin only'}), 403
    s = _load_smtp_settings()
    s_safe = {**s, 'pass': '••••••••' if s.get('pass') else ''}
    return jsonify(s_safe)


@bp.route('/api/smtp-settings', methods=['POST'])
@login_required
def api_save_smtp_settings():
    if not current_user.role == 'admin':
        return jsonify({'error': 'Admin only'}), 403
    data = safe_json_body()
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
    raw_pass = str(data.get('pass', ''))
    if raw_pass and raw_pass != '••••••••':
        updated['pass'] = raw_pass
    else:
        updated['pass'] = current.get('pass', '')
    _save_smtp_settings(updated)
    return jsonify({'ok': True, 'message': 'SMTP settings saved'})


@bp.route('/api/smtp-settings/test', methods=['POST'])
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


@bp.route('/api/about')
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


@bp.route('/api/bug-report', methods=['POST'])
@login_required
def api_bug_report():
    if request.is_json:
        _d = safe_json_body()
        description   = (_d.get('description') or '').strip()
        severity      = (_d.get('severity') or 'Medium').strip()
        contact_email = (_d.get('contact_email') or '').strip()
        include_logs  = _d.get('include_logs', True)
        if isinstance(include_logs, str):
            include_logs = include_logs.lower() != 'false'
    else:
        description   = (request.form.get('description') or '').strip()
        severity      = (request.form.get('severity') or 'Medium').strip()
        contact_email = (request.form.get('contact_email') or '').strip()
        include_logs  = request.form.get('include_logs', 'true').lower() != 'false'
    browser_info = request.headers.get('User-Agent', 'Unknown')

    if not description:
        return jsonify({'error': 'Please describe the problem'}), 400

    smtp_cfg = _load_smtp_settings()
    dest = smtp_cfg.get('bug_report_to', 'dblagbro@voipguru.org')

    log_snippet = ''
    if include_logs:
        try:
            buf = list(log_buffer)[-60:]
            log_snippet = '\n'.join(buf)
        except Exception:
            log_snippet = '(logs unavailable)'

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


@bp.route('/health')
def health():
    """Health check endpoint."""
    try:
        healthy = current_app.paperless_client.health_check()
        if healthy:
            return jsonify({'status': 'healthy'}), 200
        else:
            return jsonify({'status': 'unhealthy', 'reason': 'paperless_api_unreachable'}), 503
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'reason': str(e)}), 503


@bp.route('/api/system-health')
@login_required
def api_system_health():
    """Parallel health check for all system components."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone
    dc = _get_docker_client()
    docker_available = dc is not None

    # Capture app-context dependencies on the request thread; the ThreadPoolExecutor
    # workers run outside the Flask request context so `current_app` proxies fail there.
    paperless_client = getattr(current_app, 'paperless_client', None)
    document_analyzer = getattr(current_app, 'document_analyzer', None)
    state_manager = getattr(current_app, 'state_manager', None)

    checks = {
        'paperless_api': (lambda: _check_paperless_api(paperless_client)),
        'chromadb':      (lambda: _check_chromadb(document_analyzer)),
        'llm':           _check_llm,
        'analyzer_loop': (lambda: _check_analyzer_loop(state_manager)),
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


@bp.route('/api/containers')
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


@bp.route('/api/containers/<name>/restart', methods=['POST'])
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


@bp.route('/api/containers/<name>/logs')
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
