"""Per-project Paperless-ngx provisioning and migration.

Spins up a dedicated Paperless-ngx instance (web + consumer + postgres + redis
DB index + nginx location block) for a given project slug, and handles
migrating documents from the shared Paperless instance to the dedicated one.

Extracted verbatim from `analyzer/routes/projects.py` during the 2026-04-23
maintainability refactor. Function names, docstrings, and bodies preserved
exactly; only module-level collaborators (flask imports, Blueprint) have been
stripped since this module has no routes.

Public surface (imported by routes/projects.py):
    _provision_status    dict[slug -> {status, phase, error}]
    _migration_status    dict[slug -> {status, phase, total, migrated, failed, error}]
    _get_docker_client() -> docker.DockerClient | None
    _provision_log(slug, phase, status=None)
    _migration_log(slug, phase, migrated=None, total=None, failed=None)
    _provision_project_paperless(slug) -> None      (daemon-thread target)
    _migrate_project_to_own_paperless(slug) -> None (daemon-thread target)
"""
import logging
import os
import threading
import time
from collections import deque
from typing import Optional

from flask import current_app

from analyzer.app import _get_project_client, _project_client_cache

logger = logging.getLogger(__name__)

# In-memory migration state: {slug: {status, total, migrated, failed, error, phase}}
_migration_status: dict = {}

# In-memory provisioning state: {slug: {status, phase, error}}
_provision_status: dict = {}

# ── v3.9.5: provision throttling ─────────────────────────────────────────────
# Spinning up a dedicated Paperless stack (web + consumer + postgres + redis)
# costs a lot of CPU/RAM during the first minute. Back-to-back provisions from
# rapid-fire project creation (e.g. regression tests) can saturate the host,
# cause timeouts on live traffic, and starve earlier stacks before they finish
# warming up.
#
# Policy:
#   - First project after boot may start immediately.
#   - Every subsequent project waits PROVISION_MIN_INTERVAL_SECS (default 180s)
#     after the previous provision *started*.
#   - Requests arriving during the cooldown are queued. A single worker thread
#     drains the queue sequentially, sleeping as needed.
#
# Status is exposed via _provision_status[slug] so the UI can display:
#   {'status': 'queued_waiting', 'queue_position': N, 'eta_seconds': S, ...}

PROVISION_MIN_INTERVAL_SECS = int(os.environ.get('PROVISION_MIN_INTERVAL_SECS', '180'))

_provision_queue: "deque[str]" = deque()
_provision_queue_lock = threading.Lock()
_provision_queue_cv = threading.Condition(_provision_queue_lock)
_last_provision_start_at: float = 0.0
_provision_worker_started = False


def _eta_for_position(position: int) -> int:
    """Seconds until the Nth-in-line request will start provisioning."""
    now = time.time()
    head_ready_at = max(now, _last_provision_start_at + PROVISION_MIN_INTERVAL_SECS)
    return max(0, int(head_ready_at - now) + (position - 1) * PROVISION_MIN_INTERVAL_SECS)


def _refresh_queue_eta_locked() -> None:
    """Update eta_seconds and queue_position for every slug currently queued.
    Caller must hold _provision_queue_lock."""
    for idx, s in enumerate(list(_provision_queue), start=1):
        st = _provision_status.get(s)
        if st and st.get('status') == 'queued_waiting':
            st['queue_position'] = idx
            st['eta_seconds'] = _eta_for_position(idx)


def _provision_worker() -> None:
    """Single background worker: pop slugs off the queue and provision them,
    enforcing the cooldown between starts."""
    global _last_provision_start_at
    while True:
        with _provision_queue_cv:
            while not _provision_queue:
                _provision_queue_cv.wait()
            # Refresh ETAs first so UI sees accurate countdowns
            _refresh_queue_eta_locked()
            slug = _provision_queue[0]
            wait = max(0.0, (_last_provision_start_at + PROVISION_MIN_INTERVAL_SECS) - time.time())
        if wait > 0:
            logger.info(f"Provision throttle: waiting {wait:.0f}s before starting {slug}")
            # Sleep in 5s chunks so eta_seconds in status stays live
            end_at = time.time() + wait
            while time.time() < end_at:
                with _provision_queue_lock:
                    _refresh_queue_eta_locked()
                time.sleep(min(5.0, end_at - time.time()))
        with _provision_queue_cv:
            # Re-check: the slug could have been cancelled
            if not _provision_queue or _provision_queue[0] != slug:
                continue
            _provision_queue.popleft()
            _last_provision_start_at = time.time()
            _refresh_queue_eta_locked()
        # Provision synchronously inside the worker — serializes the heavy work.
        try:
            _provision_project_paperless(slug)
        except Exception as e:
            logger.error(f"Provision worker: {slug} failed: {e}")


def _ensure_worker_started() -> None:
    global _provision_worker_started
    if _provision_worker_started:
        return
    with _provision_queue_lock:
        if _provision_worker_started:
            return
        t = threading.Thread(target=_provision_worker, daemon=True, name='provision-worker')
        t.start()
        _provision_worker_started = True


def enqueue_provision(slug: str) -> dict:
    """Public API: request provisioning for `slug`. Respects the cooldown and
    queues if necessary. Returns the status dict the route should echo back.

    On return, _provision_status[slug] is always populated so the UI poll works.
    """
    _ensure_worker_started()
    with _provision_queue_cv:
        position = len(_provision_queue) + 1
        eta = _eta_for_position(position)
        _provision_queue.append(slug)
        _provision_status[slug] = {
            'status': 'queued_waiting' if eta > 0 else 'queued',
            'phase': (
                f'Waiting {eta}s behind {position - 1} other project(s)'
                if eta > 0 else 'Queued'
            ),
            'error': None,
            'queue_position': position,
            'eta_seconds': eta,
        }
        _refresh_queue_eta_locked()
        _provision_queue_cv.notify()
        return dict(_provision_status[slug])


def _get_docker_client():
    """Return docker.DockerClient or None — never raises."""
    try:
        import docker
        return docker.from_env(timeout=3)
    except Exception:
        return None


# ── Flattened nginx.conf management (v3.10+) ────────────────────────────────
# When the host nginx.conf has been flattened to a single file (sentinel markers
# present), we edit it in place between `# <paperless-projects-begin>` and
# `# <paperless-projects-end>`. Each project gets its own nested marker block:
#
#     # <paperless-project slug="pw-flow-04222114">
#     ...location block...
#     # </paperless-project>
#
# Concurrent project provisioning is serialized with fcntl.flock on the file.
# Returns True on success, False if markers not found (caller should then fall
# back to the legacy per-project file-write path).

_NGINX_CONF_PATH     = '/app/nginx.conf'            # mounted rw from host
_MARKER_BEGIN        = '# <paperless-projects-begin>'
_MARKER_END          = '# <paperless-projects-end>'
_PROJECT_INDENT_SP   = 8                             # voipguru server body indent


def _update_nginx_project_block(slug: str, block_content: str) -> bool:
    """Insert/replace a project's location block in the flattened nginx.conf.
    Returns True on success, False if the nginx.conf doesn't have the sentinel
    markers (caller should fall back to legacy behavior).
    """
    import fcntl
    import re as _re

    if not os.path.isfile(_NGINX_CONF_PATH):
        return False

    try:
        with open(_NGINX_CONF_PATH, 'r+') as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            content = fh.read()

            begin = content.find(_MARKER_BEGIN)
            end   = content.find(_MARKER_END)
            if begin < 0 or end < 0:
                return False

            # Find end of the line containing _MARKER_BEGIN
            begin_line_end = content.index('\n', begin) + 1
            # Find start of the line containing _MARKER_END
            end_line_start = content.rfind('\n', 0, end) + 1

            section = content[begin_line_end:end_line_start]

            # Re-indent the new block body to the server-block body indent
            indent = ' ' * _PROJECT_INDENT_SP
            indented_body = '\n'.join(
                (indent + line if line.strip() else '') for line in block_content.rstrip().split('\n')
            )
            new_sub_block = (
                f'{indent}# <paperless-project slug="{slug}">\n'
                f'{indented_body}\n'
                f'{indent}# </paperless-project>\n'
            )

            # Replace an existing sub-block for this slug (idempotent re-provision),
            # else append to the end of the section.
            pat = _re.compile(
                r'^[ \t]*# <paperless-project slug="' + _re.escape(slug) + r'">'
                r'[\s\S]*?'
                r'^[ \t]*# </paperless-project>\s*\n',
                _re.MULTILINE,
            )
            if pat.search(section):
                new_section = pat.sub(new_sub_block, section)
            else:
                new_section = section + new_sub_block

            new_content = content[:begin_line_end] + new_section + content[end_line_start:]
            fh.seek(0)
            fh.write(new_content)
            fh.truncate()
            # Lock released on file close
        return True
    except Exception as e:
        logger.warning(f"[Provision] _update_nginx_project_block failed for {slug}: {e}")
        return False


def _remove_nginx_project_block(slug: str) -> bool:
    """Remove a project's location block from the flattened nginx.conf.
    Returns True on success or if the block wasn't there. False if markers not found.
    """
    import fcntl
    import re as _re

    if not os.path.isfile(_NGINX_CONF_PATH):
        return False

    try:
        with open(_NGINX_CONF_PATH, 'r+') as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            content = fh.read()

            if _MARKER_BEGIN not in content or _MARKER_END not in content:
                return False

            pat = _re.compile(
                r'^[ \t]*# <paperless-project slug="' + _re.escape(slug) + r'">'
                r'[\s\S]*?'
                r'^[ \t]*# </paperless-project>\s*\n',
                _re.MULTILINE,
            )
            new_content, n = pat.subn('', content)
            if n > 0:
                fh.seek(0)
                fh.write(new_content)
                fh.truncate()
        return True
    except Exception as e:
        logger.warning(f"[Provision] _remove_nginx_project_block failed for {slug}: {e}")
        return False


def _provision_log(slug: str, phase: str, status: str = None):
    s = _provision_status.setdefault(slug, {})
    s['phase'] = phase
    if status:
        s['status'] = status
    logger.info(f"[Provision:{slug}] {phase}")


def deprovision_project_paperless(slug: str) -> dict:
    """Tear down per-project Paperless resources during project deletion.

    Counterpart to _provision_project_paperless(). Idempotent — safe to call
    when only a subset of resources exist (e.g. nginx block but no container).

    Removes (best-effort, in order):
      1. paperless-web-<slug> + paperless-consumer-<slug> containers
      2. The postgres database paperless_<slug-underscored>
      3. The auto-managed nginx <paperless-project slug="..."> block + reload nginx

    Returns a dict of what happened so the caller can surface warnings.
    """
    result = {
        'containers_removed': [],
        'container_errors': [],
        'db_dropped': None,
        'db_error': None,
        'nginx_block_removed': False,
        'nginx_reloaded': False,
        'nginx_error': None,
    }

    # 1. Stop+remove per-project containers
    dc = _get_docker_client()
    if dc is not None:
        for cname in (f"paperless-web-{slug}", f"paperless-consumer-{slug}"):
            try:
                c = dc.containers.get(cname)
                try:
                    c.stop(timeout=10)
                except Exception:
                    pass
                c.remove(force=True)
                result['containers_removed'].append(cname)
                logger.info(f"[Deprovision:{slug}] removed container {cname}")
            except Exception as e:
                msg = str(e)
                if 'Not Found' in msg or '404' in msg:
                    continue
                result['container_errors'].append(f"{cname}: {msg[:80]}")
                logger.warning(f"[Deprovision:{slug}] container {cname}: {e}")

    # 2. Drop the per-project postgres database
    # Naming follows the provision path: paperless_<slug-with-dashes-as-underscores>
    db_name = f"paperless_{slug.replace('-', '_')}"
    if dc is not None:
        try:
            pg = dc.containers.get('paperless-postgres')
            # First terminate any open connections (DROP DATABASE fails if any exist)
            pg.exec_run([
                'psql', '-U', 'paperless', '-d', 'postgres', '-c',
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname='{db_name}' AND pid<>pg_backend_pid();"
            ])
            ec, out = pg.exec_run([
                'psql', '-U', 'paperless', '-d', 'postgres', '-c',
                f'DROP DATABASE IF EXISTS "{db_name}";'
            ])
            if ec == 0:
                result['db_dropped'] = db_name
                logger.info(f"[Deprovision:{slug}] dropped postgres DB {db_name}")
            else:
                result['db_error'] = out.decode('utf-8', errors='replace')[:200]
        except Exception as e:
            msg = str(e)
            if 'Not Found' in msg or '404' in msg:
                pass  # paperless-postgres not running — nothing to drop
            else:
                result['db_error'] = msg[:200]
                logger.warning(f"[Deprovision:{slug}] postgres drop {db_name}: {e}")

    # 3. Strip nginx block and reload
    try:
        result['nginx_block_removed'] = _remove_nginx_project_block(slug)
        if result['nginx_block_removed'] and dc is not None:
            try:
                nginx_c = dc.containers.get('nginx')
                nginx_c.exec_run(['nginx', '-s', 'reload'])
                result['nginx_reloaded'] = True
                logger.info(f"[Deprovision:{slug}] nginx reloaded")
            except Exception as e:
                result['nginx_error'] = str(e)[:200]
                logger.warning(f"[Deprovision:{slug}] nginx reload failed (non-fatal): {e}")
    except Exception as e:
        result['nginx_error'] = str(e)[:200]
        logger.warning(f"[Deprovision:{slug}] nginx block removal: {e}")

    # Drop the cached project client so future calls don't reuse stale config
    try:
        _project_client_cache.pop(slug, None)
    except Exception:
        pass

    return result


def _provision_project_paperless(slug: str) -> None:
    """Daemon thread: spin up a dedicated Paperless-ngx instance for the given project."""
    import secrets as _secrets
    import time as _time
    import os as _os

    _provision_status[slug] = {'status': 'running', 'phase': 'Starting', 'error': None}

    try:
        dc = _get_docker_client()
        if dc is None:
            raise RuntimeError("Docker SDK not available — cannot auto-provision")

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

        _provision_log(slug, 'Generating credentials')
        secret_key = _secrets.token_hex(32)
        admin_password = _secrets.token_urlsafe(16)

        _provision_log(slug, 'Creating Postgres database')
        db_name = f"paperless_{slug.replace('-', '_')}"
        try:
            pg = dc.containers.get('paperless-postgres')
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

        _provision_log(slug, 'Creating host directories')
        base_dir = f"/mnt/s/documents/paperless-{slug}"
        for subdir in ['data', 'media', 'consume', 'export', 'tmp']:
            _os.makedirs(f"{base_dir}/{subdir}", exist_ok=True)

        web_name = f"paperless-web-{slug}"
        consumer_name = f"paperless-consumer-{slug}"
        image = 'ghcr.io/paperless-ngx/paperless-ngx:latest'

        _provision_log(slug, f'Starting {web_name}')
        for cname in [web_name, consumer_name]:
            try:
                existing = dc.containers.get(cname)
                existing.stop(timeout=10)
                existing.remove()
            except Exception:
                pass

        existing_url = pw_env.get('PAPERLESS_URL', 'https://voipguru.org').rstrip('/')
        existing_csrf = pw_env.get('PAPERLESS_CSRF_TRUSTED_ORIGINS', existing_url)
        existing_allowed = pw_env.get('PAPERLESS_ALLOWED_HOSTS', existing_url.split('/')[2])
        domain = existing_url.split('/')[2]
        web_env = {
            'PAPERLESS_REDIS': f'redis://paperless-redis:6379/{redis_db}',
            'PAPERLESS_DBHOST': 'paperless-postgres',
            'PAPERLESS_DBNAME': db_name,
            'PAPERLESS_DBUSER': 'paperless',
            'PAPERLESS_DBPASS': db_pass,
            'PAPERLESS_SECRET_KEY': secret_key,
            'PAPERLESS_URL': existing_url,
            'PAPERLESS_FORCE_SCRIPT_NAME': f'/paperless-{slug}',
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
            image, name=web_name, detach=True,
            environment=web_env, volumes=web_vols,
            network='docker_default',
            restart_policy={'Name': 'unless-stopped'},
            labels={'managed-by': 'paperless-ai-analyzer', 'project': slug},
        )

        _provision_log(slug, f'Starting {consumer_name}')
        consumer_env = {k: v for k, v in web_env.items()}
        dc.containers.run(
            image, name=consumer_name, command='document_consumer',
            detach=True, environment=consumer_env, volumes=web_vols,
            network='docker_default',
            restart_policy={'Name': 'unless-stopped'},
            labels={'managed-by': 'paperless-ai-analyzer', 'project': slug},
        )

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
                ready = True
                break
            except Exception:
                pass
            _time.sleep(5)

        if not ready:
            raise RuntimeError(
                f"{web_name} did not become ready within 8 minutes. "
                f"Check logs: docker logs {web_name}"
            )

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

        _provision_log(slug, 'Obtaining Paperless API token')
        import json as _json
        import urllib.parse as _urlparse
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
        # Try the new marker-based nginx.conf edit first (v3.10+ deployment).
        # Falls back to the legacy per-project file write if markers aren't present
        # in nginx.conf (rollback or not-yet-migrated environments).
        if not _update_nginx_project_block(slug, nginx_conf):
            nginx_conf_path = f'/app/nginx-projects-locations.d/paperless-{slug}.conf'
            with open(nginx_conf_path, 'w') as f:
                f.write(nginx_conf)

        _provision_log(slug, 'Reloading nginx')
        try:
            nginx_c = dc.containers.get('nginx')
            nginx_c.exec_run(['nginx', '-s', 'reload'])
        except Exception as e:
            logger.warning(f"[Provision:{slug}] nginx reload failed (non-fatal): {e}")

        _provision_log(slug, 'Saving project config to database')
        paperless_url = f'http://{web_name}:8000'
        doc_base_url = f'https://{domain}/paperless-{slug}'
        current_app.project_manager.update_project(slug,
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
    """Daemon thread: migrate all project documents from the shared Paperless instance
    to the project's dedicated Paperless instance."""
    import sqlite3 as _sqlite3
    import re as _re
    status = _migration_status.setdefault(slug, {})
    status['status'] = 'running'
    status['error'] = None

    try:
        _migration_log(slug, 'preflight')
        new_client = _get_project_client(slug)
        if new_client is current_app.paperless_client:
            raise ValueError("No dedicated Paperless instance configured. "
                             "Save URL + token on the Connect tab first.")
        if not new_client.health_check():
            raise RuntimeError(f"New Paperless instance is not reachable at {new_client.base_url}")
        old_client = current_app.paperless_client

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

        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=slug)
        doc_id_map = {}
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
                                nc = _re.sub(rf'(?<=#){oid}(?!\d)', str(nid), nc)
                            if nc != content:
                                updates.append((nc, msg_id))
                        if updates:
                            conn.executemany("UPDATE chat_messages SET content = ? WHERE id = ?", updates)
                            conn.commit()
                            logger.info(f"[Migration:{slug}] Patched {len(updates)} chat messages")
            except Exception as _che:
                logger.warning(f"[Migration:{slug}] Chat history patch failed: {_che}")

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

