import logging
import os
from flask import Blueprint, request, jsonify, session, current_app
from flask_login import login_required

from analyzer.app import admin_required, _get_project_client, _project_client_cache

logger = logging.getLogger(__name__)

bp = Blueprint('projects', __name__)

# In-memory migration state: {slug: {status, total, migrated, failed, error, phase}}
_migration_status: dict = {}

# In-memory provisioning state: {slug: {status, phase, error}}
_provision_status: dict = {}


# ---------------------------------------------------------------------------
# Private helpers (must be defined before routes that reference them)
# ---------------------------------------------------------------------------

def _get_docker_client():
    """Return docker.DockerClient or None — never raises."""
    try:
        import docker
        return docker.from_env(timeout=3)
    except Exception:
        return None


def _provision_log(slug: str, phase: str, status: str = None):
    s = _provision_status.setdefault(slug, {})
    s['phase'] = phase
    if status:
        s['status'] = status
    logger.info(f"[Provision:{slug}] {phase}")


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route('/api/projects', methods=['GET'])
@login_required
def api_list_projects():
    """List all projects."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        projects = current_app.project_manager.list_projects(include_archived=True)

        from analyzer.court_db import get_court_doc_count
        for project in projects:
            try:
                stats = current_app.project_manager.get_statistics(project['slug'])
                project.update(stats)
            except Exception:
                pass
            try:
                project['court_doc_count'] = get_court_doc_count(project['slug'])
            except Exception:
                project['court_doc_count'] = 0
            try:
                cfg = current_app.project_manager.get_paperless_config(project['slug'])
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


@bp.route('/api/projects', methods=['POST'])
@login_required
def api_create_project():
    """Create new project."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        data = request.json

        if not data.get('name'):
            return jsonify({'error': 'Project name is required'}), 400

        slug = data.get('slug')
        if not slug:
            slug = current_app.project_manager.suggest_slug(data['name'])

        project = current_app.project_manager.create_project(
            slug=slug,
            name=data['name'],
            description=data.get('description', ''),
            color=data.get('color'),
            metadata=data.get('metadata')
        )

        if current_app.paperless_client:
            try:
                current_app.paperless_client.get_or_create_tag(
                    f"project:{slug}",
                    color=project.get('color', '#3498db')
                )
            except Exception as tag_err:
                logger.warning(f"Could not create Paperless tag for project {slug}: {tag_err}")

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


@bp.route('/api/projects/<slug>', methods=['GET'])
@login_required
def api_get_project(slug):
    """Get project details."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = current_app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        stats = current_app.project_manager.get_statistics(slug)
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


@bp.route('/api/projects/<slug>', methods=['PUT'])
@login_required
def api_update_project(slug):
    """Update project metadata."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        data = request.json
        project = current_app.project_manager.update_project(slug, **data)
        logger.info(f"Updated project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to update project: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/<slug>', methods=['DELETE'])
@login_required
def api_delete_project(slug):
    """Delete project."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        delete_data = request.args.get('delete_data', type=bool, default=True)

        if delete_data:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=slug)
            if vs.enabled:
                vs.delete_collection()

        success = current_app.project_manager.delete_project(slug, delete_data=delete_data)

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


@bp.route('/api/projects/<slug>/archive', methods=['POST'])
@login_required
def api_archive_project(slug):
    """Archive project."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = current_app.project_manager.archive_project(slug)
        logger.info(f"Archived project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to archive project: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/<slug>/unarchive', methods=['POST'])
@login_required
def api_unarchive_project(slug):
    """Unarchive project."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project = current_app.project_manager.unarchive_project(slug)
        logger.info(f"Unarchived project: {slug}")
        return jsonify(project)

    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Failed to unarchive project: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/<slug>/paperless-config', methods=['POST'])
@login_required
def api_set_project_paperless_config(slug):
    """Save per-project Paperless-ngx connection config."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = current_app.project_manager.get_project(slug)
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

        current_app.project_manager.update_project(slug, **updates)
        _project_client_cache.pop(slug, None)

        logger.info(f"Updated Paperless config for project '{slug}'")
        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Failed to set Paperless config for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/<slug>/paperless-config', methods=['GET'])
@login_required
def api_get_project_paperless_config(slug):
    """Get per-project Paperless-ngx connection config (token is masked)."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = current_app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        cfg = current_app.project_manager.get_paperless_config(slug)
        return jsonify({
            'url': cfg.get('url') or '',
            'token_set': bool(cfg.get('token')),
            'doc_base_url': cfg.get('doc_base_url') or '',
        })
    except Exception as e:
        logger.error(f"Failed to get Paperless config for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/<slug>/provision-snippets', methods=['GET'])
@login_required
def api_provision_snippets(slug):
    """Return ready-to-paste infrastructure snippets for a per-project Paperless instance."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = current_app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        try:
            all_projects = current_app.project_manager.list_projects(include_archived=True)
            projects_with_instance = [
                p['slug'] for p in all_projects
                if current_app.project_manager.get_paperless_config(p['slug']).get('url')
            ]
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


@bp.route('/api/projects/<slug>/migrate-to-own-paperless', methods=['POST'])
@login_required
def api_migrate_to_own_paperless(slug):
    """Start a background migration of all project docs from shared to per-project Paperless."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = current_app.project_manager.get_project(slug)
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


@bp.route('/api/projects/<slug>/migration-status', methods=['GET'])
@login_required
def api_migration_status(slug):
    """Return current migration status for a project."""
    status = _migration_status.get(slug, {'status': 'idle'})
    return jsonify(status)


@bp.route('/api/projects/<slug>/provision-status', methods=['GET'])
@login_required
def api_provision_status(slug):
    """Return current auto-provisioning status for a project."""
    status = _provision_status.get(slug, {'status': 'idle'})
    return jsonify(status)


@bp.route('/api/projects/<slug>/reprovision', methods=['POST'])
@login_required
def api_reprovision(slug):
    """Trigger automated Paperless provisioning for an existing project."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        project = current_app.project_manager.get_project(slug)
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


@bp.route('/api/projects/<slug>/paperless-health-check', methods=['POST'])
@login_required
def api_paperless_health_check(slug):
    """Test-connect a Paperless URL + token without saving."""
    try:
        data = request.json or {}
        url = (data.get('url') or '').strip().rstrip('/')
        token = (data.get('token') or '').strip()
        if not url:
            return jsonify({'ok': False, 'error': 'url is required'}), 400
        if not token and current_app.project_manager:
            cfg = current_app.project_manager.get_paperless_config(slug)
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


@bp.route('/api/projects/<slug>/doc-link/<int:doc_id>', methods=['GET'])
@login_required
def api_project_doc_link(slug, doc_id):
    """Return the public Paperless URL for a specific document in a project."""
    if not current_app.project_manager:
        return jsonify({'url': None})
    try:
        cfg = current_app.project_manager.get_paperless_config(slug)
        base = (cfg.get('doc_base_url') or '').rstrip('/')
        url = f"{base}/documents/{doc_id}/details" if base else None
        return jsonify({'url': url})
    except Exception as e:
        logger.error(f"doc-link error for {slug}/{doc_id}: {e}")
        return jsonify({'url': None})


@bp.route('/api/projects/<slug>/documents', methods=['GET'])
@login_required
def api_list_project_documents(slug):
    """List all documents in a project's Chroma collection."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503
    try:
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=slug)
        if not vs.enabled:
            return jsonify({'documents': [], 'count': 0})
        raw = vs.collection.get(include=['metadatas'])
        try:
            _pcfg = current_app.project_manager.get_paperless_config(slug) if current_app.project_manager else {}
            _base_url = (_pcfg.get('doc_base_url') or '').rstrip('/')
        except Exception:
            _base_url = ''
        if not _base_url:
            _base_url = os.environ.get('PAPERLESS_PUBLIC_BASE_URL', '').rstrip('/')

        docs = []
        for i, doc_id in enumerate(raw.get('ids', [])):
            m = raw['metadatas'][i]
            try:
                _did = int(m.get('document_id', 0))
            except (ValueError, TypeError):
                continue
            if not _did:
                continue
            anomalies_str = m.get('anomalies', '')
            anomalies_list = [a.strip() for a in anomalies_str.split(',') if a.strip()] if anomalies_str else []
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


@bp.route('/api/projects/<slug>/documents/<int:doc_id>', methods=['DELETE'])
@login_required
def api_delete_project_document(slug, doc_id):
    """Delete a document from Paperless-ngx, Chroma, and processed_documents."""
    import sqlite3 as _sqlite3
    warnings = []

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

    try:
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=slug)
        if vs.enabled:
            vs.delete_document(doc_id)
    except Exception as e:
        warnings.append(f"Chroma: {e}")
        logger.warning(f"Could not delete doc {doc_id} from Chroma (slug={slug}): {e}")

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


@bp.route('/api/current-project', methods=['GET'])
@login_required
def api_get_current_project():
    """Get currently selected project from session."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        project_slug = session.get('current_project', 'default')
        project = current_app.project_manager.get_project(project_slug)

        if not project:
            project = current_app.project_manager.get_project('default')
            session['current_project'] = 'default'

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to get current project: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/current-project', methods=['POST'])
@login_required
def api_set_current_project():
    """Set current project in session."""
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        data = request.json
        project_slug = data.get('project_slug')

        if not project_slug:
            return jsonify({'error': 'project_slug is required'}), 400

        project = current_app.project_manager.get_project(project_slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        session['current_project'] = project_slug
        logger.info(f"Switched to project: {project_slug}")

        return jsonify(project)

    except Exception as e:
        logger.error(f"Failed to set current project: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/orphan-documents', methods=['GET'])
@login_required
def api_list_orphan_documents():
    """List documents without project tag."""
    if not current_app.paperless_client:
        return jsonify({'error': 'Paperless client not available'}), 503

    try:
        orphans = current_app.paperless_client.get_documents_without_project()

        orphan_list = []
        for doc in orphans[:100]:
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


@bp.route('/api/assign-project', methods=['POST'])
@login_required
def api_assign_project_to_documents():
    """Assign project to one or more documents."""
    if not current_app.paperless_client or not current_app.project_manager:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = request.json
        document_ids = data.get('document_ids', [])
        project_slug = data.get('project_slug')

        if not document_ids or not project_slug:
            return jsonify({'error': 'document_ids and project_slug are required'}), 400

        project = current_app.project_manager.get_project(project_slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        success_count = 0
        failed = []

        for doc_id in document_ids:
            if current_app.paperless_client.add_project_tag(doc_id, project_slug, color=project.get('color')):
                success_count += 1
            else:
                failed.append(doc_id)

        if success_count > 0:
            current_app.project_manager.increment_document_count(project_slug, delta=success_count)

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


@bp.route('/api/projects/<slug>/reanalyze', methods=['POST'])
@login_required
def api_reanalyze_project(slug):
    """Manually trigger re-analysis of all documents in a project with full context."""
    if not current_app.project_manager or not current_app.document_analyzer:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        project = current_app.project_manager.get_project(slug)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        logger.info(f"Manual re-analysis requested for project '{slug}'")

        from threading import Thread
        def run_reanalysis():
            try:
                current_app.document_analyzer.re_analyze_project(slug)
            except Exception as e:
                logger.error(f"Background re-analysis failed: {e}")

        Thread(target=run_reanalysis, daemon=True).start()

        return jsonify({
            'success': True,
            'message': f'Re-analysis started for project: {slug}',
            'note': 'This runs in background and may take several minutes. Check logs for progress.'
        })

    except Exception as e:
        logger.error(f"Failed to trigger re-analysis: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/projects/migrate-documents', methods=['POST'])
@login_required
@admin_required
def api_migrate_documents():
    """Migrate documents from one project to another."""
    if not current_app.project_manager or not current_app.paperless_client:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = request.json
        source_slug = data.get('source_project')
        dest_slug = data.get('destination_project')
        document_ids = data.get('document_ids', [])

        if not source_slug or not dest_slug:
            return jsonify({'error': 'Both source_project and destination_project are required'}), 400

        if source_slug == dest_slug:
            return jsonify({'error': 'Source and destination projects must be different'}), 400

        source_project = current_app.project_manager.get_project(source_slug)
        dest_project = current_app.project_manager.get_project(dest_slug)

        if not source_project:
            return jsonify({'error': f'Source project not found: {source_slug}'}), 404
        if not dest_project:
            return jsonify({'error': f'Destination project not found: {dest_slug}'}), 404

        logger.info(f"Migration requested: {source_slug} → {dest_slug} ({len(document_ids) if document_ids else 'all'} docs)")

        from threading import Thread
        migration_result = {'status': 'running', 'migrated': 0, 'errors': 0}

        def run_migration():
            try:
                from analyzer.vector_store import VectorStore
                import sqlite3

                source_tag = f"project:{source_slug}"
                dest_tag = f"project:{dest_slug}"

                source_tag_id = current_app.paperless_client.get_or_create_tag(source_tag, color='#95a5a6')
                dest_tag_id = current_app.paperless_client.get_or_create_tag(dest_tag, color='#e74c3c')

                if not source_tag_id or not dest_tag_id:
                    raise Exception("Failed to get/create project tags")

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

                        doc = current_app.paperless_client.get_document(doc_id)
                        if doc:
                            current_tags = doc.get('tags', [])
                            updated_tags = [t for t in current_tags if t != source_tag_id]
                            if dest_tag_id not in updated_tags:
                                updated_tags.append(dest_tag_id)
                            current_app.paperless_client.update_document(doc_id, {'tags': updated_tags})

                        migrated_count += 1

                    except Exception as e:
                        logger.error(f"Failed to migrate document {doc_id}: {e}")
                        error_count += 1

                try:
                    db_path = '/app/data/app.db'
                    con = sqlite3.connect(db_path)
                    if document_ids:
                        for doc_id in docs_to_migrate:
                            con.execute(
                                'UPDATE processed_documents SET project_slug=? WHERE doc_id=?',
                                (dest_slug, doc_id)
                            )
                    else:
                        con.execute(
                            'UPDATE processed_documents SET project_slug=? WHERE project_slug=?',
                            (dest_slug, source_slug)
                        )
                    con.commit()
                    con.close()
                except Exception as e:
                    logger.warning(f"Could not update processed_documents after migration: {e}")

                try:
                    db_path = '/app/data/app.db'
                    con = sqlite3.connect(db_path)
                    session_rows = con.execute(
                        'UPDATE chat_sessions SET project_slug=? WHERE project_slug=?',
                        (dest_slug, source_slug)
                    )
                    con.commit()
                    con.close()
                    logger.info(f"Migrated {session_rows.rowcount} chat session(s)")
                except Exception as e:
                    logger.warning(f"Could not migrate chat sessions: {e}")

                try:
                    ci_db_path = '/app/data/case_intelligence.db'
                    con = sqlite3.connect(ci_db_path)
                    r = con.execute(
                        'UPDATE ci_runs SET project_slug=? WHERE project_slug=?',
                        (dest_slug, source_slug)
                    )
                    con.commit()
                    con.close()
                    logger.info(f"Migrated {r.rowcount} CI run(s)")
                except Exception as e:
                    logger.warning(f"Could not migrate CI runs: {e}")

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
                    logger.info(f"Migrated {r_jobs.rowcount} court import job(s) and {r_docs.rowcount} court imported doc(s)")
                except Exception as e:
                    logger.warning(f"Could not migrate court import data: {e}")

                try:
                    current_app.project_manager.update_document_count(source_slug, source_vs.collection.count())
                    current_app.project_manager.update_document_count(dest_slug, dest_vs.collection.count())
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

        Thread(target=run_migration, daemon=True).start()

        return jsonify({
            'success': True,
            'message': f'Migration started: {source_slug} → {dest_slug}',
            'note': 'Migration runs in background. Check logs for progress.'
        })

    except Exception as e:
        logger.error(f"Failed to start migration: {e}")
        return jsonify({'error': str(e)}), 500
