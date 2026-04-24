"""Project Paperless-ngx auto-provisioning endpoints.

Heavy lifting lives in `analyzer/services/project_provisioning_service.py`.
These routes are thin wrappers around the shared queue + status dicts.
"""
import logging

from flask import current_app, jsonify
from flask_login import login_required

from analyzer.services.project_provisioning_service import (
    PROVISION_MIN_INTERVAL_SECS,
    _provision_status,
    enqueue_provision,
)

from . import bp

logger = logging.getLogger(__name__)


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
        if existing.get('status') in ('running', 'queued_waiting'):
            return jsonify({'error': 'Provisioning already in progress or queued'}), 409

        provision_state = enqueue_provision(slug)
        return jsonify({
            'success': True,
            'message': f'Provisioning queued for project {slug}',
            'queue_position': provision_state.get('queue_position'),
            'eta_seconds': provision_state.get('eta_seconds'),
            'throttle_interval_secs': PROVISION_MIN_INTERVAL_SECS,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
