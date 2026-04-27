"""Project CRUD, archive/unarchive, and current-project session state."""
import logging
import os

from flask import current_app, jsonify, request, session
from flask_login import login_required

from analyzer.app import safe_json_body
from analyzer.services.project_provisioning_service import (
    PROVISION_MIN_INTERVAL_SECS,
    enqueue_provision,
)

from . import bp

logger = logging.getLogger(__name__)


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
        data = safe_json_body()

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

        # v3.9.5: Paperless tag creation moved off the request thread — a slow
        # shared Paperless used to block POST /api/projects for 15s+ waiting on
        # get_or_create_tag. It now happens asynchronously so the UI gets an
        # instant 201. The provision worker also creates this tag as part of
        # setup; no functional loss.
        if current_app.paperless_client:
            _color = project.get('color', '#3498db')
            _pc = current_app.paperless_client
            import threading as _threading
            def _bg_create_tag(pc, slug_, color_):
                try:
                    pc.get_or_create_tag(f"project:{slug_}", color=color_)
                except Exception as tag_err:
                    logger.warning(f"Async tag creation failed for {slug_}: {tag_err}")
            _threading.Thread(target=_bg_create_tag, args=(_pc, slug, _color), daemon=True).start()

        provision_state = enqueue_provision(slug)

        logger.info(f"Created project: {slug}")
        payload = dict(project)
        payload['provision'] = {
            'status': provision_state.get('status'),
            'queue_position': provision_state.get('queue_position'),
            'eta_seconds': provision_state.get('eta_seconds'),
            'throttle_interval_secs': PROVISION_MIN_INTERVAL_SECS,
        }
        return jsonify(payload), 201

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
        data = safe_json_body()
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
    """Delete project.

    v3.9.13: also tear down provisioned per-project resources (containers,
    postgres DB, nginx block) so the host doesn't accumulate orphans across
    test/regression runs. Previously delete only removed the analyzer DB row
    and ChromaDB data, leaving stale `paperless-web-<slug>`,
    `paperless-consumer-<slug>` containers, the `paperless_<slug>` postgres
    database, and the auto-generated nginx location block — those then
    surfaced as 502s on future curl checks and required manual cleanup.
    """
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    try:
        delete_data = request.args.get('delete_data', type=bool, default=True)

        # Tear down provisioned resources first (idempotent — safe even if
        # this project never had its own Paperless instance).
        deprovision_summary = None
        try:
            from analyzer.services.project_provisioning_service import deprovision_project_paperless
            deprovision_summary = deprovision_project_paperless(slug)
        except Exception as e:
            logger.warning(f"Deprovision step for {slug} failed (continuing): {e}")

        if delete_data:
            from analyzer.vector_store import VectorStore
            vs = VectorStore(project_slug=slug)
            if vs.enabled:
                vs.delete_collection()

        success = current_app.project_manager.delete_project(slug, delete_data=delete_data)

        if success:
            logger.info(f"Deleted project: {slug}")
            payload = {'success': True, 'message': f'Project {slug} deleted'}
            if deprovision_summary:
                payload['deprovision'] = deprovision_summary
            return jsonify(payload)
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
        data = safe_json_body()
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
