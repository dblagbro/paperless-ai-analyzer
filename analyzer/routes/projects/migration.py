"""Migration endpoints: doc shuffling between projects, and moving a project
to its own Paperless instance."""
import logging
import sqlite3

from flask import current_app, jsonify
from flask_login import login_required

from analyzer.app import admin_required, safe_json_body
from analyzer.services.project_provisioning_service import (
    _migrate_project_to_own_paperless,
    _migration_status,
)

from . import bp

logger = logging.getLogger(__name__)


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


@bp.route('/api/projects/migrate-documents', methods=['POST'])
@login_required
@admin_required
def api_migrate_documents():
    """Migrate documents from one project to another."""
    if not current_app.project_manager or not current_app.paperless_client:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = safe_json_body()
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
