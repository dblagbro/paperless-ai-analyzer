"""Project document operations: list/delete project docs, orphans, bulk assign, re-analysis."""
import logging
import os
import sqlite3

from flask import current_app, jsonify
from flask_login import login_required

from analyzer.app import _get_project_client, safe_json_body, admin_required

from . import bp

logger = logging.getLogger(__name__)


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
        with sqlite3.connect('/app/data/app.db') as conn:
            conn.execute("DELETE FROM processed_documents WHERE doc_id = ?", (doc_id,))
    except Exception as e:
        warnings.append(f"DB: {e}")
        logger.warning(f"Could not delete doc {doc_id} from processed_documents: {e}")

    if warnings:
        return jsonify({'success': True, 'warnings': warnings,
                        'message': f'Document {doc_id} removed with warnings: ' + '; '.join(warnings)})

    logger.info(f"Deleted document {doc_id} from project {slug} (Paperless + Chroma + DB)")
    return jsonify({'success': True, 'message': f'Document {doc_id} deleted'})


@bp.route('/api/projects/<slug>/cleanup-stale-embeddings', methods=['POST'])
@login_required
@admin_required
def api_cleanup_stale_embeddings(slug):
    """
    Purge Chroma rows for a project whose numeric doc_id no longer exists in
    the project's Paperless instance. Fixes the "Document not found" symptom
    on Search & Analysis when Paperless docs were deleted but Chroma retained
    the embedding. CI finding rows (string IDs like "ci:...") are untouched.
    """
    if not current_app.project_manager:
        return jsonify({'error': 'Project management not enabled'}), 503

    # Collect Paperless doc IDs that actually belong to this project today.
    # - Primary path: per-project client lists all docs in its dedicated
    #   Paperless instance.
    # - Fallback path: if the project client is unreachable (stale token /
    #   unprovisioned child container), use the global client filtered by the
    #   `project:<slug>` tag — matches the set of docs that nginx+Paperless
    #   would show under the project's public view.
    def _collect_paginated(fetch_page):
        ids = set()
        page = 1
        while True:
            resp = fetch_page(page)
            for d in resp.get('results', []) or []:
                if isinstance(d, dict) and 'id' in d:
                    ids.add(int(d['id']))
            if not resp.get('next'):
                break
            page += 1
            if page > 500:
                logger.warning(f"cleanup-stale-embeddings: hit page cap (500) for {slug}")
                break
        return ids

    try:
        pc = _get_project_client(slug)
        if not pc:
            return jsonify({'error': 'No Paperless client available'}), 400

        try:
            live_paperless_ids = _collect_paginated(
                lambda page: pc.get_documents(page=page, page_size=200)
            )
            source = 'project-client'
        except Exception as proj_err:
            logger.warning(
                f"cleanup-stale-embeddings: project client for {slug} failed "
                f"({proj_err}) — falling back to global + project tag filter"
            )
            gpc = getattr(current_app, 'paperless_client', None)
            if not gpc or gpc is pc:
                return jsonify({'error': f'Paperless API unreachable for this project: {proj_err}'}), 502
            live_paperless_ids = _collect_paginated(
                lambda page: gpc.get_documents_by_project(slug, page=page, page_size=200)
            )
            source = 'global-client-project-tag'

        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=slug)
        if not vs.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        all_chroma_ids = vs.collection.get(include=[]).get('ids', [])
        stale_ids = [i for i in all_chroma_ids if i.isdigit() and int(i) not in live_paperless_ids]

        if stale_ids:
            vs.collection.delete(ids=stale_ids)
            try:
                with sqlite3.connect('/app/data/app.db') as conn:
                    conn.executemany(
                        "DELETE FROM processed_documents WHERE doc_id = ?",
                        [(int(i),) for i in stale_ids]
                    )
            except Exception as e:
                logger.warning(f"processed_documents cleanup failed during stale purge ({slug}): {e}")

        try:
            new_count = sum(1 for i in vs.collection.get(include=[]).get('ids', []) if i.isdigit())
            current_app.project_manager.update_document_count(slug, new_count)
        except Exception:
            new_count = None

        logger.info(
            f"cleanup-stale-embeddings({slug}): paperless={len(live_paperless_ids)} "
            f"chroma_before={len(all_chroma_ids)} purged={len(stale_ids)} "
            f"docs_after={new_count} source={source}"
        )
        return jsonify({
            'success': True,
            'paperless_count': len(live_paperless_ids),
            'chroma_before': len(all_chroma_ids),
            'purged': len(stale_ids),
            'docs_after': new_count,
            'source': source,
        })
    except Exception as e:
        logger.error(f"cleanup-stale-embeddings failed for {slug}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/orphan-documents', methods=['GET'])
@login_required
def api_list_orphan_documents():
    """List documents without project tag."""
    if not current_app.paperless_client:
        return jsonify({'error': 'Paperless client not available'}), 503

    # v3.9.5: Short-circuit if Paperless is unreachable — avoids a 30-60s
    # tenacity retry loop on every call. health_check() has a 3s timeout.
    try:
        if not current_app.paperless_client.health_check():
            return jsonify({
                'orphans': [],
                'count': 0,
                'paperless_available': False,
                'detail': 'Paperless-ngx not reachable; returning empty list',
            })
    except Exception:
        return jsonify({
            'orphans': [],
            'count': 0,
            'paperless_available': False,
            'detail': 'Paperless-ngx health check failed; returning empty list',
        })

    try:
        orphans = current_app.paperless_client.get_documents_without_project()
    except Exception as e:
        logger.warning(f"Paperless unreachable for orphan listing: {str(e)[:120]}")
        return jsonify({
            'orphans': [],
            'count': 0,
            'paperless_available': False,
            'detail': 'Paperless-ngx not reachable; returning empty list',
        })

    orphan_list = []
    for doc in orphans[:100]:
        # v3.9.7: Paperless occasionally returns non-dict entries mid-list; skip
        # them instead of 500ing the whole response.
        if not isinstance(doc, dict):
            continue
        try:
            orphan_list.append({
                'id': doc['id'],
                'title': doc['title'],
                'created': doc.get('created'),
                'correspondent': doc.get('correspondent'),
                'tags': [t['name'] for t in doc.get('tags', []) if isinstance(t, dict) and 'name' in t],
            })
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping malformed orphan doc: {e}")
    return jsonify({'orphans': orphan_list, 'count': len(orphan_list)})


@bp.route('/api/assign-project', methods=['POST'])
@login_required
def api_assign_project_to_documents():
    """Assign project to one or more documents."""
    if not current_app.paperless_client or not current_app.project_manager:
        return jsonify({'error': 'Required services not available'}), 503

    try:
        data = safe_json_body()
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
