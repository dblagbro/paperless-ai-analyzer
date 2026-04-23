import logging
from flask import Blueprint, request, jsonify, session, current_app
from flask_login import login_required

from analyzer.app import safe_json_body

logger = logging.getLogger(__name__)

bp = Blueprint('vector', __name__)


@bp.route('/api/vector/types', methods=['GET'])
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


@bp.route('/api/vector/delete/<int:document_id>', methods=['POST'])
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


@bp.route('/api/vector/delete-document', methods=['POST'])
@login_required
def api_vector_delete_document_json():
    """Delete a specific document from vector store (JSON body)."""
    try:
        data = safe_json_body()
        doc_id = data.get('doc_id')

        if not doc_id:
            return jsonify({'error': 'doc_id required'}), 400

        # v3.9.4: coerce to int — callers may send a stringy doc_id
        try:
            doc_id = int(doc_id)
        except (ValueError, TypeError):
            return jsonify({'error': f'doc_id must be numeric, got {doc_id!r}'}), 400

        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        success = vector_store.delete_document(doc_id)

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


@bp.route('/api/vector/delete-by-type', methods=['POST'])
@login_required
def api_vector_delete_by_type():
    """Delete all documents of a specific type from vector store."""
    try:
        data = safe_json_body()
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


@bp.route('/api/vector/clear', methods=['POST'])
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


@bp.route('/api/vector/documents', methods=['GET'])
@login_required
def api_vector_documents():
    """Get all documents from vector store with details, grouped by type."""
    try:
        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=session.get('current_project', 'default'))

        if not vector_store.enabled:
            return jsonify({'error': 'Vector store not enabled'}), 503

        try:
            all_docs = vector_store.collection.get(include=['metadatas'])

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


@bp.route('/api/vector/reembed-stale', methods=['POST'])
@login_required
def api_vector_reembed_stale():
    """Trigger a stale-embedding check: re-analyze docs whose OCR content changed after embedding."""
    try:
        da = getattr(current_app, 'document_analyzer', None)
        if not da:
            return jsonify({'error': 'Analyzer not running'}), 503

        # Capture the analyzer on the request thread so the background thread
        # doesn't need Flask's app context to access it.
        def _run(document_analyzer):
            try:
                count = document_analyzer.check_stale_embeddings()
                logger.info(f"Manual stale embedding check complete: {count} re-analyzed")
            except Exception as e:
                logger.error(f"Manual stale embedding check failed: {e}")

        from threading import Thread
        Thread(target=_run, args=(da,), daemon=True).start()
        return jsonify({'success': True, 'message': 'Stale embedding check started in background — check logs for progress'})

    except Exception as e:
        logger.error(f"Failed to start stale embedding check: {e}")
        return jsonify({'error': str(e)}), 500
