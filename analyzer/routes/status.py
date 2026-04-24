import os
from datetime import datetime
from flask import Blueprint, jsonify, session, current_app
from flask_login import login_required

from analyzer.app import ui_state, _get_uptime, _get_project_client
from analyzer.db import count_processed_documents

bp = Blueprint('status', __name__)


@bp.route('/api/status')
@login_required
def api_status():
    """Get current analyzer status."""
    with ui_state['lock']:
        state_stats = current_app.state_manager.get_stats()

        project_slug = session.get('current_project', 'default')
        try:
            project_analyzed = count_processed_documents(project_slug=project_slug)
        except Exception:
            project_analyzed = ui_state['stats'].get('total_analyzed', 0)

        from analyzer.vector_store import VectorStore
        vector_store = VectorStore(project_slug=project_slug)
        vector_stats = vector_store.get_stats() if vector_store.enabled else {'enabled': False, 'total_documents': 0}

        try:
            all_meta = vector_store.collection.get(include=['metadatas'])['metadatas']
            project_anomalies = sum(1 for m in all_meta if m.get('anomalies', '').strip())
            project_high_risk = sum(1 for m in all_meta if int(m.get('risk_score') or 0) >= 70)
        except Exception:
            project_anomalies = ui_state['stats'].get('anomalies_detected', 0)
            project_high_risk = ui_state['stats'].get('high_risk_count', 0)

        chroma_count = vector_stats.get('total_documents', project_analyzed) if vector_stats.get('enabled') else project_analyzed

        from analyzer.court_db import get_court_doc_count, get_pending_ocr_count
        try:
            court_count = get_court_doc_count(project_slug)
        except Exception:
            court_count = 0

        paperless_total = chroma_count
        awaiting_ocr = awaiting_ai = 0
        try:
            _pc = _get_project_client(project_slug)
            # v3.9.5: skip Paperless calls when the server is unreachable —
            # avoids a 15-45s tenacity retry loop blocking /api/status.
            if _pc and _pc.health_check():
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
            'active_profiles': len(current_app.profile_loader.profiles),
            'vector_store': vector_stats,
            'court_doc_count': court_count,
            'awaiting_ocr': awaiting_ocr,
            'awaiting_ai': awaiting_ai,
            'paperless_total': paperless_total,
            # v3.9.4: flat aliases at top level for API consumers that expect
            # these canonical names (e.g. regression tests, external dashboards).
            'total_documents': paperless_total,
            'analyzed_documents': chroma_count,
            'analyzed_count': chroma_count,
        })


@bp.route('/api/recent')
@login_required
def api_recent():
    """Get recent analysis results for the current project."""
    project_slug = session.get('current_project', 'default')

    try:
        from analyzer.vector_store import VectorStore
        vs = VectorStore(project_slug=project_slug)
        if vs.enabled:
            raw = vs.collection.get(include=['metadatas'])
            metas = raw.get('metadatas') or []
            metas_sorted = sorted(
                metas,
                key=lambda m: m.get('timestamp', ''),
                reverse=True
            )[:50]
            try:
                _rcfg = current_app.project_manager.get_paperless_config(project_slug) if current_app.project_manager else {}
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
        import logging
        logging.getLogger(__name__).warning(f"Chroma recent query failed for {project_slug}, falling back: {e}")

    with ui_state['lock']:
        filtered = [
            a for a in ui_state['recent_analyses']
            if a.get('project_slug', 'default') == project_slug
        ]
        return jsonify({'analyses': filtered[-50:]})
