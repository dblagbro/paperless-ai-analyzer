"""Court docket search + fetch (CourtListener, NYSCEF, etc.)."""
import logging

from flask import jsonify, request, session
from flask_login import login_required

from analyzer.app import advanced_required, safe_json_body
from . import bp
from .helpers import _court_gate, _build_court_connector, _get_current_project_slug

logger = logging.getLogger(__name__)

@bp.route('/api/court/search', methods=['POST'])
@login_required
def court_search():
    """Search for cases by case number or party name."""
    ok, err = _court_gate()
    if not ok:
        return err
    data = safe_json_body()
    court_system = data.get('court_system', 'federal')
    case_number = data.get('case_number', '').strip()
    party_name  = data.get('party_name', '').strip()
    court       = data.get('court', '').strip()
    project_slug = data.get('project_slug', 'default')

    if not case_number and not party_name:
        return jsonify({'error': 'Provide case_number or party_name'}), 400

    try:
        connector = _build_court_connector(court_system, project_slug)
        results = connector.search_cases(case_number=case_number,
                                         party_name=party_name,
                                         court=court)
        return jsonify({
            'results': [
                {
                    'case_id':    r.case_id,
                    'case_number': r.case_number,
                    'case_title': r.case_title,
                    'court':      r.court,
                    'filing_date': r.filing_date,
                    'source':     r.source,
                }
                for r in results
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/court/docket/<court_system>/<path:case_id>', methods=['GET'])
@login_required
def court_get_docket(court_system, case_id):
    """Return the full docket entry list for a case."""
    ok, err = _court_gate()
    if not ok:
        return err
    project_slug = request.args.get('project_slug', 'default')

    try:
        connector = _build_court_connector(court_system, project_slug)
    except RuntimeError as e:
        # v3.9.4: unknown court_system → 400 with clear error (was 500)
        msg = str(e)
        if 'Unknown court system' in msg:
            return jsonify({
                'error': msg,
                'supported': ['federal', 'nyscef'],
            }), 400
        return jsonify({'error': msg}), 500
    try:
        docket = connector.get_docket(case_id)
        return jsonify({
            'case_id': case_id,
            'court_system': court_system,
            'total': len(docket),
            'entries': [
                {
                    'seq':        e.seq,
                    'title':      e.title,
                    'date':       e.date,
                    'source_url': e.source_url,
                    'source':     e.source,
                    'doc_type':   e.doc_type,
                }
                for e in docket
            ],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Import job routes
# ---------------------------------------------------------------------------

