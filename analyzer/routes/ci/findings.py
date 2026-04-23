"""CI findings + tier-specific report views.

Extracted from routes/ci.py during the v3.9.3 maintainability refactor.
"""
import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, session, make_response
from flask_login import login_required, current_user

from analyzer.app import admin_required, advanced_required, _ci_gate, _ci_can_read, _ci_can_write
from analyzer.db import get_user_by_id, get_user_by_username
from analyzer.services.ai_config_service import load_ai_config, get_project_ai_config
from analyzer.services.smtp_service import (
    load_smtp_settings as _load_smtp_settings,
    smtp_send as _smtp_send,
)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# CI Notification helpers
# ---------------------------------------------------------------------------

from analyzer.routes.ci import bp
from analyzer.routes.ci.helpers import (
    _send_ci_budget_notification,
    _send_ci_complete_notification,
    _match_jurisdiction_profile,
    _ci_elapsed_seconds,
    _build_ci_llm_clients,
)

@bp.route('/api/ci/runs/<run_id>/findings')
@login_required
def ci_run_findings(run_id):
    """Full findings: entities, timeline, contradictions, theories, authorities."""
    from flask import current_app
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import (
            get_ci_run, get_ci_entities_active as get_ci_entities, get_ci_timeline,
            get_ci_contradictions, get_ci_theories, get_ci_authorities,
            get_ci_disputed_facts,
        )
        import json as _json
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Not authorized'}), 403

        findings_summary = None
        if run['findings_summary']:
            try:
                findings_summary = _json.loads(run['findings_summary'])
            except Exception:
                findings_summary = {'raw': run['findings_summary']}

        # ── Run metadata ──────────────────────────────────────────────────────
        # Compute duration
        duration_str = None
        try:
            if run.get('started_at') and run.get('completed_at'):
                from datetime import datetime as _dt
                fmt = '%Y-%m-%dT%H:%M:%S'
                t0 = _dt.fromisoformat(run['started_at'].split('.')[0].replace('Z', ''))
                t1 = _dt.fromisoformat(run['completed_at'].split('.')[0].replace('Z', ''))
                secs = int((t1 - t0).total_seconds())
                if secs >= 3600:
                    duration_str = f"{secs//3600}h {(secs%3600)//60}m"
                elif secs >= 60:
                    duration_str = f"{secs//60}m {secs%60}s"
                else:
                    duration_str = f"{secs}s"
        except Exception:
            pass

        # Look up user display name
        run_user = None
        try:
            import sqlite3 as _sq3
            with _sq3.connect('/app/data/app.db') as _uc:
                _uc.row_factory = _sq3.Row
                _ur = _uc.execute(
                    'SELECT display_name, username FROM users WHERE id=?',
                    (run.get('user_id'),)
                ).fetchone()
                if _ur:
                    run_user = _ur['display_name'] or _ur['username']
        except Exception:
            pass

        run_meta = {
            'created_at':      run.get('created_at'),
            'completed_at':    run.get('completed_at'),
            'duration':        duration_str,
            'run_by':          run_user,
            'role':            run.get('role'),
            'goal_text':       run.get('goal_text'),
            'project_slug':    run.get('project_slug'),
            'docs_total':      run.get('docs_total'),
            'docs_processed':  run.get('docs_processed'),
            'cost_usd':        run.get('cost_so_far_usd'),
            'budget_usd':      run.get('budget_per_run_usd'),
            'status':          run.get('status'),
            'progress_pct':    run.get('progress_pct'),
        }

        # ── Build doc_map from theory evidence fields ─────────────────────────
        theories_raw = [dict(t) for t in get_ci_theories(run_id)]
        doc_ids_needed = set()
        for t in theories_raw:
            for field in ('supporting_evidence', 'counter_evidence'):
                val = t.get(field)
                if not val:
                    continue
                try:
                    items = _json.loads(val) if isinstance(val, str) else val
                    for item in (items or []):
                        did = item.get('paperless_doc_id')
                        if did:
                            doc_ids_needed.add(int(did))
                except Exception:
                    pass

        doc_map = {}
        if doc_ids_needed and hasattr(current_app, 'paperless_client'):
            for did in doc_ids_needed:
                try:
                    doc = current_app.paperless_client.get_document(did)
                    if doc:
                        doc_map[did] = {
                            'id':      did,
                            'title':   doc.get('title', f'Document {did}'),
                            'summary': (doc.get('content') or '')[:300].strip(),
                        }
                except Exception:
                    doc_map[did] = {'id': did, 'title': f'Document {did}', 'summary': ''}

        # Fetch web research results
        from analyzer.case_intelligence.db import get_ci_web_research as _gcwr
        web_research_raw = _gcwr(run_id)

        return jsonify({
            'run_id':           run_id,
            'status':           run['status'],
            'run_meta':         run_meta,
            'doc_map':          doc_map,
            'findings_summary': findings_summary,
            'entities':         [dict(e) for e in get_ci_entities(run_id)],
            'timeline':         [dict(ev) for ev in get_ci_timeline(run_id)],
            'contradictions':   [dict(c) for c in get_ci_contradictions(run_id)],
            'disputed_facts':   [dict(f) for f in get_ci_disputed_facts(run_id)],
            'theories':         theories_raw,
            'authorities':      [dict(a) for a in get_ci_authorities(run_id)],
            'web_research':     web_research_raw,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/forensic-report')
@login_required
def ci_forensic_report(run_id):
    """Return forensic accounting report for a CI run (Tier 3+)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_forensic_report
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        report = get_forensic_report(run_id)
        if not report:
            return jsonify({'present': False, 'data': None})
        # Parse JSON fields
        for field in ('flagged_transactions', 'cash_flow_by_party', 'balance_discrepancies',
                      'missing_transactions', 'transaction_chains'):
            try:
                report[field] = json.loads(report.get(field) or '[]')
            except Exception:
                report[field] = []
        return jsonify({'present': True, 'data': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/discovery-gaps')
@login_required
def ci_discovery_gaps(run_id):
    """Return discovery gap analysis for a CI run (Tier 3+)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_discovery_gaps
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        gaps = get_discovery_gaps(run_id)
        if not gaps:
            return jsonify({'present': False, 'data': None})
        for field in ('missing_doc_types', 'custodian_gaps', 'spoliation_indicators',
                      'rfp_list', 'subpoena_targets'):
            try:
                gaps[field] = json.loads(gaps.get(field) or '[]')
            except Exception:
                gaps[field] = []
        return jsonify({'present': True, 'data': gaps})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/witness-cards')
@login_required
def ci_witness_cards(run_id):
    """Return witness intelligence cards for a CI run (Tier 4+)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_witness_cards
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        cards = get_witness_cards(run_id)
        # Parse JSON fields
        parsed = []
        for card in cards:
            for field in ('impeachment_points', 'prior_inconsistencies',
                          'public_record_flags', 'key_questions'):
                try:
                    card[field] = json.loads(card.get(field) or '[]')
                except Exception:
                    card[field] = []
            try:
                card['financial_interest'] = json.loads(card.get('financial_interest') or '{}')
            except Exception:
                card['financial_interest'] = {}
            parsed.append(card)
        return jsonify({'present': len(parsed) > 0, 'data': parsed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/war-room')
@login_required
def ci_war_room_report(run_id):
    """Return war room analysis for a CI run (Tier 4+)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_war_room
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        wr = get_war_room(run_id)
        if not wr:
            return jsonify({'present': False, 'data': None})
        for field in ('top_dangerous_arguments', 'client_vulnerabilities',
                      'smoking_guns', 'opposing_counsel_checklist'):
            try:
                wr[field] = json.loads(wr.get(field) or '[]')
            except Exception:
                wr[field] = []
        try:
            wr['settlement_analysis'] = json.loads(wr.get('settlement_analysis') or '{}')
        except Exception:
            wr['settlement_analysis'] = {}
        return jsonify({'present': True, 'data': wr})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/deep-forensics')
@login_required
def ci_deep_forensics_report(run_id):
    """Return deep financial forensics report for a CI run (Tier 5)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_deep_forensics
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        report = get_deep_forensics(run_id)
        if not report:
            return jsonify({'present': False, 'data': None})
        for field in ('beneficial_ownership', 'round_trip_transactions', 'shell_entity_flags',
                      'advanced_structuring', 'layering_schemes', 'suspicious_clusters'):
            try:
                report[field] = json.loads(report.get(field) or '[]')
            except Exception:
                report[field] = []
        try:
            report['benford_analysis'] = json.loads(report.get('benford_analysis') or '{}')
        except Exception:
            report['benford_analysis'] = {}
        return jsonify({'present': True, 'data': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/trial-strategy')
@login_required
def ci_trial_strategy_report(run_id):
    """Return trial strategy memo for a CI run (Tier 5)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_trial_strategy
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        strategy = get_trial_strategy(run_id)
        if not strategy:
            return jsonify({'present': False, 'data': None})
        for field in ('witness_order', 'key_exhibits', 'motions_in_limine',
                      'closing_themes', 'trial_risks'):
            try:
                strategy[field] = json.loads(strategy.get(field) or '[]')
            except Exception:
                strategy[field] = []
        try:
            strategy['jury_profile'] = json.loads(strategy.get('jury_profile') or '{}')
        except Exception:
            strategy['jury_profile'] = {}
        return jsonify({'present': True, 'data': strategy})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/multi-model')
@login_required
def ci_multi_model_report(run_id):
    """Return multi-model comparison for a CI run (Tier 5)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_multi_model_comparison
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        mm = get_multi_model_comparison(run_id)
        if not mm:
            return jsonify({'present': False, 'data': None})
        for field in ('agreed_theories', 'model_a_only', 'model_b_only', 'disagreements'):
            try:
                mm[field] = json.loads(mm.get(field) or '[]')
            except Exception:
                mm[field] = []
        for field in ('anthropic_analysis', 'openai_analysis'):
            try:
                mm[field] = json.loads(mm.get(field) or '{}')
            except Exception:
                mm[field] = {}
        return jsonify({'present': True, 'data': mm})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/settlement-valuation')
@login_required
def ci_settlement_valuation_report(run_id):
    """Return settlement valuation analysis for a CI run (Tier 5)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, get_settlement_valuation
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Access denied'}), 403
        sv = get_settlement_valuation(run_id)
        if not sv:
            return jsonify({'present': False, 'data': None})
        for field in ('damages_breakdown', 'insurance_flags', 'leverage_timeline'):
            try:
                sv[field] = json.loads(sv.get(field) or '[]')
            except Exception:
                sv[field] = []
        for field in ('total_exposure', 'litigation_cost_model', 'fee_shifting_risk',
                      'settlement_recommendation', 'mediation_strategy'):
            try:
                sv[field] = json.loads(sv.get(field) or '{}')
            except Exception:
                sv[field] = {}
        return jsonify({'present': True, 'data': sv})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


