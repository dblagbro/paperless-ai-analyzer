"""CI custom report generation + PDF export.

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

@bp.route('/api/ci/runs/<run_id>/reports', methods=['POST'])
@login_required
@advanced_required
def ci_create_report(run_id):
    """Generate a report for a CI run."""
    from flask import current_app
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, create_ci_report
        from analyzer.case_intelligence.report_generator import ReportGenerator
        import threading as _threading

        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if run['status'] != 'completed':
            return jsonify({'error': 'Run must be completed before generating reports'}), 400

        data = request.json or {}
        instructions = data.get('instructions', 'Generate a comprehensive case summary.')
        template = data.get('template', 'custom')

        report_id = create_ci_report(
            run_id=run_id,
            user_id=current_user.id,
            instructions=instructions,
            template=template,
        )

        # Generate in background thread
        llm_clients = _build_ci_llm_clients()
        generator = ReportGenerator(
            llm_clients=llm_clients,
            usage_tracker=getattr(current_app, 'usage_tracker', None),
        )

        def _generate():
            from analyzer.case_intelligence.db import update_ci_report
            update_ci_report(report_id, content='', status='generating')
            generator.generate(run_id, report_id, instructions, template)

        _threading.Thread(target=_generate, daemon=True,
                           name=f'ci-report-{report_id[:8]}').start()

        return jsonify({'report_id': report_id, 'status': 'generating'}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/reports/<report_id>', methods=['GET'])
@login_required
def ci_get_report(run_id, report_id):
    """Get report content."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_report
        report = get_ci_report(report_id)
        if not report or report['run_id'] != run_id:
            return jsonify({'error': 'Report not found'}), 404
        return jsonify(dict(report))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/reports/<report_id>/pdf', methods=['GET'])
@login_required
def ci_download_report_pdf(run_id, report_id):
    """Download report as PDF."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_report
        from analyzer.case_intelligence.report_generator import ReportGenerator

        report = get_ci_report(report_id)
        if not report or report['run_id'] != run_id:
            return jsonify({'error': 'Report not found'}), 404
        if report['status'] != 'complete' or not report['content']:
            return jsonify({'error': 'Report not yet complete'}), 400

        generator = ReportGenerator(llm_clients={}, usage_tracker=None)
        pdf_bytes = generator.generate_pdf(report['content'], title=f'CI Report {run_id[:8]}')

        if not pdf_bytes:
            # Fallback: return markdown as text file
            response = make_response(report['content'])
            response.headers['Content-Type'] = 'text/markdown; charset=utf-8'
            response.headers['Content-Disposition'] = f'attachment; filename="ci_report_{report_id[:8]}.md"'
            return response

        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="ci_report_{report_id[:8]}.pdf"'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


