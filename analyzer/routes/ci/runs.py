"""CI run CRUD + lifecycle (start/cancel/interrupt/rerun/shares/questions).

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

@bp.route('/api/ci/runs', methods=['GET'])
@login_required
def ci_list_runs():
    """List CI runs for current project.
    Admins see all runs; others see own runs + runs shared with them.
    """
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import (
            list_ci_runs, get_run_ids_shared_with, get_ci_run,
        )
        project_slug = session.get('current_project', 'default')

        if current_user.is_admin:
            runs = [dict(r) for r in list_ci_runs(project_slug)]
        else:
            own = [dict(r) for r in list_ci_runs(project_slug, user_id=current_user.id)]
            shared_ids = get_run_ids_shared_with(current_user.id)
            shared = []
            for rid in shared_ids:
                r = get_ci_run(rid)
                if r and r['project_slug'] == project_slug:
                    d = dict(r)
                    d['_shared'] = True
                    shared.append(d)
            own_ids = {r['id'] for r in own}
            runs = own + [r for r in shared if r['id'] not in own_ids]
            runs.sort(key=lambda r: r.get('created_at', ''), reverse=True)

        # Annotate with owner display_name
        for r in runs:
            u = get_user_by_id(r['user_id'])
            r['owner_name'] = u['display_name'] if u else f"user#{r['user_id']}"

        return jsonify({'runs': runs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs', methods=['POST'])
@login_required
@advanced_required
def ci_create_run():
    """Create a new CI run (draft status)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import json as _json
        from analyzer.case_intelligence.db import create_ci_run
        data = request.json or {}
        project_slug = session.get('current_project', 'default')

        jurisdiction_json = '{}'
        if 'jurisdiction' in data:
            jurisdiction_json = _json.dumps(data['jurisdiction'])

        notification_email = data.get('notification_email', '') or ''
        notify_on_complete = 1 if data.get('notify_on_complete', True) else 0
        notify_on_budget   = 1 if data.get('notify_on_budget',   True) else 0
        # allow_overage_pct: 0=hard block at 100%, 20=allow 20% overage, -1=unlimited
        allow_overage_pct  = int(data.get('allow_overage_pct', 0))

        run_id = create_ci_run(
            project_slug=project_slug,
            user_id=current_user.id,
            role=data.get('role', 'neutral'),
            goal_text=data.get('goal_text', ''),
            budget_per_run_usd=float(data.get('budget_per_run_usd', 10.0)),
            jurisdiction_json=jurisdiction_json,
            objectives=_json.dumps(data.get('objectives', [])),
            max_tier=int(data.get('max_tier', 3)),
            notification_email=notification_email,
            notify_on_complete=notify_on_complete,
            notify_on_budget=notify_on_budget,
            allow_overage_pct=allow_overage_pct,
        )
        # Store web research config if provided
        if 'web_research_config' in data:
            from analyzer.case_intelligence.db import update_ci_run as _ucr
            wrc = data['web_research_config']
            if isinstance(wrc, dict):
                _ucr(run_id, web_research_config=_json.dumps(wrc))

        start_url = f"{request.script_root}/api/ci/runs/{run_id}/start"
        auto_start = bool(data.get('auto_start', False))

        if auto_start:
            from analyzer.case_intelligence.db import update_ci_run
            from analyzer.case_intelligence.job_manager import get_job_manager
            from analyzer.case_intelligence.orchestrator import CIOrchestrator
            import os as _os
            update_ci_run(run_id, status='queued', progress_pct=0,
                          cost_so_far_usd=0, started_at=datetime.utcnow().isoformat())
            llm_clients = _build_ci_llm_clients()
            orchestrator = CIOrchestrator(
                llm_clients=llm_clients,
                paperless_client=getattr(current_app, 'paperless_client', None),
                usage_tracker=getattr(current_app, 'usage_tracker', None),
                cohere_api_key=_os.environ.get('COHERE_API_KEY'),
                budget_notification_cb=_send_ci_budget_notification,
                completion_notification_cb=_send_ci_complete_notification,
            )
            get_job_manager().start_run(run_id, orchestrator.execute_run, run_id)
            return jsonify({'run_id': run_id, 'status': 'queued', 'start_url': start_url}), 201

        return jsonify({'run_id': run_id, 'status': 'draft', 'start_url': start_url}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>', methods=['GET'])
@login_required
def ci_get_run(run_id):
    """Get run config + status."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_read(run):
            return jsonify({'error': 'Not authorized'}), 403
        return jsonify(dict(run))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>', methods=['PUT'])
@login_required
@advanced_required
def ci_update_run(run_id):
    """Update run config (draft only)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import json as _json
        from analyzer.case_intelligence.db import get_ci_run, update_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if run['status'] not in ('draft',):
            return jsonify({'error': 'Can only edit draft runs'}), 400
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        data = request.json or {}
        allowed_fields = {
            'role', 'goal_text', 'budget_per_run_usd', 'max_tier', 'auto_routing',
        }
        kwargs = {k: v for k, v in data.items() if k in allowed_fields}
        if 'jurisdiction' in data:
            kwargs['jurisdiction_json'] = _json.dumps(data['jurisdiction'])
        if 'objectives' in data:
            kwargs['objectives'] = _json.dumps(data['objectives'])
        if 'web_research_config' in data:
            wrc = data['web_research_config']
            kwargs['web_research_config'] = _json.dumps(wrc) if isinstance(wrc, dict) else wrc
        update_ci_run(run_id, **kwargs)
        return jsonify({'status': 'updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/start', methods=['POST'])
@login_required
@advanced_required
def ci_start_run(run_id):
    """Launch a CI run as a background job."""
    from flask import current_app
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import os as _os
        from analyzer.case_intelligence.db import get_ci_run, update_ci_run
        from analyzer.case_intelligence.orchestrator import CIOrchestrator
        from analyzer.case_intelligence.job_manager import get_job_manager
        from analyzer.case_intelligence.db import init_ci_db

        # Ensure DB is initialized
        init_ci_db()

        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        if run['status'] not in ('draft', 'failed', 'cancelled', 'interrupted'):
            return jsonify({'error': f"Cannot start run in status '{run['status']}'"}), 400

        # Reset run for fresh start
        update_ci_run(run_id, status='queued', progress_pct=0,
                      cost_so_far_usd=0, budget_blocked=0,
                      budget_blocked_note=None, error_message=None,
                      started_at=datetime.utcnow().isoformat())

        # Build LLM clients from app config
        llm_clients = _build_ci_llm_clients()

        orchestrator = CIOrchestrator(
            llm_clients=llm_clients,
            paperless_client=getattr(current_app, 'paperless_client', None),
            usage_tracker=getattr(current_app, 'usage_tracker', None),
            cohere_api_key=_os.environ.get('COHERE_API_KEY'),
            budget_notification_cb=_send_ci_budget_notification,
            completion_notification_cb=_send_ci_complete_notification,
        )

        job_manager = get_job_manager()
        started = job_manager.start_run(run_id, orchestrator.execute_run, run_id)

        if not started:
            return jsonify({'error': 'Run is already active'}), 409

        return jsonify({'status': 'started', 'run_id': run_id})
    except Exception as e:
        logger.error(f"CI start_run failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/cancel', methods=['POST'])
@login_required
@advanced_required
def ci_cancel_run(run_id):
    """Send cancellation signal to a running CI job."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run
        from analyzer.case_intelligence.job_manager import get_job_manager

        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        sent = get_job_manager().cancel_run(run_id)
        return jsonify({'cancelled': sent, 'run_id': run_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/interrupt', methods=['POST'])
@login_required
@advanced_required
def ci_interrupt_run(run_id):
    """Interrupt a running CI run (can be restarted). Use /cancel for a hard stop."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, update_ci_run
        from analyzer.case_intelligence.job_manager import get_job_manager

        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        if run['status'] not in ('running', 'queued'):
            return jsonify({'error': f"Cannot interrupt run in status '{run['status']}'"}), 400

        get_job_manager().cancel_run(run_id)  # stops the background thread
        update_ci_run(run_id, status='interrupted')
        return jsonify({'success': True, 'run_id': run_id, 'status': 'interrupted'})
    except Exception as e:
        logger.error(f"CI interrupt_run failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/rerun', methods=['POST'])
@login_required
def ci_rerun(run_id):
    """Create and immediately start a new CI run using the same parameters as run_id."""
    from flask import current_app
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import json as _json
        from analyzer.case_intelligence.db import get_ci_run, create_ci_run, update_ci_run as _ucr
        from analyzer.case_intelligence.job_manager import get_job_manager

        orig = get_ci_run(run_id)
        if not orig:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(orig):
            return jsonify({'error': 'Not authorized'}), 403

        new_id = create_ci_run(
            project_slug=orig['project_slug'],
            user_id=current_user.id,
            role=orig.get('role', 'neutral'),
            goal_text=orig.get('goal_text', ''),
            budget_per_run_usd=float(orig.get('budget_per_run_usd') or 10.0),
            jurisdiction_json=orig.get('jurisdiction_json', '{}'),
            objectives=orig.get('objectives', '[]'),
            max_tier=int(orig.get('max_tier') or orig.get('analysis_tier') or 3),
            notification_email=orig.get('notification_email', ''),
            notify_on_complete=int(orig.get('notify_on_complete', 1)),
            notify_on_budget=int(orig.get('notify_on_budget', 1)),
            allow_overage_pct=int(orig.get('allow_overage_pct', 0)),
        )
        if orig.get('web_research_config'):
            _ucr(new_id, web_research_config=orig['web_research_config'])

        import os as _os
        from analyzer.case_intelligence.db import init_ci_db
        from analyzer.case_intelligence.orchestrator import CIOrchestrator
        init_ci_db()

        # Reset cost/progress fields for a clean restart; skip clarifying questions
        _ucr(new_id,
             proceed_with_assumptions=1,
             assumptions_made=orig.get('assumptions_made', ''),
             status='queued',
             progress_pct=0,
             cost_so_far_usd=0,
             budget_blocked=0,
             budget_blocked_note=None,
             error_message=None,
             started_at=datetime.utcnow().isoformat())

        orch = CIOrchestrator(
            llm_clients=_build_ci_llm_clients(),
            paperless_client=getattr(current_app, 'paperless_client', None),
            usage_tracker=getattr(current_app, 'usage_tracker', None),
            cohere_api_key=_os.environ.get('COHERE_API_KEY'),
            budget_notification_cb=_send_ci_budget_notification,
            completion_notification_cb=_send_ci_complete_notification,
        )
        started = get_job_manager().start_run(new_id, orch.execute_run, new_id)
        if not started:
            return jsonify({'error': 'Could not start rerun (already active?)'}), 409

        return jsonify({'run_id': new_id, 'status': 'queued'})
    except Exception as e:
        logger.error(f"CI rerun failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/status')
@login_required
def ci_run_status(run_id):
    """Live progress polling endpoint (every 3s from UI)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        return jsonify({
            'run_id': run_id,
            'status': run['status'],
            'current_stage': run['current_stage'],
            'progress_pct': run['progress_pct'],
            'cost_so_far_usd': run['cost_so_far_usd'],
            'budget_per_run_usd': run['budget_per_run_usd'],
            'docs_processed': run['docs_processed'],
            'docs_total': run['docs_total'],
            'error_message': run['error_message'],
            'budget_blocked': bool(run['budget_blocked']),
            'budget_blocked_note': run['budget_blocked_note'],
            'tokens_in':       run.get('tokens_in', 0) or 0,
            'tokens_out':      run.get('tokens_out', 0) or 0,
            'active_managers': run.get('active_managers', 0) or 0,
            'active_workers':  run.get('active_workers', 0) or 0,
            'elapsed_seconds': _ci_elapsed_seconds(run),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>', methods=['DELETE'])
@login_required
@advanced_required
def ci_delete_run(run_id):
    """Delete a CI run and all its associated findings."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import sqlite3 as _sq3
        from analyzer.case_intelligence.db import get_ci_run as _get_run
        run = _get_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        with _sq3.connect('/app/data/case_intelligence.db') as conn:
            row = conn.execute('SELECT id, status FROM ci_runs WHERE id=?', (run_id,)).fetchone()
            if not row:
                return jsonify({'error': 'Run not found'}), 404
            if row[1] == 'running':
                return jsonify({'error': 'Cannot delete a run that is currently running. Cancel it first.'}), 409
            # Cascade delete (FK ON DELETE CASCADE covers child tables if enabled,
            # but delete explicitly to be safe)
            for tbl in ('ci_entities', 'ci_timeline_events', 'ci_contradictions',
                        'ci_disputed_facts', 'ci_theory_ledger', 'ci_authorities',
                        'ci_reports', 'ci_manager_reports', 'ci_run_questions'):
                conn.execute(f'DELETE FROM {tbl} WHERE run_id=?', (run_id,))
            conn.execute('DELETE FROM ci_runs WHERE id=?', (run_id,))
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/shares', methods=['GET'])
@login_required
@advanced_required
def ci_get_run_shares(run_id):
    """List users this run is shared with. Owner or admin only."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, list_ci_run_shares
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        shares = list_ci_run_shares(run_id)
        result = []
        for s in shares:
            u = get_user_by_id(s['shared_with'])
            result.append({
                'user_id': s['shared_with'],
                'display_name': u['display_name'] if u else f"user#{s['shared_with']}",
                'username': u['username'] if u else '',
                'shared_by': s['shared_by'],
                'shared_at': s['shared_at'],
            })
        return jsonify({'shares': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/shares', methods=['POST'])
@login_required
@advanced_required
def ci_add_run_share(run_id):
    """Share a run with another user by username or user_id. Owner or admin only."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, add_ci_run_share
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        data = request.json or {}
        target = None
        if 'username' in data:
            target = get_user_by_username(data['username'])
        elif 'user_id' in data:
            target = get_user_by_id(int(data['user_id']))
        if not target:
            return jsonify({'error': 'User not found'}), 404
        if not target['is_active']:
            return jsonify({'error': 'Cannot share with an inactive user'}), 400
        if target['role'] not in ('advanced', 'admin'):
            return jsonify({'error': 'Target user must have Advanced or Admin role to access CI runs'}), 400
        if target['id'] == run['user_id']:
            return jsonify({'error': 'Run already belongs to this user'}), 400

        add_ci_run_share(run_id, shared_with=target['id'], shared_by=current_user.id)
        return jsonify({'ok': True, 'shared_with': target['display_name']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/shares/<int:uid>', methods=['DELETE'])
@login_required
@advanced_required
def ci_remove_run_share(run_id, uid):
    """Remove a user's access to a shared run. Owner or admin only."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, remove_ci_run_share
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403
        remove_ci_run_share(run_id, uid)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/questions', methods=['GET'])
@login_required
def ci_get_questions(run_id):
    """Get clarifying questions for a run."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_questions
        questions = get_ci_questions(run_id)
        return jsonify({'questions': [dict(q) for q in questions]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/runs/<run_id>/answers', methods=['POST'])
@login_required
@advanced_required
def ci_submit_answers(run_id):
    """Submit answers to clarifying questions."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_ci_run, answer_ci_question, update_ci_run
        run = get_ci_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        if not _ci_can_write(run):
            return jsonify({'error': 'Not authorized'}), 403

        data = request.json or {}
        answers = data.get('answers', {})  # {question_id: answer_text}
        proceed_with_assumptions = data.get('proceed_with_assumptions', False)

        for qid_str, answer_text in answers.items():
            try:
                answer_ci_question(int(qid_str), answer_text)
            except Exception as e:
                logger.warning(f"CI answer question {qid_str} failed: {e}")

        if proceed_with_assumptions:
            update_ci_run(run_id, proceed_with_assumptions=1)

        return jsonify({'status': 'answers_saved'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


