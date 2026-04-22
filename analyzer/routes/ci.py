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

bp = Blueprint('ci', __name__)


# ---------------------------------------------------------------------------
# CI Notification helpers
# ---------------------------------------------------------------------------

def _send_ci_budget_notification(run_id: str, pct_complete: float,
                                  cost_so_far: float, projected_total: float,
                                  budget: float, status: str, is_urgent: bool = False):
    """Send a budget checkpoint email for a CI run."""
    try:
        from email.message import EmailMessage
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return
        email = run['notification_email'] if 'notification_email' in run.keys() else ''
        if not email:
            return
        notify_on_budget = run['notify_on_budget'] if 'notify_on_budget' in run.keys() else 1
        if not notify_on_budget:
            return

        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info("SMTP not configured — skipping CI budget notification")
            return

        goal_text = run['goal_text'] if 'goal_text' in run.keys() else 'Unknown Case'
        allow_overage_pct = run.get('allow_overage_pct') or 0
        status_label = {'under_budget': 'Under Budget', 'on_track': 'On Track',
                        'over_budget': 'OVER BUDGET', 'blocked': 'BUDGET BLOCKED'
                        }.get(status, status)
        pct_int = int(round(pct_complete))

        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'

        if is_urgent or status == 'blocked':
            subject = (f"URGENT: CI Budget {pct_int}% — {goal_text[:40]} — {status_label}")
        else:
            subject = f"CI Budget Update — {goal_text[:40]} — {pct_int}% complete — {status_label}"

        overage_line = ''
        if allow_overage_pct == -1:
            overage_line = 'Overage Policy: Unlimited (budget is a goal only — run will not be blocked)\n'
        elif allow_overage_pct > 0:
            hard_limit = budget * (1 + allow_overage_pct / 100)
            overage_line = f'Overage Policy: Up to {allow_overage_pct}% allowed (hard limit: ${hard_limit:.2f})\n'

        body = (
            f"{'URGENT — ' if is_urgent or status == 'blocked' else ''}Case Intelligence Budget {'ALERT' if is_urgent or status == 'blocked' else 'Update'}\n"
            f"{'=' * 50}\n\n"
            f"Case:        {goal_text}\n"
            f"Run ID:      {run_id}\n"
            f"Progress:    {pct_int}% complete\n"
            f"Status:      {status_label}\n"
            f"{overage_line}\n"
            f"Cost So Far: ${cost_so_far:.4f}\n"
            f"Projected:   ${projected_total:.4f}\n"
            f"Budget:      ${budget:.4f}\n"
        )
        if status == 'blocked':
            body += "\nThe run has been STOPPED. Budget ceiling reached.\n"
        elif status == 'over_budget':
            body += "\nWARNING: Projected cost exceeds budget.\n"
        if is_urgent and status != 'blocked':
            body += "\nApproaching budget limit — review and consider adjusting budget or stopping the run.\n"

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"CI budget notification sent to {email} ({pct_int}%, {status}, urgent={is_urgent})")
    except Exception as e:
        logger.warning(f"Failed to send CI budget notification: {e}")


def _send_ci_complete_notification(run_id: str):
    """Send run-complete email for a CI run."""
    try:
        from email.message import EmailMessage
        from analyzer.case_intelligence.db import get_ci_run
        run = get_ci_run(run_id)
        if not run:
            return
        email = run['notification_email'] if 'notification_email' in run.keys() else ''
        if not email:
            return
        notify_on_complete = run['notify_on_complete'] if 'notify_on_complete' in run.keys() else 1
        if not notify_on_complete:
            return

        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info("SMTP not configured — skipping CI complete notification")
            return

        goal_text = run['goal_text'] if 'goal_text' in run.keys() else 'Unknown Case'
        cost = run['cost_so_far_usd'] or 0
        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        body = (
            f"Case Intelligence Run Complete\n"
            f"{'=' * 50}\n\n"
            f"Case:       {goal_text}\n"
            f"Run ID:     {run_id}\n"
            f"Total Cost: ${cost:.4f}\n\n"
            f"The analysis report is ready. Log in to view or download it.\n"
        )
        msg = EmailMessage()
        msg['Subject'] = f"CI Complete — {goal_text[:50]}"
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"CI complete notification sent to {email} for run {run_id}")
    except Exception as e:
        logger.warning(f"Failed to send CI complete notification: {e}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _match_jurisdiction_profile(extracted: dict) -> str:
    """Map LLM-extracted jurisdiction details to a JURISDICTION_PROFILES key."""
    court = (extracted.get('court_name') or '').lower()
    is_bk = extracted.get('is_bankruptcy', False) or 'bankruptcy' in court
    bk_ch = extracted.get('bankruptcy_chapter')
    is_fam = extracted.get('is_family_court', False) or 'family' in court
    is_sur = extracted.get('is_surrogate', False) or 'surrogate' in court
    is_fed = extracted.get('is_federal', False)
    is_commercial = 'commercial' in court

    if is_bk:
        if 'eastern' in court or 'edny' in court or 'e.d.n.y' in court:
            return 'edny-bankruptcy'
        ch = int(bk_ch) if bk_ch and str(bk_ch).isdigit() else 7
        return f'sdny-bankruptcy-ch{ch}' if ch in (7, 11) else 'sdny-bankruptcy-ch7'
    if is_fam:
        return 'nys-family-court'
    if is_sur:
        return 'nys-surrogate'
    if 'appellate' in court:
        return 'nys-appellate-div-1' if ('first' in court or '1st' in court) else 'nys-appellate-div-2'
    if is_fed or 'district' in court or 'sdny' in court or 's.d.n.y' in court:
        if 'eastern' in court or 'edny' in court or 'e.d.n.y' in court:
            return 'edny-civil'
        return 'sdny-civil'
    if is_commercial:
        return 'nys-commercial-division'
    # Default: NYS Supreme Court (most common for NY cases)
    state = (extracted.get('state') or 'NY').upper()
    if state == 'NY' or 'supreme' in court:
        return 'nys-supreme-civil'
    return 'custom'


def _ci_elapsed_seconds(run):
    """Return seconds since run started_at, or 0."""
    if not run.get('started_at'):
        return 0
    try:
        from datetime import datetime as _dt, timezone
        start = _dt.fromisoformat(run['started_at'].replace('Z', '+00:00'))
        return int((_dt.now(timezone.utc) - start).total_seconds())
    except Exception:
        return 0


def _build_ci_llm_clients() -> dict:
    """
    Build LLM client dict for CI components.
    Returns {'openai': client_or_None, 'anthropic': client_or_None}
    """
    from flask import current_app
    import os as _os
    clients = {}

    # Get usage tracker from document_analyzer if available
    _usage_tracker = None
    if hasattr(current_app, 'document_analyzer') and hasattr(current_app.document_analyzer, 'usage_tracker'):
        _usage_tracker = current_app.document_analyzer.usage_tracker

    # Check provider from env
    lm_provider = _os.environ.get('LLM_PROVIDER', 'anthropic').lower()

    # LLM_API_KEY belongs to the configured provider only — never cross-assign
    _generic_key = _os.environ.get('LLM_API_KEY', '')
    openai_key = _os.environ.get('OPENAI_API_KEY') or (
        _generic_key if lm_provider == 'openai' else ''
    )
    anthropic_key = _os.environ.get('ANTHROPIC_API_KEY') or (
        _generic_key if lm_provider == 'anthropic' else ''
    )

    # Try to reuse the existing app llm_client
    existing_client = getattr(current_app, 'llm_client', None)
    if existing_client:
        if lm_provider == 'openai':
            clients['openai'] = existing_client
            # Also build an anthropic client if key exists
            if anthropic_key and anthropic_key != openai_key:
                try:
                    from analyzer.llm.llm_client import LLMClient
                    clients['anthropic'] = LLMClient(
                        provider='anthropic',
                        api_key=anthropic_key,
                        usage_tracker=_usage_tracker,
                    )
                except Exception:
                    pass
        else:
            clients['anthropic'] = existing_client
            # Also build an openai client if key exists
            if openai_key:
                try:
                    from analyzer.llm.llm_client import LLMClient
                    clients['openai'] = LLMClient(
                        provider='openai',
                        api_key=openai_key,
                        usage_tracker=_usage_tracker,
                    )
                except Exception:
                    pass

    return clients


# ---------------------------------------------------------------------------
# CI Routes
# ---------------------------------------------------------------------------

@bp.route('/api/ci/status')
@login_required
def ci_status():
    """Feature status + authority corpus stats."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_authority_corpus_stats
        from analyzer.case_intelligence.job_manager import get_job_manager
        corpus_stats = get_authority_corpus_stats()
        active_runs = get_job_manager().list_active_runs()
        return jsonify({
            'enabled': True,
            'authority_corpus': corpus_stats,
            'active_runs': len(active_runs),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/jurisdictions')
@login_required
def ci_jurisdictions():
    """List pre-built jurisdiction profiles."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.jurisdiction import list_jurisdiction_profiles
        return jsonify({'jurisdictions': list_jurisdiction_profiles()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/detect-jurisdiction', methods=['POST'])
@login_required
def ci_detect_jurisdiction():
    """Detect court jurisdiction from project documents using LLM."""
    from flask import current_app
    ok, err = _ci_gate()
    if not ok:
        return err
    import json as _json
    try:
        project_slug = session.get('current_project', 'default')
        from analyzer.vector_store import VectorStore
        from analyzer.case_intelligence.jurisdiction import JURISDICTION_PROFILES

        vs = VectorStore(project_slug=project_slug)
        if not vs.enabled:
            return jsonify({'detected': False, 'reason': 'Vector store not enabled for this project'})

        # Search for jurisdiction-relevant document excerpts
        results = vs.search(
            "court supreme district county filed plaintiff defendant address jurisdiction caption index number",
            n_results=8
        )
        if not results:
            return jsonify({'detected': False, 'reason': 'No documents found in project'})

        # Build document excerpts (first 500 chars each)
        excerpts = []
        for r in results[:6]:
            title = r.get('title', 'Untitled')
            content = (r.get('content') or r.get('document') or '')[:500].strip()
            if content:
                excerpts.append(f"[{title}]\n{content}")
        if not excerpts:
            return jsonify({'detected': False, 'reason': 'Documents found but content unavailable'})
        excerpts_text = "\n\n---\n\n".join(excerpts)

        # Use the app's existing LLM client
        llm = getattr(current_app, 'llm_client', None)
        if not llm or not llm.client:
            return jsonify({'detected': False, 'reason': 'LLM client not available'})

        prompt = (
            "You are analyzing legal documents to identify their court jurisdiction.\n\n"
            "Examine these document excerpts and identify:\n"
            "- The specific court (e.g. 'NYS Supreme Court', 'S.D.N.Y.', 'E.D.N.Y. Bankruptcy Court')\n"
            "- The county (for NY state cases, e.g. 'Kings', 'New York', 'Queens')\n"
            "- The state (two-letter code, e.g. 'NY')\n"
            "- Whether it is a federal case (SDNY, EDNY, etc.)\n"
            "- Whether it is a bankruptcy case, and if so, chapter number (7 or 11)\n"
            "- Whether it is a Family Court or Surrogate's Court case\n"
            "- Addresses of parties (to confirm location)\n\n"
            "Document excerpts:\n"
            f"{excerpts_text[:3500]}\n\n"
            "Respond with ONLY valid JSON, no other text:\n"
            '{"court_name": "string or null", "county": "string or null", '
            '"state": "NY", "is_federal": false, "is_bankruptcy": false, '
            '"bankruptcy_chapter": null, "is_family_court": false, '
            '"is_surrogate": false, "confidence": 0.85, '
            '"reasoning": "one sentence explanation"}'
        )

        if llm.provider == 'openai':
            resp = llm.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_tokens=350,
                response_format={'type': 'json_object'},
            )
            raw = resp.choices[0].message.content
        else:
            resp = llm.client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=350,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw = resp.content[0].text

        if not raw or not raw.strip():
            return jsonify({'detected': False, 'reason': 'LLM returned an empty response'})

        # Strip markdown code fences if the model wrapped the JSON
        raw_stripped = raw.strip()
        if raw_stripped.startswith('```'):
            raw_stripped = raw_stripped.split('\n', 1)[-1]
            raw_stripped = raw_stripped.rsplit('```', 1)[0].strip()

        extracted = _json.loads(raw_stripped)
        profile_id = _match_jurisdiction_profile(extracted)
        profile = JURISDICTION_PROFILES.get(profile_id)

        return jsonify({
            'detected': True,
            'jurisdiction_id': profile_id,
            'display_name': profile.display_name if profile else 'Custom',
            'court': profile.court if profile else (extracted.get('court_name') or ''),
            'county': extracted.get('county'),
            'confidence': extracted.get('confidence', 0.5),
            'reason': extracted.get('reasoning', ''),
        })

    except Exception as e:
        logger.error(f"CI detect-jurisdiction error: {e}", exc_info=True)
        return jsonify({'detected': False, 'reason': f'Detection failed: {str(e)}'})


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


@bp.route('/api/ci/goal-assistant', methods=['POST'])
@login_required
@advanced_required
def ci_goal_assistant():
    """Lightweight AI chat that helps the user write a focused CI goal statement.
    Stateless — caller maintains conversation history in the browser.
    """
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.routes.chat import _fetch_url_text
        data = request.json or {}
        messages = data.get('messages', [])          # [{role, content}, ...]
        ctx = data.get('context', {})
        role = ctx.get('role', 'neutral')
        jurisdiction = ctx.get('jurisdiction', 'Not specified')
        draft_goal = ctx.get('draft_goal', '').strip()

        # Fetch any URLs the user included in their messages so the AI can reference them
        import re as _re_goal
        _goal_url_pat = _re_goal.compile(r'https?://[^\s<>"\']+', _re_goal.IGNORECASE)
        _goal_url_context = ''
        _seen_urls = set()
        for _msg in messages:
            for _u in _goal_url_pat.findall(_msg.get('content', ''))[:2]:
                if _u not in _seen_urls:
                    _seen_urls.add(_u)
                    _txt, _err = _fetch_url_text(_u, max_chars=3000)
                    if _txt:
                        _goal_url_context += f"\n\n--- Content from {_u} ---\n{_txt}"
                    else:
                        _goal_url_context += f"\n\n[Could not fetch {_u}: {_err}]"

        system_prompt = (
            "You are a legal case strategy advisor helping an attorney write a clear, focused "
            "goal statement for a document-analysis AI called Case Intelligence.\n\n"
            "Case Intelligence will analyze legal documents to:\n"
            "- Extract people, organizations, accounts, and key properties\n"
            "- Build a chronological timeline of events\n"
            "- Detect financial flows and amounts\n"
            "- Find contradictions between documents\n"
            "- Generate factual and legal theories\n"
            "- Identify relevant legal authorities\n\n"
            "A great goal statement tells Case Intelligence EXACTLY what to find:\n"
            "  - Who the client represents and the core legal dispute\n"
            "  - What specific evidence or patterns to surface\n"
            "  - What outcome the attorney needs (support damages, build defense, find smoking-gun)\n\n"
            f"CURRENT SETUP:\n"
            f"  Role: {role}\n"
            f"  Jurisdiction: {jurisdiction}\n"
        )
        if draft_goal:
            system_prompt += f"  Draft goal: \"{draft_goal}\"\n"
        if _goal_url_context:
            system_prompt += f"\nWEB CONTENT PROVIDED BY USER:{_goal_url_context}\nUse the above web content when relevant to help craft the goal statement.\n"
        system_prompt += (
            "\nINSTRUCTIONS:\n"
            "1. On the first turn, briefly acknowledge the context and ask 2-3 targeted "
            "questions to understand the case better. Be concise.\n"
            "2. After you have enough information, produce a polished goal statement.\n"
            "3. When you are ready to suggest a goal, end your message with exactly:\n"
            "Suggested Goal:\n[goal text — 2-4 sentences, specific and actionable]\n"
            "4. Only produce ONE suggested goal block per message. Keep the rest of your "
            "message brief.\n"
            "5. If the user wants changes, revise the suggested goal in the same format."
        )

        project_slug = session.get('current_project', 'default')
        chat_cfg = get_project_ai_config(project_slug, 'chat')
        _full_cfg = load_ai_config()

        def _global_key(p):
            return _full_cfg.get('global', {}).get(p, {}).get('api_key', '').strip()

        provider = chat_cfg.get('provider', 'openai')
        api_key = (chat_cfg.get('api_key') or '').strip() or _global_key(provider)
        model = chat_cfg.get('model', 'gpt-4o')

        # Fallback if primary has no key
        if not api_key:
            fb_prov = chat_cfg.get('fallback_provider')
            api_key = _global_key(fb_prov) if fb_prov else ''
            if api_key:
                provider = fb_prov
                model = chat_cfg.get('fallback_model', model)

        if not api_key:
            return jsonify({'error': 'No AI API key configured.'}), 503

        if provider == 'openai':
            import openai as _oai
            client = _oai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=600,
                messages=[{'role': 'system', 'content': system_prompt}] + messages,
            )
            reply = resp.choices[0].message.content
        elif provider == 'anthropic':
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            # Anthropic requires at least one message; on the first greeting call
            # the browser sends an empty list, so inject a synthetic opener.
            ant_messages = messages if messages else [
                {'role': 'user', 'content': 'Hello, I need help writing a focused goal statement for my Case Intelligence analysis.'}
            ]
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=system_prompt,
                messages=ant_messages,
            )
            reply = resp.content[0].text
        else:
            return jsonify({'error': f'Unsupported provider: {provider}'}), 400

        # Extract suggested goal if present
        suggested_goal = None
        marker = 'Suggested Goal:'
        if marker in reply:
            suggested_goal = reply.split(marker, 1)[1].strip()

        return jsonify({'response': reply, 'suggested_goal': suggested_goal})
    except Exception as e:
        logger.error(f"CI goal assistant error: {e}")
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


@bp.route('/api/ci/authority/ingest', methods=['POST'])
@login_required
@admin_required
def ci_ingest_authorities():
    """Trigger authority corpus ingestion (admin only)."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        import os as _os
        import threading as _threading
        from analyzer.case_intelligence.authority_ingester import AuthorityIngester
        from analyzer.case_intelligence.authority_retriever import AuthorityRetriever
        from analyzer.case_intelligence.db import init_ci_db

        init_ci_db()
        data = request.json or {}
        sources = data.get('sources', ['nysenate', 'ecfr', 'courtlistener'])

        def _ingest():
            ingester = AuthorityIngester(
                courtlistener_token=_os.environ.get('COURTLISTENER_API_TOKEN'),
                nysenate_token=_os.environ.get('NYSENATE_API_TOKEN'),
            )
            results = ingester.ingest_all(sources=sources)
            logger.info(f"Authority ingestion complete: {results}")

            # Embed newly added authorities
            retriever = AuthorityRetriever(
                cohere_api_key=_os.environ.get('COHERE_API_KEY'),
            )
            if retriever.enabled:
                embedded = retriever.embed_pending_authorities(batch_size=200)
                logger.info(f"Embedded {embedded} new authorities")

        _threading.Thread(target=_ingest, daemon=True, name='ci-authority-ingest').start()
        return jsonify({'status': 'ingestion_started', 'sources': sources})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/authority/status')
@login_required
def ci_authority_status():
    """Authority corpus statistics."""
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        from analyzer.case_intelligence.db import get_authority_corpus_stats
        from analyzer.case_intelligence.authority_retriever import AuthorityRetriever
        import os as _os

        db_stats = get_authority_corpus_stats()
        retriever = AuthorityRetriever(cohere_api_key=_os.environ.get('COHERE_API_KEY'))
        chroma_stats = retriever.get_corpus_stats()
        return jsonify({
            'db': db_stats,
            'chroma': chroma_stats,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/ci/key-guide', methods=['POST'])
@login_required
def ci_key_guide():
    """Conversational AI agent to help users obtain and activate API keys for CI web research services."""
    import json as _json

    CREDENTIALS_MAP = {
        'brave':         {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'gcse':          {'fields': {'key': 'API Key', 'cx': 'Custom Search Engine ID (CX)'}, 'creds_desc': 'two credentials: an API Key (from https://console.cloud.google.com/apis/credentials) and a Custom Search Engine ID / CX (from https://programmablesearchengine.google.com/controlpanel/all → your engine → Overview page)'},
        'exa':           {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'perplexity':    {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'tavily':        {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'serper':        {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'fec':           {'fields': {'key': 'API key'}, 'creds_desc': 'one API key (email signup at api.data.gov)'},
        'opensanctions': {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'opencorp':      {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'docket':        {'fields': {'user': 'username', 'pass': 'password'}, 'creds_desc': 'username and password'},
        'unicourt':      {'fields': {'client_id': 'Client ID', 'client_secret': 'Client Secret'}, 'creds_desc': 'OAuth2 Client ID and Client Secret'},
        'newsapi':       {'fields': {'key': 'API key'}, 'creds_desc': 'one API key'},
        'courtlistener': {'fields': {'key': 'API token'}, 'creds_desc': 'an API token (or leave blank for unauthenticated access)'},
    }

    # Service-specific setup notes injected into system prompt
    SETUP_NOTES = {
        'gcse': (
            "GOOGLE CUSTOM SEARCH — CRITICAL SETUP NOTES:\n"
            "1. The user needs TWO things: an API Key AND a Search Engine ID (CX).\n"
            "2. API Key: go to https://console.cloud.google.com/apis/credentials → Create Credentials → API Key. "
            "The Programmable Search JSON API must be enabled at https://console.cloud.google.com/apis/library/customsearch.googleapis.com\n"
            "3. Search Engine ID (CX): go to https://programmablesearchengine.google.com/controlpanel/all → "
            "create or open an engine → the CX/Search engine ID is shown on the Overview page.\n"
            "4. TO SEARCH THE ENTIRE WEB: In the engine control panel, open your engine → Edit → "
            "'Basics' or 'Setup' tab. Look for a toggle or radio button labelled 'Search the entire web' "
            "or 'Search the entire web but emphasize included sites'. This is a section-level toggle SEPARATE "
            "from the 'Sites to search' list — do NOT try to add * or any wildcard as a site pattern, that will fail. "
            "IMPORTANT: As of 2025 Google has been deprecating the whole-web PSE feature. "
            "If the toggle is absent or greyed out, the PSE engine is SITE-RESTRICTED ONLY. "
            "In that case, honest advice is: use Brave Search API (https://brave.com/search/api/) or "
            "Serper.dev (https://serper.dev/) instead — both search the full web without domain restrictions, "
            "are much simpler to set up, and have free tiers.\n"
            "5. Walk the user through BOTH credentials one at a time. Ask for the API Key first, then the CX.\n"
            "6. Always use full clickable URLs when directing the user somewhere."
        ),
    }

    SERVICE_URLS = {
        'brave':         'https://brave.com/search/api/',
        'gcse':          'https://programmablesearchengine.google.com/controlpanel/create',
        'exa':           'https://dashboard.exa.ai/',
        'perplexity':    'https://www.perplexity.ai/settings/api',
        'tavily':        'https://app.tavily.com/home',
        'serper':        'https://serper.dev/',
        'fec':           'https://api.data.gov/signup/',
        'opensanctions': 'https://www.opensanctions.org/api/',
        'opencorp':      'https://opencorporates.com/api_accounts/new',
        'docket':        'https://www.docketalarm.com/accounts/register/',
        'unicourt':      'https://unicourt.com/api',
        'newsapi':       'https://newsapi.org/register',
        'courtlistener': 'https://www.courtlistener.com/sign-in/',
    }

    try:
        data = request.json or {}
        service = (data.get('service') or '').strip()
        service_name = (data.get('service_name') or service).strip()
        messages_in = data.get('messages') or []
        if not service:
            return jsonify({'error': 'service required'}), 400

        svc_info = CREDENTIALS_MAP.get(service)
        creds_desc = svc_info['creds_desc'] if svc_info else 'the required credentials'
        fields = svc_info['fields'] if svc_info else {'key': 'API key'}
        url = SERVICE_URLS.get(service, '')

        # Build json_fields_example for CREDENTIALS_READY signal
        json_fields_example = _json.dumps({k: f'<{v}>' for k, v in fields.items()})

        extra_notes = SETUP_NOTES.get(service, '')
        system_prompt = (
            f"You are an AI assistant embedded in a legal document analysis platform. "
            f"Your sole job right now is to help the user obtain and activate their {service_name} API key.\n\n"
            f"RULES:\n"
            f"- Be conversational and direct. Maximum 3 sentences per reply.\n"
            f"- Ask ONE question or give ONE instruction at a time — never dump a full list.\n"
            f"- Guide them step by step: first ask if they have an account, then walk through account creation or key generation as needed.\n"
            f"- Always provide full URLs as plain text so the user can click them.\n"
            f"- Registration page: {url}\n"
            f"- This service requires: {creds_desc}\n"
            + (f"\n{extra_notes}\n" if extra_notes else '')
            + f"- When the user gives you real credential value(s), repeat them back to confirm, then output EXACTLY this on its own line (no other text after it):\n"
            f"  CREDENTIALS_READY:{json_fields_example}\n"
            f"- CRITICAL: Only output CREDENTIALS_READY when you have ALL required real values from the user — never with placeholder text.\n"
            f"- After outputting CREDENTIALS_READY, say \"Done! I've filled in your credentials and enabled this source.\"\n"
            f"- Do not tell the user to \"paste it into the form\" — you will do that automatically."
        )

        chat_cfg = get_project_ai_config(
            session.get('current_project', 'default'), 'chat'
        )
        _full_cfg = load_ai_config()

        def _global_key(p):
            return _full_cfg.get('global', {}).get(p, {}).get('api_key', '').strip()

        provider = chat_cfg.get('provider', 'anthropic')
        api_key = (chat_cfg.get('api_key') or '').strip() or _global_key(provider)
        model = chat_cfg.get('model', 'claude-sonnet-4-6')

        if not api_key:
            fb_prov = chat_cfg.get('fallback_provider')
            if fb_prov:
                api_key = _global_key(fb_prov)
                if api_key:
                    provider = fb_prov
                    model = chat_cfg.get('fallback_model', model)

        if not api_key:
            return jsonify({'error': 'No AI API key configured.'}), 503

        # Build message list for LLM
        llm_messages = list(messages_in)  # [{role, content}, ...]

        if provider == 'openai':
            import openai as _oai
            client = _oai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=600,
                messages=[{'role': 'system', 'content': system_prompt}] + llm_messages,
            )
            raw_reply = resp.choices[0].message.content
        elif provider == 'anthropic':
            import anthropic as _ant
            client = _ant.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=system_prompt,
                messages=llm_messages if llm_messages else [{'role': 'user', 'content': 'Hello'}],
            )
            raw_reply = resp.content[0].text
        else:
            return jsonify({'error': f'Unsupported provider: {provider}'}), 400

        # Parse CREDENTIALS_READY signal from response
        credentials = None
        display_lines = []
        after_ready = []
        found_ready = False
        for line in raw_reply.splitlines():
            stripped = line.strip()
            if stripped.startswith('CREDENTIALS_READY:'):
                found_ready = True
                json_str = stripped[len('CREDENTIALS_READY:'):]
                try:
                    credentials = _json.loads(json_str)
                    # Filter out placeholder values
                    placeholders = {'<the actual key>', '<the actual cx>', '<username>', '<password>',
                                    '<client_id>', '<client_secret>', '<api token>'}
                    if any(v.lower().startswith('<') and v.lower().endswith('>') for v in credentials.values()):
                        credentials = None
                except Exception:
                    credentials = None
            elif found_ready:
                after_ready.append(line)
            else:
                display_lines.append(line)

        # Build display text: lines before CREDENTIALS_READY + any lines after it
        display_text = '\n'.join(display_lines).strip()
        if after_ready:
            suffix = '\n'.join(after_ready).strip()
            if suffix:
                display_text = (display_text + '\n' + suffix).strip()

        return jsonify({'response': display_text, 'credentials': credentials})
    except Exception as e:
        logger.error(f"CI key guide error: {e}")
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


@bp.route('/api/ci/cost-estimate')
@login_required
def ci_cost_estimate():
    """
    Return estimated cost for a CI run.
    Query params: docs=N&tier=N
    """
    ok, err = _ci_gate()
    if not ok:
        return err
    try:
        docs = max(1, int(request.args.get('docs', 10)))
        tier = max(1, min(5, int(request.args.get('tier', 3))))
        from analyzer.case_intelligence.task_registry import estimate_run_cost, TIER_INFO
        estimate = estimate_run_cost(docs, tier)
        tier_meta = TIER_INFO.get(tier, {})
        return jsonify({
            'estimated_usd': estimate['total_usd'],
            'breakdown_by_task': estimate['breakdown_by_task'],
            'tier': tier,
            'tier_name': tier_meta.get('name', f'Tier {tier}'),
            'docs': docs,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
