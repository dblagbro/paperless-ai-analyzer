"""Shared helpers used across the CI route modules.

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

