"""
SMTP / email service — settings I/O and all outbound email helpers.

This module is framework-agnostic (no Flask). Import it from route handlers.
"""

import json
import logging
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from analyzer import __version__ as _APP_VERSION

logger = logging.getLogger(__name__)

_SMTP_SETTINGS_FILE = Path('/app/data/smtp_settings.json')
_SMTP_DEFAULTS = {
    'host': '', 'port': 587, 'starttls': True,
    'user': '', 'pass': '', 'from': '', 'helo': '',
    'bug_report_to': 'dblagbro@voipguru.org',
}


# ---------------------------------------------------------------------------
# Settings I/O
# ---------------------------------------------------------------------------

def load_smtp_settings() -> dict:
    try:
        if _SMTP_SETTINGS_FILE.exists():
            return {**_SMTP_DEFAULTS, **json.loads(_SMTP_SETTINGS_FILE.read_text())}
    except Exception:
        pass
    return dict(_SMTP_DEFAULTS)


def save_smtp_settings(settings: dict):
    _SMTP_SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


# ---------------------------------------------------------------------------
# Low-level send
# ---------------------------------------------------------------------------

def smtp_send(smtp_cfg: dict, msg: EmailMessage):
    host = smtp_cfg.get('host', '')
    port = int(smtp_cfg.get('port', 587))
    starttls = bool(smtp_cfg.get('starttls', True))
    user = smtp_cfg.get('user', '')
    pwd = smtp_cfg.get('pass', '')
    helo = smtp_cfg.get('helo') or None
    if not host:
        raise RuntimeError('SMTP host is not configured')
    with smtplib.SMTP(host, port, local_hostname=helo) as s:
        s.ehlo()
        if starttls:
            s.starttls(context=ssl.create_default_context())
            s.ehlo()
        if user:
            s.login(user, pwd)
        s.send_message(msg)


# ---------------------------------------------------------------------------
# Outbound email templates
# ---------------------------------------------------------------------------

def send_welcome_email(email: str, display_name: str, username: str, role: str,
                       app_base_url: str, job_title: str = ''):
    try:
        smtp_cfg = load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info(f"SMTP not configured — skipping welcome email for {username}")
            return
        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        base = app_base_url.rstrip('/')
        docs_url = f"{base}/docs"
        github_url = 'https://github.com/dblagbro/paperless-ai-analyzer'
        job_title_line = f"\n  Job Title: {job_title}" if job_title else ""
        body = f"""Hi {display_name},

Your Paperless AI Analyzer account has been created and is ready to use.

Account Details
───────────────
  Username : {username}
  Role     : {role.capitalize()}{job_title_line}

Access the Application
──────────────────────
  {base}/

User Manual
───────────
  The full user manual is available at:
  {docs_url}

  Key sections:
  • Quick Start          {docs_url}/getting-started
  • Projects             {docs_url}/projects
  • Smart Upload         {docs_url}/upload
  • AI Chat              {docs_url}/chat
  • Search & Analysis    {docs_url}/search
  • Anomaly Detection    {docs_url}/anomaly-detection
  • Configuration        {docs_url}/configuration

Resources
─────────
  GitHub / README : {github_url}#readme

If you have any questions, please contact your system administrator.

—
Paperless AI Analyzer v{_APP_VERSION}
"""
        msg = EmailMessage()
        msg['Subject'] = 'Welcome to Paperless AI Analyzer — Your Account is Ready'
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        smtp_send(smtp_cfg, msg)
        logger.info(f"Welcome email sent to {email} for user '{username}'")
    except Exception as e:
        logger.warning(f"Failed to send welcome email to {email}: {e}")


def send_manual_email(email: str, display_name: str, app_base_url: str):
    try:
        smtp_cfg = load_smtp_settings()
        if not smtp_cfg.get('host'):
            raise RuntimeError("SMTP is not configured")
        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        base = app_base_url.rstrip('/')
        docs_url = f"{base}/docs"
        body = f"""Hi {display_name},

Here is a link to the Paperless AI Analyzer user manual:

  {docs_url}

Manual Sections
───────────────
  Overview & Features    {docs_url}/overview
  Quick Start Guide      {docs_url}/getting-started
  Projects               {docs_url}/projects
  Smart Upload           {docs_url}/upload
  AI Chat                {docs_url}/chat
  Search & Analysis      {docs_url}/search
  Anomaly Detection      {docs_url}/anomaly-detection
  Debug & Tools          {docs_url}/tools
  Configuration          {docs_url}/configuration
  User Management        {docs_url}/users
  LLM Usage & Cost       {docs_url}/llm-usage
  API Reference          {docs_url}/api

—
Paperless AI Analyzer v{_APP_VERSION}
"""
        msg = EmailMessage()
        msg['Subject'] = 'Paperless AI Analyzer — User Manual'
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        smtp_send(smtp_cfg, msg)
        logger.info(f"Manual email sent to {email}")
    except Exception as e:
        logger.warning(f"Failed to send manual email to {email}: {e}")
        raise


def send_ci_budget_notification(run_id: str, pct_complete: float,
                                cost_so_far: float, projected_total: float,
                                budget: float, status: str, is_urgent: bool = False):
    try:
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
        smtp_cfg = load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info("SMTP not configured — skipping CI budget notification")
            return
        goal_text = run['goal_text'] if 'goal_text' in run.keys() else 'Unknown Case'
        allow_overage_pct = run.get('allow_overage_pct') or 0
        status_label = {
            'under_budget': 'Under Budget', 'on_track': 'On Track',
            'over_budget': 'OVER BUDGET', 'blocked': 'BUDGET BLOCKED',
        }.get(status, status)
        pct_int = int(round(pct_complete))
        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        if is_urgent or status == 'blocked':
            subject = f"URGENT: CI Budget {pct_int}% — {goal_text[:40]} — {status_label}"
        else:
            subject = f"CI Budget Update — {goal_text[:40]} — {pct_int}% complete — {status_label}"
        overage_line = ''
        if allow_overage_pct == -1:
            overage_line = 'Overage Policy: Unlimited (budget is a goal only — run will not be blocked)\n'
        elif allow_overage_pct > 0:
            hard_limit = budget * (1 + allow_overage_pct / 100)
            overage_line = f'Overage Policy: Up to {allow_overage_pct}% allowed (hard limit: ${hard_limit:.2f})\n'
        body = (
            f"{'⚠️  URGENT — ' if is_urgent or status == 'blocked' else ''}"
            f"Case Intelligence Budget {'ALERT' if is_urgent or status == 'blocked' else 'Update'}\n"
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
            body += "\n🛑 The run has been STOPPED. Budget ceiling reached.\n"
        elif status == 'over_budget':
            body += "\n⚠️  WARNING: Projected cost exceeds budget.\n"
        if is_urgent and status != 'blocked':
            body += "\n⚠️  Approaching budget limit — review and consider adjusting budget or stopping the run.\n"
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        smtp_send(smtp_cfg, msg)
        logger.info(f"CI budget notification sent to {email} ({pct_int}%, {status}, urgent={is_urgent})")
    except Exception as e:
        logger.warning(f"Failed to send CI budget notification: {e}")


def send_ci_complete_notification(run_id: str):
    try:
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
        smtp_cfg = load_smtp_settings()
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
        smtp_send(smtp_cfg, msg)
        logger.info(f"CI complete notification sent to {email} for run {run_id}")
    except Exception as e:
        logger.warning(f"Failed to send CI complete notification: {e}")
