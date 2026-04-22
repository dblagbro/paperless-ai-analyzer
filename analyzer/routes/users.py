import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from analyzer.app import admin_required
from analyzer.db import (
    get_user_by_id, get_user_by_username,
    list_users, create_user as db_create_user, update_user as db_update_user,
)
from analyzer.services.smtp_service import (
    load_smtp_settings as _load_smtp_settings,
    smtp_send as _smtp_send,
)

logger = logging.getLogger(__name__)

bp = Blueprint('users', __name__)

try:
    from analyzer.app import _APP_VERSION
except ImportError:
    _APP_VERSION = '0.0.0'


# ---------------------------------------------------------------------------
# Helper: send welcome email (uses smtp_service)
# ---------------------------------------------------------------------------

def _send_welcome_email(email: str, display_name: str, username: str, role: str,
                        app_base_url: str, job_title: str = ''):
    """Send a welcome notification to a newly created user. Does not include the password."""
    try:
        from email.message import EmailMessage
        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            logger.info(f"SMTP not configured — skipping welcome email for {username}")
            return

        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        version = _APP_VERSION
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
Paperless AI Analyzer v{version}
"""
        msg = EmailMessage()
        msg['Subject'] = 'Welcome to Paperless AI Analyzer — Your Account is Ready'
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"Welcome email sent to {email} for user '{username}'")
    except Exception as e:
        logger.warning(f"Failed to send welcome email to {email}: {e}")


def _send_manual_email(email: str, display_name: str, app_base_url: str):
    """Send the user manual link to a user."""
    try:
        from email.message import EmailMessage
        smtp_cfg = _load_smtp_settings()
        if not smtp_cfg.get('host'):
            raise RuntimeError("SMTP is not configured")

        from_addr = smtp_cfg.get('from') or smtp_cfg.get('user') or 'noreply@localhost'
        version = _APP_VERSION
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
Paperless AI Analyzer v{version}
"""
        msg = EmailMessage()
        msg['Subject'] = 'Paperless AI Analyzer — User Manual'
        msg['From'] = from_addr
        msg['To'] = email
        msg.set_content(body)
        _smtp_send(smtp_cfg, msg)
        logger.info(f"Manual email sent to {email}")
    except Exception as e:
        logger.warning(f"Failed to send manual email to {email}: {e}")
        raise


# ---------------------------------------------------------------------------
# Current user routes
# ---------------------------------------------------------------------------

@bp.route('/api/me', methods=['GET'])
@login_required
def api_me_get():
    """Return the current user's own profile data."""
    row = get_user_by_id(current_user.id)
    if not row:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'id': row['id'],
        'username': row['username'],
        'display_name': row['display_name'] or '',
        'email': row['email'] or '',
        'phone': row['phone'] or '',
        'address': row['address'] or '',
        'job_title': row['job_title'] or '',
        'role': row['role'],
    })


@bp.route('/api/me', methods=['PATCH'])
@login_required
def api_me_update():
    """Update current user's own editable profile fields."""
    data = request.json or {}
    allowed = {k: v for k, v in data.items()
               if k in ('display_name', 'email', 'phone', 'address', 'job_title')}
    if not allowed:
        return jsonify({'error': 'No valid fields provided'}), 400
    if 'display_name' in allowed and not str(allowed['display_name']).strip():
        return jsonify({'error': 'Display name cannot be empty'}), 400
    try:
        db_update_user(current_user.id, **allowed)
        if 'display_name' in allowed:
            current_user.display_name = allowed['display_name'].strip()
        logger.info(f"Profile updated for user {current_user.username}: {list(allowed.keys())}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/change-password', methods=['POST'])
@login_required
def api_change_password():
    """Allow the current user to change their own password."""
    from analyzer.db import get_user_by_id, DB_PATH
    from werkzeug.security import check_password_hash, generate_password_hash
    import sqlite3
    data = request.json or {}
    current_pw = data.get('current_password', '').strip()
    new_pw = data.get('new_password', '').strip()
    if not current_pw or not new_pw:
        return jsonify({'success': False, 'error': 'Both fields are required'}), 400
    if len(new_pw) < 6:
        return jsonify({'success': False, 'error': 'New password must be at least 6 characters'}), 400
    row = get_user_by_id(current_user.id)
    if not row or not check_password_hash(row['password_hash'], current_pw):
        return jsonify({'success': False, 'error': 'Current password is incorrect'}), 403
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('UPDATE users SET password_hash=? WHERE id=?',
                     (generate_password_hash(new_pw), current_user.id))
        conn.commit()
        conn.close()
        logger.info(f"Password changed for user {current_user.username}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Admin user management routes
# ---------------------------------------------------------------------------

@bp.route('/api/users', methods=['GET'])
@login_required
@admin_required
def api_users_list():
    """List all users (admin only)."""
    try:
        rows = list_users()
        users = [{
            'id': r['id'],
            'username': r['username'],
            'display_name': r['display_name'],
            'email': r['email'] or '',
            'role': r['role'],
            'created_at': r['created_at'],
            'last_login': r['last_login'],
            'is_active': bool(r['is_active']),
            'phone': r['phone'] or '',
            'address': r['address'] or '',
            'github': r['github'] or '',
            'linkedin': r['linkedin'] or '',
            'facebook': r['facebook'] or '',
            'instagram': r['instagram'] or '',
            'other_handles': r['other_handles'] or '',
            'timezone': r['timezone'] or '',
            'job_title': r['job_title'] or '',
        } for r in rows]
        return jsonify({'users': users})
    except Exception as e:
        logger.error(f"List users error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/users', methods=['POST'])
@login_required
@admin_required
def api_users_create():
    """Create a new user (admin only), then send a welcome email if SMTP is configured."""
    try:
        data = request.json or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        role = data.get('role', 'basic')
        display_name = data.get('display_name', '').strip() or username
        email = data.get('email', '').strip()
        job_title = data.get('job_title', '').strip()
        if not username or not password:
            return jsonify({'error': 'username and password required'}), 400
        if role not in ('basic', 'advanced', 'admin'):
            return jsonify({'error': 'role must be basic, advanced, or admin'}), 400
        if get_user_by_username(username):
            return jsonify({'error': f"User '{username}' already exists"}), 409
        db_create_user(username, password, role=role, display_name=display_name, email=email)

        if email:
            app_url = request.host_url.rstrip('/') + request.script_root
            _send_welcome_email(email, display_name, username, role, app_url, job_title=job_title)

        return jsonify({'success': True, 'username': username, 'email_sent': bool(email)}), 201
    except Exception as e:
        logger.error(f"Create user error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/users/<int:uid>', methods=['PATCH'])
@login_required
@admin_required
def api_users_update(uid):
    """Update role / display_name / email / password / is_active (admin only)."""
    try:
        data = request.json or {}
        allowed = {k: v for k, v in data.items() if k in (
            'role', 'display_name', 'email', 'password', 'is_active',
            'phone', 'address', 'github', 'linkedin', 'facebook',
            'instagram', 'other_handles', 'timezone', 'job_title',
        )}
        if 'role' in allowed and allowed['role'] not in ('basic', 'advanced', 'admin'):
            return jsonify({'error': 'role must be basic, advanced, or admin'}), 400
        db_update_user(uid, **allowed)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Update user error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/users/<int:uid>', methods=['DELETE'])
@login_required
@admin_required
def api_users_deactivate(uid):
    """Soft-delete a user by setting is_active=0 (admin only)."""
    try:
        if uid == current_user.id:
            return jsonify({'error': 'Cannot deactivate yourself'}), 400
        db_update_user(uid, is_active=0)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/users/<int:uid>/send-manual', methods=['POST'])
@login_required
@admin_required
def api_users_send_manual(uid):
    """Send the user manual link to a user via email (admin only)."""
    try:
        row = get_user_by_id(uid)
        if not row:
            return jsonify({'error': 'User not found'}), 404
        user = dict(row)
        if not user.get('email'):
            return jsonify({'error': 'User has no email address configured'}), 400
        base_url = request.host_url.rstrip('/') + request.script_root
        _send_manual_email(user['email'], user.get('display_name') or user['username'], base_url)
        return jsonify({'success': True, 'message': f"Manual sent to {user['email']}"})
    except Exception as e:
        logger.error(f"Send manual error for uid {uid}: {e}")
        return jsonify({'error': str(e)}), 500
