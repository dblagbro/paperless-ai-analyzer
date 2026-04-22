from flask import Blueprint, request, jsonify, current_app, redirect, url_for, render_template
from flask_login import login_required, current_user, login_user, logout_user

from analyzer.db import get_user_by_username, update_last_login
from werkzeug.security import check_password_hash

bp = Blueprint('auth', __name__)


@bp.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page."""
    if current_user.is_authenticated:
        return redirect(url_for('auth.index'))

    error = None
    username_val = ''
    if request.method == 'POST':
        username_val = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        row = get_user_by_username(username_val)
        if row and check_password_hash(row['password_hash'], password):
            from analyzer.auth import User
            user = User(row)
            login_user(user, remember=True)
            update_last_login(user.id)
            next_page = request.args.get('next') or url_for('auth.index')
            return redirect(next_page)
        error = 'Invalid username or password.'

    return render_template('login.html', error=error, username=username_val)


@bp.route('/logout')
@login_required
def logout():
    """Log out current user."""
    logout_user()
    return redirect(url_for('auth.login_page'))


@bp.route('/')
@login_required
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')
