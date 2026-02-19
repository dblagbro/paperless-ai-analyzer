"""
Flask-Login integration for Paperless AI Analyzer.
"""

from flask_login import LoginManager, UserMixin

login_manager = LoginManager()
login_manager.login_view = 'login_page'
login_manager.login_message = 'Please log in to access the analyzer.'
login_manager.login_message_category = 'info'


class User(UserMixin):
    """Wraps a sqlite3.Row from the users table."""

    def __init__(self, row):
        self.id = row['id']
        self.username = row['username']
        self.display_name = row['display_name'] or row['username']
        self.role = row['role']
        self._is_active = bool(row['is_active'])

    # Flask-Login requires this to be a property
    @property
    def is_active(self):
        return self._is_active

    @property
    def is_admin(self):
        return self.role == 'admin'

    def get_id(self):
        return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    from analyzer.db import get_user_by_id
    row = get_user_by_id(int(user_id))
    return User(row) if row else None
