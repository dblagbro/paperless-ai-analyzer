"""
Web UI orchestrator — registers all Blueprint route modules onto the Flask app.

This file's only job is to wire everything together. Feature code lives in:
  - analyzer/app.py          (Flask app instance, middleware, auth, factories)
  - analyzer/routes/*.py     (Blueprint route modules)
  - analyzer/services/*.py   (cross-cutting service logic)

Entry points re-exported here for backward compatibility with main.py:
  start_web_server_thread, update_ui_stats
"""

from analyzer.app import app, create_app, start_web_server_thread, update_ui_stats  # noqa: F401

from analyzer.routes import (  # noqa: F401
    auth, status, profiles, chat, vector, documents,
    projects, upload, ai_config, users, system,
    ci, court, forms, docs,
)

_BLUEPRINTS = [
    auth.bp,
    status.bp,
    profiles.bp,
    chat.bp,
    vector.bp,
    documents.bp,
    projects.bp,
    upload.bp,
    ai_config.bp,
    users.bp,
    system.bp,
    ci.bp,
    court.bp,
    forms.bp,
    docs.bp,
]

for _bp in _BLUEPRINTS:
    app.register_blueprint(_bp)
