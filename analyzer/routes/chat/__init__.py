"""routes/chat — blueprint aggregator for chat endpoints.

Handlers are split across core/sessions/branching/export modules which all
register their routes on the shared `bp`.
"""
from flask import Blueprint

bp = Blueprint('chat', __name__)

from analyzer.routes.chat import core       # noqa: F401
from analyzer.routes.chat import sessions   # noqa: F401
from analyzer.routes.chat import branching  # noqa: F401
from analyzer.routes.chat import export     # noqa: F401

__all__ = ["bp"]
