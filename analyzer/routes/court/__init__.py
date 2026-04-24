"""Court import blueprint — split into cohesive modules during the v3.9.9 refactor.

Package layout (see refactor-log Entry 012):
    helpers.py       # _court_gate, _build_court_connector, _run_court_import, analyze helpers
    credentials.py   # credential CRUD + paste-parse
    search.py        # docket search + fetch
    imports.py       # import start/status/cancel/history + analyze-missing

All submodules register routes onto the `bp` defined here via @bp.route.
"""
import logging

from flask import Blueprint

logger = logging.getLogger(__name__)
bp = Blueprint('court', __name__)

# Side-effect imports — each submodule's @bp.route decorators fire on import.
from . import helpers          # noqa: E402, F401
from . import credentials      # noqa: E402, F401
from . import search           # noqa: E402, F401
from . import imports          # noqa: E402, F401
