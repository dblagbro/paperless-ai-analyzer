"""routes/ci — blueprint aggregator for CI routes.

Imports each sub-module, which register their handlers on the shared `bp`.
Public surface: `bp` (imported by analyzer/web_ui.py).
"""
from flask import Blueprint

bp = Blueprint('ci', __name__)

# Import modules so their @bp.route decorators execute and register handlers.
# They all refer to the `bp` symbol below via `from analyzer.routes.ci import bp`.
# Order is intentional — helpers before modules that import them.
from analyzer.routes.ci import helpers  # noqa: F401
from analyzer.routes.ci import setup    # noqa: F401
from analyzer.routes.ci import runs     # noqa: F401
from analyzer.routes.ci import findings # noqa: F401
from analyzer.routes.ci import reports  # noqa: F401

__all__ = ["bp"]
