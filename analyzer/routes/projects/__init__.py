"""Projects blueprint — split into cohesive modules during the v3.9.7 refactor.

Package layout (see refactor-log Entry 010):
    core.py              # CRUD + archive/unarchive + current-project
    paperless_config.py  # per-project Paperless config + health-check + doc-link
    provisioning.py      # provision-snippets + provision-status + reprovision
    migration.py         # migrate-to-own-paperless + migration-status + migrate-documents
    documents.py         # list/delete project docs, orphan-documents, assign-project, reanalyze

All submodules decorate routes onto the single `bp` defined here. Importing
this package registers every route via module side-effects.
"""
import logging

from flask import Blueprint

logger = logging.getLogger(__name__)
bp = Blueprint('projects', __name__)

# Side-effect imports — each submodule's @bp.route decorators fire on import.
# Order doesn't affect behavior but we go lifecycle-ish: create → configure
# → provision → migrate → operate.
from . import core                # noqa: E402, F401
from . import paperless_config    # noqa: E402, F401
from . import provisioning        # noqa: E402, F401
from . import migration           # noqa: E402, F401
from . import documents           # noqa: E402, F401
