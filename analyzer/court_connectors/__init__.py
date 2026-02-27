"""
Court Document Importer — v3.5.0

Pull entire case files from federal PACER/CourtListener and NYS NYSCEF
directly into a Paperless-ngx project library.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ['COURT_IMPORT_ENABLED', 'init_court_import']

# Always enabled — no longer gated by environment variable
COURT_IMPORT_ENABLED: bool = True


def init_court_import() -> None:
    """Initialize court import subsystem — creates DB schema if not exists."""
    try:
        from analyzer.court_db import init_court_db
        init_court_db()
        logger.info("Court Document Importer initialized (DB schema ready)")
    except Exception as e:
        logger.error(f"Court Document Importer init failed: {e}", exc_info=True)
