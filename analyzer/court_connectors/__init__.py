"""
Court Document Importer — v3.5.0

Pull entire case files from federal PACER/CourtListener and NYS NYSCEF
directly into a Paperless-ngx project library.

Gated by COURT_IMPORT_ENABLED=true environment variable.
Never import connectors without checking COURT_IMPORT_ENABLED first.
"""

import os
import logging

logger = logging.getLogger(__name__)

__all__ = ['COURT_IMPORT_ENABLED', 'init_court_import']

# Feature flag — cached at import time
COURT_IMPORT_ENABLED: bool = (
    os.environ.get('COURT_IMPORT_ENABLED', 'false').lower() == 'true'
)


def init_court_import() -> None:
    """
    Initialize court import subsystem — creates DB schema if not exists.
    Call once on application startup when COURT_IMPORT_ENABLED is True.
    """
    if not COURT_IMPORT_ENABLED:
        return

    try:
        from analyzer.court_db import init_court_db
        init_court_db()
        logger.info("Court Document Importer initialized (DB schema ready)")
    except Exception as e:
        logger.error(f"Court Document Importer init failed: {e}", exc_info=True)
