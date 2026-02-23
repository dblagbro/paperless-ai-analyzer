"""
Case Intelligence AI — v3.1.0

Proactive, evidence-grounded legal case analysis for Paperless-ngx.

Gated by the CASE_INTELLIGENCE_ENABLED=true environment variable.
Never import this package in production code paths without calling
_ci_safety_check() first.
"""

import os
import logging

logger = logging.getLogger(__name__)

__all__ = ['_ci_safety_check', 'CI_ENABLED']

# Feature flag — cached at import time
CI_ENABLED: bool = (
    os.environ.get('CASE_INTELLIGENCE_ENABLED', 'false').lower() == 'true'
)


def _ci_safety_check() -> bool:
    """
    Returns True if CASE_INTELLIGENCE_ENABLED=true is set, False otherwise.
    """
    return os.environ.get('CASE_INTELLIGENCE_ENABLED', 'false').lower() == 'true'


def init_case_intelligence():
    """
    Initialize Case Intelligence AI — creates DB schema if not exists.
    Call this once on application startup when CI_ENABLED is True.
    """
    if not _ci_safety_check():
        return

    try:
        from analyzer.case_intelligence.db import init_ci_db
        init_ci_db()
        logger.info("Case Intelligence AI initialized (DB schema ready)")
    except Exception as e:
        logger.error(f"Case Intelligence AI init failed: {e}", exc_info=True)
