"""
Case Intelligence AI — v3.0.0

Proactive, evidence-grounded legal case analysis for Paperless-ngx.

This package is DEV-ONLY. It is gated by:
  1. CASE_INTELLIGENCE_ENABLED=true environment variable
  2. URL_PREFIX must be 'paperless-ai-analyzer-dev' or '' (local dev)

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
    Returns True if CI is enabled and running in a permitted environment.

    Permitted environments:
      - URL_PREFIX = 'paperless-ai-analyzer-dev'  (Docker dev container)
      - URL_PREFIX = ''                            (local testing, no nginx prefix)

    Raises RuntimeError if CI is enabled but URL_PREFIX is not permitted.
    This is a hard safety guard to prevent CI from running in staging or production.
    """
    enabled = os.environ.get('CASE_INTELLIGENCE_ENABLED', 'false').lower() == 'true'
    if not enabled:
        return False

    url_prefix = os.environ.get('URL_PREFIX', '').strip('/')
    permitted = ('paperless-ai-analyzer-dev', '')

    if url_prefix not in permitted:
        raise RuntimeError(
            f"SAFETY VIOLATION: Case Intelligence AI is DEV-only. "
            f"URL_PREFIX={url_prefix!r} is not a permitted dev environment "
            f"(permitted: {permitted}). Refusing to execute."
        )

    return True


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
