"""
Case Intelligence AI — v3.1.0

Proactive, evidence-grounded legal case analysis for Paperless-ngx.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = ['_ci_safety_check', 'CI_ENABLED']

# Always enabled — no longer gated by environment variable
CI_ENABLED: bool = True


def _ci_safety_check() -> bool:
    """Returns True — Case Intelligence is always enabled."""
    return True


def init_case_intelligence():
    """Initialize Case Intelligence AI — creates DB schema if not exists."""
    try:
        from analyzer.case_intelligence.db import init_ci_db
        init_ci_db()
        logger.info("Case Intelligence AI initialized (DB schema ready)")
    except Exception as e:
        logger.error(f"Case Intelligence AI init failed: {e}", exc_info=True)
