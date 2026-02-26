"""
Federal court connector — RECAP-first, PACER fallback.

Strategy:
  1. Use CourtListenerConnector for case search and docket.
  2. For each docket entry:
     - If source_url exists (RECAP copy available): download via CourtListener.
     - If not in RECAP and PACER credentials are configured: download via PACER.
     - If neither available: mark doc as failed with 'no_source'.

This keeps costs near-zero (RECAP is free) while still reaching PACER-only docs
when the operator has configured PACER credentials.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyzer.court_connectors.base import CourtConnector, CaseResult, DocketEntry
from analyzer.court_connectors.recap_courtlistener import CourtListenerConnector

logger = logging.getLogger(__name__)


class FederalConnector(CourtConnector):
    """
    Composite federal connector: CourtListener (free) + optional PACER fallback.
    """

    def __init__(self, project_slug: str, credentials: Dict[str, Any],
                 pacer_password: str = ''):
        super().__init__(project_slug, credentials)
        self._cl = CourtListenerConnector(project_slug, credentials)
        self._pacer: Optional[Any] = None
        self._pacer_available = bool(
            credentials.get('username') and pacer_password
        )
        if self._pacer_available:
            try:
                from analyzer.court_connectors.pacer import PACERConnector
                self._pacer = PACERConnector(project_slug, credentials, pacer_password)
            except Exception as e:
                logger.warning(f"PACER connector init failed: {e}")
                self._pacer_available = False

    def authenticate(self) -> None:
        """Authenticate both connectors."""
        self._cl.authenticate()
        if self._pacer:
            try:
                self._pacer.authenticate()
            except Exception as e:
                logger.warning(f"PACER auth failed (will use RECAP-only): {e}")
                self._pacer_available = False
        self._authenticated = True

    def test_connection(self) -> Dict[str, Any]:
        """Test CourtListener (always) and PACER (if credentials present)."""
        cl_result = self._cl.test_connection()
        if not cl_result['ok']:
            return cl_result

        info_parts = [cl_result['account_info']]
        if self._pacer:
            pacer_result = self._pacer.test_connection()
            if pacer_result['ok']:
                info_parts.append(pacer_result['account_info'])
            else:
                info_parts.append(f"PACER: {pacer_result['error']}")

        return {
            'ok': True,
            'account_info': ' | '.join(info_parts),
            'error': '',
        }

    def search_cases(self, case_number: str = '',
                     party_name: str = '',
                     court: str = '') -> List[CaseResult]:
        return self._cl.search_cases(case_number=case_number,
                                     party_name=party_name,
                                     court=court)

    def get_docket(self, case_id: str) -> List[DocketEntry]:
        return self._cl.get_docket(case_id)

    def download_document(self, entry: DocketEntry) -> Optional[Path]:
        """Try RECAP first; fall back to PACER if available."""
        if entry.source_url:
            # RECAP copy available
            result = self._cl.download_document(entry)
            if result:
                return result
            logger.debug(f"RECAP download failed for seq {entry.seq}, trying PACER fallback")

        if self._pacer_available and self._pacer:
            result = self._pacer.download_document(entry)
            if result:
                return result

        if not entry.source_url and not self._pacer_available:
            logger.debug(f"No source for seq {entry.seq}: not in RECAP and no PACER credentials")

        return None

    @property
    def pacer_available(self) -> bool:
        return self._pacer_available
