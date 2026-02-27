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
        """
        Fetch docket via CourtListener/RECAP first.
        Falls back to PACER direct CM/ECF docket if:
          - The docket-entries API returns a RECAP-access 403 (no contributor access), OR
          - CourtListener returns 0 entries (case not yet in RECAP archive).

        case_id is expected to be the compound "{cl_id}|{court}|{case_number}"
        string produced by CourtListenerConnector._search_result_to_case().
        """
        def _pacer_fallback(reason: str) -> Optional[List[DocketEntry]]:
            if not (self._pacer_available and self._pacer):
                return None
            parts = case_id.split('|')
            if len(parts) >= 3:
                cl_docket_id, court_code, case_number = parts[0], parts[1], parts[2]
                # Enrich with pacer_case_id so PACER can navigate directly to DktRpt.pl
                pacer_case_id = self._cl.lookup_pacer_case_id(cl_docket_id)
                pacer_id = (
                    f"{court_code}|{case_number}|{pacer_case_id}"
                    if pacer_case_id
                    else f"{court_code}|{case_number}"
                )
                logger.info(f"{reason} — trying PACER direct docket for {pacer_id}")
                return self._pacer.get_docket(pacer_id)
            logger.warning(
                "PACER docket fallback skipped: case_id is missing "
                "court and case_number (search-based case_ids have them; "
                "manually entered IDs do not)."
            )
            return None

        try:
            entries = self._cl.get_docket(case_id)
            if not entries and self._pacer_available:
                # Case exists in CL search index but has no RECAP-contributed documents.
                # Try PACER direct docket before giving up.
                pacer_entries = _pacer_fallback("CourtListener returned 0 RECAP entries")
                if pacer_entries:
                    return pacer_entries
            return entries
        except RuntimeError as e:
            if 'RECAP' in str(e):
                pacer_entries = _pacer_fallback("CourtListener RECAP access blocked (403)")
                if pacer_entries is not None:
                    return pacer_entries
            raise

    def download_document(self, entry: DocketEntry) -> Optional[Path]:
        """Download a document, preferring RECAP; falling back to PACER.

        ECF/PACER URLs (*.uscourts.gov/doc1/...) bypass CourtListener entirely
        and go straight to the PACER connector so that fee-gate handling runs.
        """
        # Route ECF URLs directly to PACER to avoid the CourtListener connector
        # downloading an HTML fee-confirmation page and returning it as a PDF.
        is_ecf_url = bool(
            entry.source_url and 'uscourts.gov' in entry.source_url
        )

        if entry.source_url and not is_ecf_url:
            # Try RECAP / CourtListener for non-PACER URLs
            result = self._cl.download_document(entry)
            if result:
                return result
            logger.debug(f"RECAP download failed for seq {entry.seq}, trying PACER fallback")
        elif is_ecf_url:
            logger.debug(f"ECF URL for seq {entry.seq} — routing directly to PACER connector")

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
