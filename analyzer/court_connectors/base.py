"""
Abstract base class for all court connectors.

Every connector must implement:
  - authenticate()        → None   (raises on failure)
  - test_connection()     → dict   {'ok': bool, 'account_info': str, 'error': str}
  - search_cases(...)     → list[CaseResult]
  - get_docket(case_id)   → list[DocketEntry]
  - download_document(entry) → Path  (temp file, caller must clean up)
"""

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """A case returned by a case-search query."""
    case_id: str             # Court-system-specific unique ID (docket ID for CL, etc.)
    case_number: str         # Human-readable case number, e.g. "1:23-cv-04567"
    case_title: str          # Full case caption
    court: str               # Court identifier, e.g. "S.D.N.Y."
    filing_date: str         # ISO date or empty string
    source: str              # 'courtlistener' | 'pacer' | 'nyscef'
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocketEntry:
    """A single docket entry (one document in the case)."""
    seq: str                 # Docket entry / sequence number
    title: str               # Description of the document
    date: str                # ISO date or empty string
    source_url: str          # Direct download URL (may be empty for PACER-only docs)
    source: str              # 'recap' | 'pacer' | 'nyscef'
    doc_type: str = ''       # e.g. 'motion', 'order', 'complaint' — informational
    extra: Dict[str, Any] = field(default_factory=dict)


class CourtConnector(abc.ABC):
    """Abstract interface for all court system connectors."""

    def __init__(self, project_slug: str, credentials: Dict[str, Any]):
        self.project_slug = project_slug
        self.credentials = credentials
        self._authenticated = False

    # ── Required abstract methods ────────────────────────────────────────────

    @abc.abstractmethod
    def authenticate(self) -> None:
        """
        Establish an authenticated session with the court system.
        Raises RuntimeError or requests.HTTPError on failure.
        """

    @abc.abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Test credentials without starting a real import.

        Returns:
            {'ok': bool, 'account_info': str, 'error': str}
        """

    @abc.abstractmethod
    def search_cases(self, case_number: str = '',
                     party_name: str = '',
                     court: str = '') -> List[CaseResult]:
        """
        Search for cases matching the given criteria.

        Args:
            case_number: Partial or full case number
            party_name:  Party name (plaintiff or defendant)
            court:       Court identifier filter

        Returns:
            List of CaseResult, up to 25 results.
        """

    @abc.abstractmethod
    def get_docket(self, case_id: str) -> List[DocketEntry]:
        """
        Retrieve all docket entries for a case.

        Args:
            case_id: Court-system-specific case identifier

        Returns:
            List of DocketEntry in chronological order.
        """

    @abc.abstractmethod
    def download_document(self, entry: DocketEntry) -> Optional[Path]:
        """
        Download a single docket entry to a temp file.

        Args:
            entry: DocketEntry to download

        Returns:
            Path to temp file, or None if download not possible.
            Caller is responsible for deleting the file when done.
        """

    # ── Helper ───────────────────────────────────────────────────────────────

    def _ensure_authenticated(self):
        if not self._authenticated:
            self.authenticate()
            self._authenticated = True
