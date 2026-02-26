"""
CourtListener / RECAP connector.

Uses the free CourtListener REST API (no authentication required).
Rate limit: 5,000 requests/day unauthenticated; pass COURTLISTENER_API_TOKEN
in extra_config_json to raise the limit.

API docs: https://www.courtlistener.com/help/api/rest/
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from analyzer.court_connectors.base import CourtConnector, CaseResult, DocketEntry

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
_SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
_STORAGE_URL = "https://storage.courtlistener.com/recap"
_USER_AGENT = "Paperless-AI-Analyzer/1.0 (legal research tool)"
_TIMEOUT = 30


class CourtListenerConnector(CourtConnector):
    """
    RECAP/CourtListener REST API connector.

    No authentication required for basic access. Pass
    extra_config_json={'courtlistener_api_token': '...'} to use a token.
    """

    def __init__(self, project_slug: str, credentials: Dict[str, Any]):
        super().__init__(project_slug, credentials)
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': _USER_AGENT})

        extra = {}
        try:
            import json
            raw = credentials.get('extra_config_json', '{}') or '{}'
            extra = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass

        token = extra.get('courtlistener_api_token', '')
        if token:
            self._session.headers['Authorization'] = f'Token {token}'
        self._authenticated = True  # No auth step required for public API

    def authenticate(self) -> None:
        """CourtListener public API requires no authentication."""
        self._authenticated = True

    def test_connection(self) -> Dict[str, Any]:
        """
        Verify the CourtListener API responds.
        Case search uses the public /search/ endpoint (no auth needed).
        Docket entry downloads require an API token.
        """
        try:
            url = f"{_BASE_URL}/courts/?page_size=1"
            resp = self._session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            count = data.get('count', 0)
            token_set = 'Authorization' in self._session.headers
            if token_set:
                info = f"CourtListener API authenticated — {count} courts indexed"
            else:
                info = (
                    f"CourtListener API accessible — {count} courts indexed "
                    f"(anonymous; case search works; docket downloads require a free "
                    f"API token at courtlistener.com)"
                )
            return {'ok': True, 'account_info': info, 'error': ''}
        except requests.HTTPError as e:
            return {'ok': False, 'account_info': '', 'error': str(e)}
        except Exception as e:
            return {'ok': False, 'account_info': '', 'error': str(e)}

    def search_cases(self, case_number: str = '',
                     party_name: str = '',
                     court: str = '') -> List[CaseResult]:
        """
        Search CourtListener dockets using the public /search/ endpoint
        (works anonymously; returns up to 25 results).
        """
        # Build a query combining case number and party name
        q_parts = []
        if case_number:
            q_parts.append(case_number)
        if party_name:
            q_parts.append(party_name)
        query = ' '.join(q_parts) if q_parts else ''
        if not query:
            return []

        params: Dict[str, Any] = {
            'type': 'd',          # docket search
            'q': query,
            'page_size': 25,
        }
        if court:
            params['court'] = court

        try:
            resp = self._session.get(_SEARCH_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get('results', [])
            return [self._search_result_to_case(d) for d in results]
        except Exception as e:
            logger.error(f"CourtListener search failed: {e}")
            return []

    def get_docket(self, case_id: str) -> List[DocketEntry]:
        """Fetch all docket entries for a CourtListener docket ID.

        case_id may be a compound "{cl_id}|{court}|{case_number}" string
        (produced by _search_result_to_case); only the first part is used here.

        Raises RuntimeError if the API requires authentication (free token needed).
        """
        # Handle compound case_id — use only the numeric CL docket ID
        cl_id = case_id.split('|')[0] if '|' in case_id else case_id

        entries = []
        page = 1
        while True:
            try:
                params = {
                    'docket': cl_id,
                    'order_by': 'entry_number',
                    'page_size': 100,
                    'page': page,
                }
                resp = self._session.get(
                    f"{_BASE_URL}/docket-entries/", params=params, timeout=_TIMEOUT
                )
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        "CourtListener RECAP access required to view docket entries. "
                        "Your API token is valid for search, but the docket-entries API "
                        "requires RECAP contributor access. This case may also have no "
                        "RECAP documents if it hasn't been captured by any RECAP user. "
                        "Install the free RECAP browser extension at free.law/recap to "
                        "contribute documents and gain API access, or add PACER credentials "
                        "in ⚙️ Manage Credentials to download directly from PACER."
                    )
                resp.raise_for_status()
                data = resp.json()
                for row in data.get('results', []):
                    for doc in row.get('recap_documents', []) or []:
                        url = ''
                        if doc.get('filepath_local'):
                            url = f"{_STORAGE_URL}/{doc['filepath_local']}"
                        entries.append(DocketEntry(
                            seq=str(row.get('entry_number', '')),
                            title=(row.get('description') or doc.get('description') or '').strip(),
                            date=row.get('date_filed') or '',
                            source_url=url,
                            source='recap',
                            doc_type=doc.get('document_type', ''),
                            extra={'cl_doc_id': doc.get('id'), 'cl_entry_id': row.get('id')},
                        ))
                if not data.get('next'):
                    break
                page += 1
                time.sleep(0.2)
            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"CourtListener docket fetch failed (page {page}): {e}")
                break
        return entries

    def download_document(self, entry: DocketEntry) -> Optional[Path]:
        """Download a RECAP document to a temp file."""
        if not entry.source_url:
            return None
        try:
            resp = self._session.get(entry.source_url, timeout=60, stream=True)
            resp.raise_for_status()
            suffix = '.pdf'
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix,
                prefix=f"court_recap_{entry.seq}_"
            )
            for chunk in resp.iter_content(chunk_size=65536):
                tmp.write(chunk)
            tmp.flush()
            tmp.close()
            return Path(tmp.name)
        except Exception as e:
            logger.error(f"RECAP download failed for seq {entry.seq}: {e}")
            return None

    # ── Private helpers ──────────────────────────────────────────────────────

    def _search_result_to_case(self, d: dict) -> CaseResult:
        """Convert a /search/ result dict to a CaseResult.

        case_id is a compound string "{cl_docket_id}|{court_code}|{case_number}"
        so that FederalConnector.get_docket() can pass court+case_number to
        PACERConnector as a fallback when RECAP access is blocked.
        """
        docket_id = str(d.get('docket_id', ''))
        court_id = (d.get('court_id', '') or d.get('court', '')).lower()
        case_number = d.get('docketNumber', '')
        compound_id = (
            f"{docket_id}|{court_id}|{case_number}"
            if court_id and case_number
            else docket_id
        )
        return CaseResult(
            case_id=compound_id,
            case_number=case_number,
            case_title=d.get('caseName', '') or d.get('case_name_full', ''),
            court=d.get('court', '') or d.get('court_id', ''),
            filing_date=d.get('dateFiled') or '',
            source='courtlistener',
            extra={'cl_docket_id': docket_id},
        )

    def lookup_pacer_case_id(self, cl_docket_id: str) -> str:
        """Fetch the PACER case ID for a CourtListener docket ID."""
        try:
            resp = self._session.get(
                f"{_BASE_URL}/dockets/{cl_docket_id}/", timeout=_TIMEOUT
            )
            if resp.ok:
                return str(resp.json().get('pacer_case_id', '') or '')
        except Exception:
            pass
        return ''

    def _docket_to_case(self, d: dict) -> CaseResult:
        """Convert a /dockets/ result dict to a CaseResult (requires auth)."""
        return CaseResult(
            case_id=str(d.get('id', '')),
            case_number=d.get('docket_number', ''),
            case_title=d.get('case_name', ''),
            court=d.get('court_id', ''),
            filing_date=d.get('date_filed') or '',
            source='courtlistener',
            extra={'cl_docket_id': d.get('id'), 'pacer_case_id': d.get('pacer_case_id')},
        )
