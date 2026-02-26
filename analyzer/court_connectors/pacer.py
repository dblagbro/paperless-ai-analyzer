"""
PACER direct connector (fallback when a document is not in RECAP).

Auth: POST to PACER login form → session cookie.
Downloads: GET /doc1/<case_id>/<doc_number> with session cookie.
Rate limit: 1-second sleep between downloads (configurable).
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from analyzer.court_connectors.base import CourtConnector, CaseResult, DocketEntry

logger = logging.getLogger(__name__)

_LOGIN_URL = "https://pacer.login.uscourts.gov/csologin/login.jsf"
_USER_AGENT = "Paperless-AI-Analyzer/1.0 (legal research tool)"
_DEFAULT_RATE_LIMIT = 1.0


class PACERConnector(CourtConnector):
    """
    PACER direct connector for federal court documents not in the RECAP archive.

    Requires PACER username + password (stored encrypted in court_credentials).
    Optional: client_code in extra_config_json.
    """

    def __init__(self, project_slug: str, credentials: Dict[str, Any],
                 password: str = ''):
        super().__init__(project_slug, credentials)
        self._password = password
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': _USER_AGENT})

        extra = {}
        try:
            import json
            raw = credentials.get('extra_config_json', '{}') or '{}'
            extra = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass

        self._client_code = extra.get('pacer_client_code', '')
        self._rate_limit = float(extra.get('pacer_rate_limit_seconds', _DEFAULT_RATE_LIMIT))

    def authenticate(self) -> None:
        """
        Authenticate with PACER via form POST and capture the session cookie.
        Raises RuntimeError if login fails.

        PACER uses PrimeFaces/Jakarta Faces — the login form requires a real
        ViewState token scraped from the page before submitting.
        """
        username = self.credentials.get('username', '')
        if not username or not self._password:
            raise RuntimeError("PACER credentials (username + password) are required")

        try:
            import re

            # Step 1: GET the login page to obtain the real ViewState token
            get_resp = self._session.get(_LOGIN_URL, timeout=30)
            get_resp.raise_for_status()

            vs_match = re.search(r'jakarta\.faces\.ViewState[^>]*value="([^"]+)"', get_resp.text)
            view_state = vs_match.group(1) if vs_match else 'stateless'
            logger.debug(f"PACER ViewState: {view_state[:20]}...")

            # Step 2: POST via PrimeFaces partial/ajax — this is what PACER's login
            # button actually does (PrimeFaces.ab({s:"loginForm:fbtnLogin",...})).
            # A plain form POST shows errors differently; AJAX is the canonical path.
            ajax_headers = {
                'Faces-Request':    'partial/ajax',
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type':     'application/x-www-form-urlencoded; charset=UTF-8',
                'Accept':           'application/xml, text/xml, */*; q=0.01',
            }
            payload = {
                'javax.faces.partial.ajax':    'true',
                'javax.faces.source':          'loginForm:fbtnLogin',
                'javax.faces.partial.execute': 'loginForm',
                'javax.faces.partial.render':  (
                    'redactionConfirmation,popupMsgId,'
                    'loginForm:userNamePanel,userUpdateDlg'
                ),
                'loginForm':                   'loginForm',
                'loginForm:loginName':         username,
                'loginForm:password':          self._password,
                'loginForm:clientCode':        self._client_code,
                'loginForm:courtId_input':     '',
                'loginForm:fbtnLogin':         'loginForm:fbtnLogin',
                'jakarta.faces.ViewState':     view_state,
            }
            resp = self._session.post(
                _LOGIN_URL, data=payload, headers=ajax_headers,
                allow_redirects=True, timeout=30
            )

            # On failure the partial response contains error text; check for it
            if 'Invalid' in resp.text or 'invalid' in resp.text:
                err_match = re.search(
                    r'ui-messages-error-detail[^>]*>([^<]+)<', resp.text
                )
                detail = err_match.group(1).strip() if err_match else 'Invalid username or password'
                raise RuntimeError(f"PACER login failed — {detail}. "
                                   "Re-enter your credentials in ⚙️ Manage Credentials.")

            # On success PACER sets a session cookie
            if 'PacerSession' not in self._session.cookies and \
               'NextGenCSO' not in self._session.cookies:
                # Also try a follow-up GET — some PACER versions redirect after AJAX
                follow = self._session.get(_LOGIN_URL, timeout=30, allow_redirects=True)
                if 'PacerSession' not in self._session.cookies and \
                   'NextGenCSO' not in self._session.cookies:
                    raise RuntimeError("PACER login failed — no session cookie received. "
                                       "Check username and password.")

            self._authenticated = True
            logger.info("PACER authentication successful")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"PACER authentication error: {e}") from e

    def test_connection(self) -> Dict[str, Any]:
        """Test PACER credentials by attempting login."""
        try:
            self.authenticate()
            return {
                'ok': True,
                'account_info': f"PACER authenticated as {self.credentials.get('username', '')}",
                'error': '',
            }
        except Exception as e:
            return {'ok': False, 'account_info': '', 'error': str(e)}

    def search_cases(self, case_number: str = '',
                     party_name: str = '',
                     court: str = '') -> List[CaseResult]:
        """
        PACER case search is complex and varies by court.
        For now, return empty — use CourtListenerConnector for search,
        PACER is used only for document downloads.
        """
        return []

    def get_docket(self, case_id: str) -> List[DocketEntry]:
        """
        PACER docket retrieval requires court-specific CM/ECF URLs.
        For now, return empty — docket is fetched via CourtListener;
        PACER is used only as a download fallback.
        """
        return []

    def download_document(self, entry: DocketEntry) -> Optional[Path]:
        """
        Download a single PACER document to a temp file.

        entry.source_url should be a full PACER CM/ECF doc URL.
        """
        if not entry.source_url:
            return None

        self._ensure_authenticated()

        # Respect rate limit
        time.sleep(self._rate_limit)

        try:
            resp = self._session.get(entry.source_url, timeout=60, stream=True)
            resp.raise_for_status()

            # Check content type — should be PDF
            content_type = resp.headers.get('Content-Type', '')
            if 'html' in content_type.lower():
                # PACER sometimes redirects to a CAPTCHA or fee confirmation page
                logger.warning(f"PACER returned HTML for seq {entry.seq} — possible fee gate")
                return None

            suffix = '.pdf'
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix,
                prefix=f"court_pacer_{entry.seq}_"
            )
            for chunk in resp.iter_content(chunk_size=65536):
                tmp.write(chunk)
            tmp.flush()
            tmp.close()
            return Path(tmp.name)
        except Exception as e:
            logger.error(f"PACER download failed for seq {entry.seq}: {e}")
            return None
