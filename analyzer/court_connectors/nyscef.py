"""
NYSCEF connector — New York State Courts Electronic Filing System.

Uses Playwright headless Chromium to:
  1. Log in with NY Attorney Registration # + password
  2. Search by index number + county
  3. Scrape the document list
  4. Extract cookies and replay downloads via requests.Session

ALL DOM selectors are centralised in NYSCEF_SELECTORS so a one-line
fix handles any layout change NYS OCA makes to the site.

Requires playwright>=1.40.0 and Chromium installed:
  playwright install chromium --with-deps
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyzer.court_connectors.base import CourtConnector, CaseResult, DocketEntry

logger = logging.getLogger(__name__)

# ── All NYSCEF selectors live here ──────────────────────────────────────────
# Verified against live site 2026-02
NYSCEF_SELECTORS = {
    'username':     '#txtUserName',
    'password':     '#pwPassword',          # was #txtPassword — live site uses #pwPassword
    'login_btn':    '#btnLogin',
    'index_number': '#txtCaseIdentifierNumber',  # full case number field (e.g. EF2002-477)
    'county_select':'#txtCounty',           # text autocomplete, not a <select>
    'search_btn':   '[name=btnSubmit]',     # no id — identified by name
    'doc_list':     '#documentList tr[data-docid]',
    'doc_title':    'td.docTitle',
    'doc_date':     'td.docDate',
}

# Chromium launch args + init script to pass Cloudflare Bot Management
_STEALTH_ARGS = [
    '--disable-blink-features=AutomationControlled',
    '--no-sandbox',
]
_STEALTH_SCRIPT = 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
_STEALTH_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)

_NYSCEF_BASE = "https://iapps.courts.state.ny.us/nyscef"
_LOGIN_URL   = f"{_NYSCEF_BASE}/Login"


def _check_playwright() -> bool:
    """Return True if Playwright and Chromium are available."""
    try:
        from playwright.sync_api import sync_playwright  # noqa
        return True
    except ImportError:
        return False


class NYSCEFConnector(CourtConnector):
    """
    NYSCEF connector using Playwright headless Chromium.

    Requires COURT_IMPORT_ENABLED=true AND Playwright installed.
    If Playwright is not available, all methods raise RuntimeError
    with a clear message rather than crashing silently.
    """

    def __init__(self, project_slug: str, credentials: Dict[str, Any],
                 password: str = ''):
        super().__init__(project_slug, credentials)
        self._password = password
        self._playwright = None
        self._browser = None
        self._context = None
        self._cookies: List[dict] = []

        import json
        extra = {}
        try:
            raw = credentials.get('extra_config_json', '{}') or '{}'
            extra = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass
        self._default_county = extra.get('nyscef_county', '')
        raw_pub = extra.get('public_only', credentials.get('public_only', False))
        self._public_only = raw_pub if isinstance(raw_pub, bool) else str(raw_pub).lower() == 'true'

    def _require_playwright(self):
        if not _check_playwright():
            raise RuntimeError(
                "Playwright is not installed. Rebuild the Docker image with "
                "INCLUDE_PLAYWRIGHT=true to enable NYSCEF support."
            )

    def authenticate(self) -> None:
        """Log in to NYSCEF via Playwright and capture session cookies.

        In public_only mode, starts an anonymous browser session without logging
        in — only publicly available case documents can be viewed.
        """
        self._require_playwright()

        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True, args=_STEALTH_ARGS)
        context = browser.new_context(
            user_agent=_STEALTH_UA,
            viewport={'width': 1280, 'height': 720},
            locale='en-US',
        )
        context.add_init_script(_STEALTH_SCRIPT)

        username = self.credentials.get('username', '')

        if not username or not self._password:
            if self._public_only:
                # No credentials at all — start an anonymous session.
                # NOTE: anonymous CaseSearch submissions will hit Cloudflare CAPTCHA.
                # A free Pro Se account (Unrepresented Litigants → Create an Account)
                # is required for reliable case searching.
                self._playwright = pw
                self._browser = browser
                self._context = context
                self._authenticated = True
                logger.warning(
                    "NYSCEF public_only mode started without credentials — "
                    "anonymous searches will encounter CAPTCHA. "
                    "Create a free Pro Se account at iapps.courts.state.ny.us/nyscef "
                    "and enter those credentials for reliable access."
                )
                return
            browser.close()
            pw.stop()
            raise RuntimeError("NYSCEF requires a username and password. "
                               "Attorneys use their NY Attorney Registration # + e-filing password. "
                               "Non-attorneys (Pro Se litigants) can create a free account at "
                               "iapps.courts.state.ny.us/nyscef → Unrepresented Litigants → Create an Account.")

        page = context.new_page()
        try:
            page.goto(_LOGIN_URL, wait_until='domcontentloaded', timeout=30000)
            page.fill(NYSCEF_SELECTORS['username'], username)
            page.fill(NYSCEF_SELECTORS['password'], self._password)
            page.click(NYSCEF_SELECTORS['login_btn'])
            # Wait for navigation away from the login page
            page.wait_for_function(
                "() => !window.location.pathname.endsWith('/Login')",
                timeout=15000
            )
            # Check for error still on login page
            if '/Login' in page.url:
                err_text = page.inner_text('.errors, [class*=error]') or 'Login failed'
                raise RuntimeError(err_text.strip())
            page.close()
            self._cookies = context.cookies()
            self._playwright = pw
            self._browser = browser
            self._context = context
            self._authenticated = True
            logger.info("NYSCEF authentication successful")
        except RuntimeError:
            browser.close()
            pw.stop()
            raise
        except Exception as e:
            browser.close()
            pw.stop()
            raise RuntimeError(f"NYSCEF login failed: {e}") from e

    def test_connection(self) -> Dict[str, Any]:
        """Test NYSCEF credentials by logging in and navigating to Case Search.

        In public_only mode, verifies the NYSCEF site is reachable without login.
        """
        try:
            self.authenticate()
            if self._public_only:
                # Verify public CaseSearch page is reachable
                page = self._context.new_page()
                page.goto(f"{_NYSCEF_BASE}/CaseSearch", wait_until='domcontentloaded', timeout=20000)
                page.close()
                return {
                    'ok': True,
                    'account_info': (
                        'NYSCEF public access configured — search by index number '
                        '(no attorney login required)'
                    ),
                    'error': '',
                }
            return {
                'ok': True,
                'account_info': (
                    f"NYSCEF authenticated as {self.credentials.get('username', '')} "
                    f"(NY Attorney Reg #)"
                ),
                'error': '',
            }
        except Exception as e:
            return {'ok': False, 'account_info': '', 'error': str(e)}
        finally:
            self._close_browser()

    def search_cases(self, case_number: str = '',
                     party_name: str = '',
                     court: str = '') -> List[CaseResult]:
        """
        Search NYSCEF by index number (and optional county).

        Returns ALL matching cases — the same index number may exist in multiple
        counties (e.g. EF2022-477 appears in Montgomery, Delaware, Otsego, etc.).
        Each CaseResult.case_id uses the compound format "EF2022-477|Montgomery County"
        so get_docket() can fill the county field to select the right case.

        party_name is not supported by NYSCEF's public search.
        """
        self._require_playwright()
        self._ensure_authenticated()

        if not case_number:
            return []

        import re as _re

        page = self._context.new_page()
        try:
            page.goto(f"{_NYSCEF_BASE}/CaseSearch", wait_until='domcontentloaded', timeout=20000)
            page.fill(NYSCEF_SELECTORS['index_number'], case_number)
            # NOTE: county_select (#txtCounty) is a complex AJAX widget — page.fill()
            # does not work on it. Search without county to get all matching results
            # across all counties; the compound case_id carries county for get_docket().
            page.click(NYSCEF_SELECTORS['search_btn'])
            page.wait_for_load_state('domcontentloaded', timeout=15000)

            # Direct hit: single result sent straight to DocumentList
            if 'DocumentList' in page.url:
                return [CaseResult(
                    case_id=case_number,
                    case_number=case_number,
                    case_title=(page.title() or case_number).replace('NYSCEF', '').strip(' -:'),
                    court='nyscef',
                    filing_date='',
                    source='nyscef',
                    extra={'county': self._default_county, 'search_url': page.url},
                )]

            # Results list: extract each link to DocumentList and its surrounding row data
            results = []
            seen_ids: set = set()
            for link in page.query_selector_all('a[href*="DocumentList"]'):
                href = link.get_attribute('href') or ''
                try:
                    cells = link.evaluate(
                        '''el => {
                            const row = el.closest("tr");
                            if (!row) return [];
                            return Array.from(row.querySelectorAll("td"))
                                       .map(td => td.innerText.trim());
                        }'''
                    )
                except Exception:
                    cells = []

                # Find the court/county cell and the caption cell
                court_text = next((c for c in cells if 'County' in c), '')
                caption = next(
                    (c for c in cells
                     if len(c) > 20 and 'County' not in c and not c[:5].count('/') >= 2),
                    case_number
                )

                # Extract "Foo County" from "Foo County Supreme Court"
                county_m = _re.search(r'^([\w\s]+?County)', court_text)
                county = county_m.group(1).strip() if county_m else self._default_county

                compound_id = f"{case_number}|{county}" if county else case_number
                if compound_id in seen_ids:
                    continue
                seen_ids.add(compound_id)

                results.append(CaseResult(
                    case_id=compound_id,
                    case_number=case_number,
                    case_title=caption,
                    court=court_text or 'nyscef',
                    filing_date='',
                    source='nyscef',
                    extra={'county': county, 'search_url': href},
                ))

            if results:
                return results

            # Fallback: no DocumentList links — return page title as single result
            title = page.title() or ''
            return [CaseResult(
                case_id=case_number,
                case_number=case_number,
                case_title=title.replace('NYSCEF', '').strip(' -:') or case_number,
                court='nyscef',
                filing_date='',
                source='nyscef',
                extra={'county': self._default_county, 'search_url': page.url},
            )]
        except Exception as e:
            logger.error(f"NYSCEF case search failed: {e}")
            return []
        finally:
            page.close()

    def get_docket(self, case_id: str) -> List[DocketEntry]:
        """Scrape the NYSCEF document list for a case index number.

        case_id may be a plain index number ("EF2022-477") or a compound
        "index|county" string ("EF2022-477|Montgomery County") produced by
        search_cases() when multiple counties share the same index number.
        """
        self._require_playwright()
        self._ensure_authenticated()

        # Parse compound case_id: "EF2022-477|Montgomery County"
        if '|' in case_id:
            index_number, county = case_id.split('|', 1)
        else:
            index_number = case_id
            county = self._default_county

        page = self._context.new_page()
        entries: List[DocketEntry] = []
        try:
            page.goto(f"{_NYSCEF_BASE}/CaseSearch", wait_until='domcontentloaded', timeout=20000)
            page.fill(NYSCEF_SELECTORS['index_number'], index_number)
            # NOTE: county_select (#txtCounty) is a complex AJAX widget that page.fill()
            # cannot interact with reliably. Search without county to get all results,
            # then pick the row matching the county from the compound case_id.
            page.click(NYSCEF_SELECTORS['search_btn'])
            page.wait_for_load_state('domcontentloaded', timeout=20000)

            # Authenticated users land on DocumentList directly; if not, follow the link
            if 'DocumentList' not in page.url:
                county_to_match = county or self._default_county
                target_link = None
                all_links = page.query_selector_all('a[href*="DocumentList"]')
                if county_to_match and len(all_links) > 1:
                    # Multiple results — click the row that contains our county
                    for link in all_links:
                        row_text = link.evaluate(
                            'el => { const r = el.closest("tr"); return r ? r.innerText : ""; }'
                        )
                        if county_to_match.lower().split()[0] in row_text.lower():
                            target_link = link
                            logger.debug(f"NYSCEF: matched county '{county_to_match}' in row")
                            break
                if not target_link and all_links:
                    target_link = all_links[0]
                if target_link:
                    target_link.click()
                    page.wait_for_load_state('domcontentloaded', timeout=15000)

            rows = page.query_selector_all(NYSCEF_SELECTORS['doc_list'])
            for i, row in enumerate(rows):
                doc_id = row.get_attribute('data-docid') or str(i + 1)
                title_el = row.query_selector(NYSCEF_SELECTORS['doc_title'])
                date_el  = row.query_selector(NYSCEF_SELECTORS['doc_date'])
                title = title_el.inner_text().strip() if title_el else f"Document {i + 1}"
                date  = date_el.inner_text().strip()  if date_el  else ''
                doc_url = f"{_NYSCEF_BASE}/ViewDocument?docIndex={doc_id}&caseid={index_number}"
                entries.append(DocketEntry(
                    seq=doc_id,
                    title=title,
                    date=date,
                    source_url=doc_url,
                    source='nyscef',
                    extra={'nyscef_doc_id': doc_id},
                ))
        except Exception as e:
            logger.error(f"NYSCEF docket scrape failed: {e}")
        finally:
            page.close()
        return entries

    def download_document(self, entry: DocketEntry) -> Optional[Path]:
        """
        Download a NYSCEF document by replaying the browser session cookies
        via requests.Session (no browser overhead per-doc).
        """
        self._require_playwright()
        if not entry.source_url or not self._cookies:
            return None

        import requests
        s = requests.Session()
        s.headers['User-Agent'] = (
            "Mozilla/5.0 (compatible; Paperless-AI-Analyzer/1.0)"
        )
        for c in self._cookies:
            s.cookies.set(c['name'], c['value'], domain=c.get('domain', ''))

        try:
            resp = s.get(entry.source_url, timeout=60, stream=True)
            resp.raise_for_status()
            suffix = '.pdf'
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix,
                prefix=f"court_nyscef_{entry.seq}_"
            )
            for chunk in resp.iter_content(chunk_size=65536):
                tmp.write(chunk)
            tmp.flush()
            tmp.close()
            return Path(tmp.name)
        except Exception as e:
            logger.error(f"NYSCEF download failed for doc {entry.seq}: {e}")
            return None

    def _close_browser(self):
        """Clean up Playwright resources."""
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._context = None
        self._playwright = None
        self._authenticated = False

    def __del__(self):
        self._close_browser()
