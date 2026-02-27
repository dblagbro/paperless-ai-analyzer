"""
PACER direct connector (fallback when a document is not in RECAP).

Auth: Playwright headless Chromium (handles PrimeFaces/JSF login reliably).
     Falls back to requests-based AJAX if Playwright is unavailable.
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
_DEFAULT_RATE_LIMIT = 1.0

# Stealth args for Playwright (same pattern as NYSCEFConnector)
_STEALTH_ARGS = [
    '--disable-blink-features=AutomationControlled',
    '--no-sandbox',
    '--disable-dev-shm-usage',
]
_STEALTH_SCRIPT = 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
_STEALTH_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)


def _parse_pacer_docket_html(html: str, ecf_base: str) -> 'List[DocketEntry]':
    """
    Parse a CM/ECF docket report HTML page into DocketEntry objects.

    CM/ECF docket tables have rows of the form:
      <td>entry_num</td> | <td>MM/DD/YYYY</td> | <td>docket text + doc links</td>

    Document links look like: href="/doc1/{pacer_case_id}/{doc_seq}"
    """
    import re as _re
    from analyzer.court_connectors.base import DocketEntry

    entries: list = []
    date_pat = _re.compile(r'\b(\d{1,2}/\d{1,2}/\d{4})\b')
    doc_link_pat = _re.compile(r'href="(/doc1/[^"]+)"', _re.I)
    strip_tags = lambda s: _re.sub(r'<[^>]+>', ' ', s)

    # Split on <tr> boundaries (match full opening tag including attributes)
    for row_html in _re.split(r'<tr[^>]*>', html, flags=_re.I):
        date_m = date_pat.search(row_html)
        if not date_m:
            continue
        date_str = date_m.group(1)

        # Split into cells (match full opening tag including attributes)
        cells = _re.split(r'<td[^>]*>', row_html, flags=_re.I)
        if len(cells) < 3:
            continue

        # Entry number: look for a lone integer in the first 1–2 cells
        seq = ''
        for cell in cells[1:3]:
            ct = strip_tags(cell)
            ct = _re.sub(r'</.*', '', ct).strip()
            if _re.fullmatch(r'\d+', ct):
                seq = ct
                break

        # Description: largest / last cell, stripped of tags
        desc_cell = cells[-1] if len(cells) > 2 else ''
        desc = strip_tags(desc_cell)
        desc = _re.sub(r'\s+', ' ', desc).strip()
        # Drop leading noise (entry numbers, dates repeated)
        desc = _re.sub(r'^[\s\d/]+', '', desc).strip()

        # Document links
        doc_links = doc_link_pat.findall(row_html)
        source_url = (ecf_base + doc_links[0]) if doc_links else ''

        if not seq and not source_url:
            continue  # Not a real docket row

        # Normalise date to ISO (YYYY-MM-DD) so callers don't have to
        # know that PACER returns MM/DD/YYYY.
        try:
            import datetime as _ddt
            iso_date = _ddt.datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            iso_date = date_str  # keep whatever we got

        entries.append(DocketEntry(
            seq=seq,
            title=desc[:600],
            date=iso_date,
            source_url=source_url,
            source='pacer',
            doc_type='pdf',
            extra={'all_doc_urls': [ecf_base + l for l in doc_links]},
        ))

    return entries


def _check_playwright() -> bool:
    """Return True if Playwright and Chromium are available."""
    try:
        from playwright.sync_api import sync_playwright  # noqa
        return True
    except ImportError:
        return False


class PACERConnector(CourtConnector):
    """
    PACER direct connector for federal court documents not in the RECAP archive.

    Requires PACER username + password (stored encrypted in court_credentials).
    Optional: client_code in extra_config_json.

    Authentication uses Playwright headless Chromium when available (more reliable
    for the PrimeFaces/JSF login form), falling back to requests-based AJAX.
    """

    def __init__(self, project_slug: str, credentials: Dict[str, Any],
                 password: str = ''):
        super().__init__(project_slug, credentials)
        self._password = password
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': _STEALTH_UA})

        extra = {}
        try:
            import json
            raw = credentials.get('extra_config_json', '{}') or '{}'
            extra = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            pass

        self._client_code = extra.get('pacer_client_code', '')
        self._rate_limit = float(extra.get('pacer_rate_limit_seconds', _DEFAULT_RATE_LIMIT))
        # Court code/name for PACER login autocomplete (e.g. "nysb", "nysd", "ca9")
        self._pacer_login_court = extra.get('pacer_login_court', '')

    def authenticate(self) -> None:
        """
        Authenticate with PACER and capture the session cookie into self._session.
        Uses Playwright when available; falls back to requests-based AJAX.
        Raises RuntimeError if login fails.
        """
        username = self.credentials.get('username', '')
        if not username or not self._password:
            raise RuntimeError("PACER credentials (username + password) are required")

        if _check_playwright():
            self._authenticate_playwright(username)
        else:
            self._authenticate_requests(username)

    def _authenticate_playwright(self, username: str) -> None:
        """Log in to PACER via Playwright and extract session cookies into self._session."""
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True, args=_STEALTH_ARGS)
        context = browser.new_context(
            user_agent=_STEALTH_UA,
            viewport={'width': 1280, 'height': 720},
        )
        context.add_init_script(_STEALTH_SCRIPT)
        page = context.new_page()
        try:
            # Use 'load' to wait for JS-rendered PrimeFaces widgets (court select)
            page.goto(_LOGIN_URL, wait_until='load', timeout=30000)
            # Wait for the court select element specifically (PrimeFaces renders it via JS)
            try:
                page.wait_for_selector('select[name="loginForm:courtId"]', timeout=5000)
            except Exception:
                pass

            page.fill('input[name="loginForm:loginName"]', username)
            page.fill('input[name="loginForm:password"]', self._password)
            if self._client_code:
                page.fill('input[name="loginForm:clientCode"]', self._client_code)

            # PACER requires court selection.
            # The court field is PrimeFaces SelectOneMenu (widget_loginForm_courtId).
            # Strategy: open the dropdown, then use widget.selectItem() via JS so
            # PrimeFaces properly updates the underlying hidden <select> value.
            if self._pacer_login_court:
                court_set = False

                # Step 1: click the dropdown trigger to open the panel so PrimeFaces
                # renders the li items (needed before widget.items is populated).
                try:
                    trigger = page.query_selector(
                        '.ui-selectonemenu-trigger, '
                        '.ui-selectonemenu .ui-selectonemenu-label'
                    )
                    if trigger and trigger.is_visible():
                        trigger.click()
                        page.wait_for_timeout(600)
                        logger.debug("PACER: opened SelectOneMenu panel")
                except Exception:
                    pass

                # Step 2: use PrimeFaces widget.selectItem() to pick the right court.
                # widget.items is a jQuery collection of <li> elements in the open panel.
                try:
                    js_result = page.evaluate(
                        """
                        (term) => {
                            var widget = window.PF &&
                                         window.PF('widget_loginForm_courtId');
                            if (!widget) return 'no-widget';
                            if (!widget.items || !widget.items.length)
                                return 'no-items';
                            var termLow = term.toLowerCase();
                            var found = null;
                            widget.items.each(function() {
                                var t = this.textContent.trim().toLowerCase();
                                if (t && t !== '-- select one --' &&
                                        t.includes(termLow)) {
                                    found = this;
                                    return false; // break
                                }
                            });
                            if (found) {
                                widget.selectItem($(found));
                                return 'ok:' + found.textContent.trim();
                            }
                            var sample = [];
                            widget.items.each(function() {
                                var t = this.textContent.trim();
                                if (t && sample.length < 10) sample.push(t);
                            });
                            return 'no-match:' + sample.join(' | ');
                        }
                        """,
                        self._pacer_login_court,
                    )
                    logger.info(f"PACER court widget JS: {js_result}")
                    if js_result and js_result.startswith('ok:'):
                        court_set = True
                        page.wait_for_timeout(300)
                    elif js_result and js_result.startswith('no-match:'):
                        logger.warning(
                            f"PACER court '{self._pacer_login_court}' not found. "
                            f"Available: {js_result[9:]}"
                        )
                    else:
                        logger.warning(f"PACER court JS: {js_result}")
                except Exception as je:
                    logger.warning(f"PACER court widget JS failed: {je}")

                if not court_set:
                    logger.warning(
                        f"PACER court selection failed for '{self._pacer_login_court}'"
                    )

            page.click('[name="loginForm:fbtnLogin"]')

            # Wait for JS redirect away from login page (PrimeFaces AJAX triggers a JS redirect)
            try:
                page.wait_for_function(
                    "() => !window.location.href.includes('login.jsf')",
                    timeout=15000
                )
            except Exception:
                pass  # Timeout — check URL below

            # Still on login page → extract error message
            if 'login.jsf' in page.url:
                err_el = page.query_selector(
                    '.ui-messages-error-detail, .ui-message-error-detail, '
                    '[class*="error"] span'
                )
                err = err_el.inner_text().strip() if err_el else 'Login failed — check username and password'
                raise RuntimeError(
                    f"PACER login failed — {err}. "
                    "Re-enter your credentials in ⚙️ Manage Credentials."
                )

            # Inject Playwright session cookies into self._session for document downloads
            for c in context.cookies():
                self._session.cookies.set(c['name'], c['value'], domain=c.get('domain', ''))

            if ('PacerSession' not in self._session.cookies and
                    'NextGenCSO' not in self._session.cookies):
                raise RuntimeError(
                    "PACER login failed — credentials rejected (no session cookie). "
                    "Re-enter your PACER username and password in ⚙️ Manage Credentials."
                )

            self._authenticated = True
            logger.info("PACER Playwright authentication successful")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"PACER authentication error: {e}") from e
        finally:
            page.close()
            browser.close()
            pw.stop()

    def _authenticate_requests(self, username: str) -> None:
        """
        Fallback: requests-based AJAX login for when Playwright is unavailable.

        PACER uses PrimeFaces/Jakarta Faces — the login form requires a real
        ViewState token scraped from the page before submitting.
        """
        try:
            import re

            # Step 1: GET the login page to obtain the real ViewState token
            get_resp = self._session.get(_LOGIN_URL, timeout=30)
            get_resp.raise_for_status()

            vs_match = re.search(r'jakarta\.faces\.ViewState[^>]*value="([^"]+)"', get_resp.text)
            view_state = vs_match.group(1) if vs_match else 'stateless'
            logger.debug(f"PACER ViewState: {view_state[:20]}...")

            # Step 2: POST via PrimeFaces partial/ajax
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

            if 'Invalid' in resp.text or 'invalid' in resp.text:
                err_match = re.search(
                    r'ui-messages-error-detail[^>]*>([^<]+)<', resp.text
                )
                detail = err_match.group(1).strip() if err_match else 'Invalid username or password'
                raise RuntimeError(f"PACER login failed — {detail}. "
                                   "Re-enter your credentials in ⚙️ Manage Credentials.")

            redirect_match = re.search(r'<redirect\s+url="([^"]+)"', resp.text)
            if redirect_match:
                redirect_url = redirect_match.group(1)
                if redirect_url.startswith('/'):
                    from urllib.parse import urljoin
                    redirect_url = urljoin(_LOGIN_URL, redirect_url)
                logger.debug(f"Following PACER XML redirect to: {redirect_url[:80]}")
                self._session.get(redirect_url, timeout=30, allow_redirects=True)

            if ('PacerSession' not in self._session.cookies and
                    'NextGenCSO' not in self._session.cookies):
                self._session.get(_LOGIN_URL, timeout=30, allow_redirects=True)
                if ('PacerSession' not in self._session.cookies and
                        'NextGenCSO' not in self._session.cookies):
                    raise RuntimeError(
                        "PACER login failed — credentials rejected (no session cookie). "
                        "Re-enter your PACER username and password in ⚙️ Manage Credentials."
                    )

            self._authenticated = True
            logger.info("PACER requests-based authentication successful")
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

    def get_docket(self, case_id: str) -> List[DocketEntry]:
        """
        Fetch docket entries directly from PACER CM/ECF.

        case_id formats:
          - "{court_code}|{case_number}"                  e.g. "nysb|23-36006"
          - "{court_code}|{case_number}|{pacer_case_id}"  e.g. "nysb|23-36006|318848"

        When pacer_case_id is included, navigates directly to DktRpt.pl (preferred).
        Otherwise falls back to iquery.pl case search.
        """
        if '|' not in case_id:
            logger.warning(
                f"PACER get_docket: expected 'court|case_number[|pacer_case_id]' format, "
                f"got: {case_id!r}"
            )
            return []

        parts = case_id.split('|', 2)
        court_code = parts[0].lower().strip()
        case_number = parts[1] if len(parts) > 1 else ''
        pacer_case_id = parts[2].strip() if len(parts) > 2 else ''

        if not court_code or not case_number:
            return []

        self._ensure_authenticated()
        ecf_base = f"https://ecf.{court_code}.uscourts.gov"
        logger.info(
            f"PACER: fetching docket for {case_number} at {ecf_base}"
            + (f" (pacer_case_id={pacer_case_id})" if pacer_case_id else "")
        )

        if pacer_case_id and _check_playwright():
            return self._get_docket_direct(ecf_base, pacer_case_id, case_number)
        if _check_playwright():
            return self._get_docket_playwright(ecf_base, case_number)
        return self._get_docket_requests(ecf_base, case_number)

    def _get_docket_direct(
        self, ecf_base: str, pacer_case_id: str, case_number: str = ''
    ) -> List[DocketEntry]:
        """
        Navigate directly to DktRpt.pl using a known pacer_case_id.
        This avoids the iquery.pl case-search form entirely.
        """
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True, args=_STEALTH_ARGS)
        context = browser.new_context(
            user_agent=_STEALTH_UA,
            viewport={'width': 1280, 'height': 900},
        )
        context.add_init_script(_STEALTH_SCRIPT)

        for ck in self._session.cookies:
            try:
                context.add_cookies([{
                    'name':   ck.name,
                    'value':  ck.value,
                    'domain': ck.domain or '.uscourts.gov',
                    'path':   ck.path or '/',
                }])
            except Exception:
                pass

        page = context.new_page()
        entries: List[DocketEntry] = []
        try:
            dkt_url = f"{ecf_base}/cgi-bin/DktRpt.pl?{pacer_case_id}"
            page.goto(dkt_url, wait_until='domcontentloaded', timeout=30000)
            logger.info(f"PACER DktRpt: url={page.url!r} title={page.title()!r}")

            if 'login' in page.url.lower():
                logger.warning("PACER: DktRpt redirected to login — session expired")
                return []

            # If there's a submit button on the page, we're looking at an
            # options/filter form (date range, format, etc.) — submit it to
            # get the actual docket report.
            submit_btn = None
            for sel in [
                'input[type="submit"]',
                'button[type="submit"]',
                'input[value*="Run"]',
                'input[value*="Submit"]',
            ]:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    submit_btn = btn
                    break
            if submit_btn:
                # Clear date-range fields so PACER returns ALL entries, not
                # just the default "last 2 weeks" window.
                page.evaluate("""() => {
                    document.querySelectorAll('input[type="text"]').forEach(function(i) {
                        i.value = '';
                    });
                }""")
                logger.info("PACER: submitting docket options form (all dates)")
                submit_btn.click()
                # Use 'load' (all resources) rather than 'domcontentloaded' so
                # PACER has time to write the full docket HTML before we parse it.
                try:
                    page.wait_for_load_state('load', timeout=30000)
                except Exception:
                    page.wait_for_load_state('domcontentloaded', timeout=10000)

            import re as _re
            m = _re.match(r'(https?://[^/]+)', page.url)
            actual_base = m.group(1) if m else ecf_base
            entries = _parse_pacer_docket_html(page.content(), actual_base)
            logger.info(
                f"PACER: found {len(entries)} docket entries for {case_number or pacer_case_id}"
            )

        except Exception as e:
            logger.error(f"PACER docket direct failed for {case_number}: {e}")
        finally:
            page.close()
            browser.close()
            pw.stop()

        return entries

    def _get_docket_playwright(self, ecf_base: str, case_number: str) -> List[DocketEntry]:
        """
        Playwright-based CM/ECF docket fetch.

        Injects the authenticated PACER session cookies into a new browser context
        so that CM/ECF recognises the session without a full re-login.
        If the session has expired (redirect to PACER login), returns empty list
        with a warning — the user should re-test/save credentials to refresh the cookie.
        """
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True, args=_STEALTH_ARGS)
        context = browser.new_context(
            user_agent=_STEALTH_UA,
            viewport={'width': 1280, 'height': 900},
        )
        context.add_init_script(_STEALTH_SCRIPT)

        # Inject PACER session cookies — NextGenCSO has domain .uscourts.gov
        # which covers all ecf.*.uscourts.gov subdomains.
        for ck in self._session.cookies:
            try:
                domain = ck.domain or '.uscourts.gov'
                context.add_cookies([{
                    'name':   ck.name,
                    'value':  ck.value,
                    'domain': domain,
                    'path':   ck.path or '/',
                }])
            except Exception:
                pass

        page = context.new_page()
        entries: List[DocketEntry] = []
        try:
            # ── 1. Court CM/ECF home — SSO cookie authenticates automatically ──
            page.goto(ecf_base + '/', wait_until='domcontentloaded', timeout=30000)

            if 'login.uscourts.gov' in page.url or 'login.jsf' in page.url:
                logger.warning(
                    "PACER CM/ECF: session cookie expired — "
                    "re-test credentials in ⚙️ Manage Credentials to refresh the session."
                )
                return []

            # ── 2. Case query (iquery.pl) ─────────────────────────────────────
            page.goto(
                ecf_base + '/cgi-bin/iquery.pl?1-L_0_0-1',
                wait_until='domcontentloaded', timeout=30000,
            )

            if 'login' in page.url.lower():
                logger.warning("PACER: iquery.pl redirected to login")
                return []

            # ── 3. Fill in case number ────────────────────────────────────────
            filled = False
            for sel in [
                'input[name="case_num"]',
                'input[id="case_num"]',
                'input[name="case_no"]',
                'input[name="CaseNum"]',
                'input[name="case_number"]',
                'input[name="caseNumber"]',
            ]:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.fill(case_number)
                    filled = True
                    logger.debug(f"PACER iquery: filled {sel!r}")
                    break

            if not filled:
                # Fallback: first visible text input on the page
                for inp in page.query_selector_all('input[type="text"]'):
                    if inp.is_visible():
                        inp.fill(case_number)
                        filled = True
                        logger.debug("PACER iquery: used fallback text input")
                        break

            if not filled:
                logger.warning(
                    f"PACER: no case-number input found at {ecf_base}/cgi-bin/iquery.pl"
                )
                return []

            # ── 4. Submit ─────────────────────────────────────────────────────
            for sel in [
                'input[type="submit"][value*="Run"]',
                'input[type="submit"][value*="Query"]',
                'input[type="submit"][value*="Submit"]',
                'input[type="submit"]',
                'button[type="submit"]',
            ]:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    btn.click()
                    break
            try:
                page.wait_for_load_state('load', timeout=15000)
            except Exception:
                page.wait_for_load_state('domcontentloaded', timeout=10000)

            # ── 5. Navigate to docket report ──────────────────────────────────
            if 'DktRpt' not in page.url:
                dkt_link = page.query_selector('a[href*="DktRpt.pl"]')
                if not dkt_link:
                    logger.warning(
                        f"PACER: no docket link found for {case_number} — "
                        "case may not exist in this court's CM/ECF."
                    )
                    return []
                dkt_link.click()
                page.wait_for_load_state('domcontentloaded', timeout=30000)

            # ── 6. Handle PACER fee-receipt confirmation (if shown) ───────────
            if any(k in page.url.lower() for k in ('receipt', 'rcpt', 'confirm')):
                for sel in ['input[value*="View"]', 'input[type="submit"]', 'button']:
                    btn = page.query_selector(sel)
                    if btn and btn.is_visible():
                        btn.click()
                        page.wait_for_load_state('domcontentloaded', timeout=30000)
                        break

            # ── 7. Parse docket HTML ──────────────────────────────────────────
            import re as _re
            m = _re.match(r'(https?://[^/]+)', page.url)
            actual_base = m.group(1) if m else ecf_base
            entries = _parse_pacer_docket_html(page.content(), actual_base)
            logger.info(
                f"PACER: found {len(entries)} docket entries for {case_number}"
            )

        except Exception as e:
            logger.error(f"PACER docket (Playwright) failed for {case_number}: {e}")
        finally:
            page.close()
            browser.close()
            pw.stop()

        return entries

    def _get_docket_requests(self, ecf_base: str, case_number: str) -> List[DocketEntry]:
        """
        Requests-based CM/ECF docket fetch (when Playwright is unavailable).
        Navigates iquery.pl, finds a DktRpt link, fetches and parses the HTML.
        """
        import re as _re
        try:
            resp = self._session.get(
                ecf_base + '/cgi-bin/iquery.pl?1-L_0_0-1',
                timeout=30, allow_redirects=True,
            )
            if 'login' in resp.url.lower():
                logger.warning("PACER requests: session expired, cannot fetch docket")
                return []

            dkt_m = _re.search(r'href="(/cgi-bin/DktRpt\.pl\?[^"]+)"', resp.text)
            if not dkt_m:
                logger.warning(
                    f"PACER requests: no DktRpt link for {case_number} at {ecf_base}"
                )
                return []

            dkt_resp = self._session.get(
                ecf_base + dkt_m.group(1), timeout=60, allow_redirects=True
            )
            return _parse_pacer_docket_html(dkt_resp.text, ecf_base)
        except Exception as e:
            logger.error(f"PACER docket (requests) failed for {case_number}: {e}")
            return []

    def search_cases(self, case_number: str = '',
                     party_name: str = '',
                     court: str = '') -> List[CaseResult]:
        """
        PACER case search is complex and varies by court.
        For now, return empty — use CourtListenerConnector for search,
        PACER is used only for document downloads.
        """
        return []

    def download_document(self, entry: DocketEntry) -> Optional[Path]:
        """
        Download a single PACER document to a temp file.

        entry.source_url should be a full PACER CM/ECF doc URL.

        PACER serves a fee-receipt HTML confirmation page before delivering
        the actual PDF.  We handle this by sniffing the first bytes of the
        response: if HTML is detected we parse the form and POST it to obtain
        the real PDF.
        """
        if not entry.source_url:
            return None

        self._ensure_authenticated()

        # Respect rate limit
        time.sleep(self._rate_limit)

        try:
            resp = self._session.get(entry.source_url, timeout=60)
            resp.raise_for_status()

            content_type = resp.headers.get('Content-Type', '').lower()
            content = resp.content  # read into memory so we can sniff

            # Detect HTML by content-type OR magic bytes
            is_html = 'html' in content_type or content.lstrip()[:15].lower().startswith((b'<!doctype', b'<html'))

            if is_html:
                # PACER fee-receipt confirmation page requires browser-level
                # interaction.  Use Playwright to navigate, click the confirmation
                # button, and capture the PDF response.
                logger.info(f"PACER: fee-gate HTML for seq {entry.seq} — using Playwright to confirm")
                pdf_content = self._download_via_playwright(entry.source_url, str(entry.seq))
                if pdf_content is None:
                    logger.warning(f"PACER: Playwright download failed for seq {entry.seq}")
                    return None
                content = pdf_content

            suffix = '.pdf'
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix,
                prefix=f"court_pacer_{entry.seq}_"
            )
            tmp.write(content)
            tmp.flush()
            tmp.close()
            return Path(tmp.name)
        except Exception as e:
            logger.error(f"PACER download failed for seq {entry.seq}: {e}")
            return None

    def _download_via_playwright(self, doc_url: str, seq: str) -> Optional[bytes]:
        """
        Use Playwright to download a PACER document through the fee-receipt
        confirmation page that requires browser-level JavaScript.

        PACER flow:
          1. GET /doc1/<old_id>       → HTML fee-confirmation page (submit button)
          2. Click "View Document"    → browser navigates to /doc1/<new_id> (HTML wrapper)
          3. The HTML wrapper embeds  → /cgi-bin/show_temp.pl?file=<random>.pdf (actual PDF)

        Strategy:
          - Set up a response interceptor to capture the show_temp.pl PDF response.
          - Navigate to doc URL, click the submit button, wait for page+PDF to load.
          - Return the captured PDF bytes.  If response.body() fails (timing), fall back
            to context.request.get(pdf_url) which shares the browser's cookies.

        Returns the raw PDF bytes, or None on failure.
        """
        if not _check_playwright():
            return None
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True, args=_STEALTH_ARGS)
        context = browser.new_context(
            user_agent=_STEALTH_UA,
            viewport={'width': 1280, 'height': 900},
        )
        context.add_init_script(_STEALTH_SCRIPT)

        # Inject PACER session cookies so the browser is authenticated
        for ck in self._session.cookies:
            try:
                context.add_cookies([{
                    'name':   ck.name,
                    'value':  ck.value,
                    'domain': ck.domain or '.uscourts.gov',
                    'path':   ck.path or '/',
                }])
            except Exception:
                pass

        pdf_bytes: Optional[bytes] = None
        page = context.new_page()
        try:
            # Intercept responses; capture PDF bytes or URL for fallback download
            captured_pdf_url: list = []   # [(url, body_or_None), ...]

            def _handle_response(response):
                ct = response.headers.get('content-type', '').lower()
                if 'pdf' in ct:
                    try:
                        body = response.body()
                        if body and body[:4] == b'%PDF':
                            logger.debug(
                                f"PACER: intercepted PDF response "
                                f"{len(body)} bytes from {response.url!r}"
                            )
                            captured_pdf_url.append((response.url, body))
                        else:
                            captured_pdf_url.append((response.url, None))
                    except Exception:
                        captured_pdf_url.append((response.url, None))

            page.on('response', _handle_response)

            page.goto(doc_url, wait_until='domcontentloaded', timeout=30000)
            logger.debug(f"PACER Playwright: loaded {page.url!r} for seq {seq}")

            if 'login' in page.url.lower():
                logger.warning(
                    f"PACER: doc URL redirected to login for seq {seq} — session expired"
                )
                return None

            # Detect fee-confirmation page: PACER shows an intermediate receipt
            # page with a single "View Document" submit button before delivering
            # the actual PDF.
            submit_btn = None
            for sel in [
                'input[value="View Document"]',
                'input[value*="View"]',
                'input[value*="view"]',
                'input[type="submit"]',
                'button[type="submit"]',
            ]:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    submit_btn = el
                    break

            if submit_btn:
                original_url = page.url
                logger.info(
                    f"PACER: clicking submit on fee-confirmation page for seq {seq} "
                    f"(url={original_url!r})"
                )
                submit_btn.click()

                # Wait for navigation to the HTML wrapper page
                try:
                    page.wait_for_function(
                        f"() => window.location.href !== {repr(original_url)}",
                        timeout=20000,
                    )
                except Exception:
                    page.wait_for_load_state('domcontentloaded', timeout=20000)

                # Wait for the embedded PDF resource to be requested and received
                try:
                    page.wait_for_load_state('networkidle', timeout=10000)
                except Exception:
                    page.wait_for_timeout(2000)

                new_url = page.url
                logger.info(f"PACER: page at {new_url!r} for seq {seq}")
            else:
                # No fee gate — page may already have embedded PDF
                logger.debug(f"PACER: no fee-confirmation button on {page.url!r} for seq {seq}")
                try:
                    page.wait_for_load_state('networkidle', timeout=5000)
                except Exception:
                    pass

            # ── Resolve the PDF ─────────────────────────────────────────────
            # Priority 1: captured body from response interceptor
            for pdf_url, body in captured_pdf_url:
                if body and body[:4] == b'%PDF':
                    pdf_bytes = body
                    logger.info(
                        f"PACER: got {len(pdf_bytes)} PDF bytes "
                        f"(intercepted) for seq {seq}"
                    )
                    break

            # Priority 2: download from captured PDF URL using context.request
            # (shares browser cookies/session, so PACER temp file is accessible)
            if not pdf_bytes and captured_pdf_url:
                for pdf_url, _ in captured_pdf_url:
                    logger.info(
                        f"PACER: downloading PDF from {pdf_url!r} "
                        f"via context.request for seq {seq}"
                    )
                    try:
                        api_resp = context.request.get(pdf_url, timeout=60000)
                        logger.info(
                            f"PACER: context.request status={api_resp.status} "
                            f"ct={api_resp.headers.get('content-type','?')!r} "
                            f"for seq {seq}"
                        )
                        if api_resp.ok:
                            body = api_resp.body()
                            if body and body[:4] == b'%PDF':
                                pdf_bytes = body
                                logger.info(
                                    f"PACER: got {len(pdf_bytes)} PDF bytes "
                                    f"(context.request) for seq {seq}"
                                )
                                break
                            else:
                                logger.warning(
                                    f"PACER: context.request body not PDF for seq {seq}: "
                                    f"{len(body) if body else 0} bytes, "
                                    f"start={(body[:20] if body else b'')!r}"
                                )
                        else:
                            logger.warning(
                                f"PACER: context.request returned {api_resp.status} "
                                f"for {pdf_url!r}, seq {seq}"
                            )
                    except Exception as re_err:
                        logger.warning(
                            f"PACER: context.request failed for {pdf_url!r}: {re_err}"
                        )

            if not pdf_bytes:
                logger.warning(
                    f"PACER: Playwright found no PDF for seq {seq} "
                    f"(captured_pdf_url={[u for u,_ in captured_pdf_url]})"
                )

        except Exception as e:
            logger.error(f"PACER: Playwright download failed for seq {seq}: {e}")
        finally:
            page.close()
            browser.close()
            pw.stop()

        return pdf_bytes if pdf_bytes else None
