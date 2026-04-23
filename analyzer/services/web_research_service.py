"""Web-research helpers used by the AI Chat flow.

Pure business logic — no Flask, no routes. Extracted verbatim from
`analyzer/routes/chat.py` during the 2026-04-23 maintainability refactor.
The only change is the function names (stripped the leading underscore);
all logic is preserved byte-for-byte.

Functions:
    load_session_web_context(session_id)           -> dict
    save_session_web_context(session_id, ctx)      -> None
    ddg_search(query, max_results=6)               -> list of {title, excerpt, url}
    resolve_court_docket_url(url)                  -> (text_content, error_message)
    fetch_url_text(url, max_chars=4000)            -> (text_content, error_message)

None of these touch `flask.request` or `flask.session`; they receive all
inputs as arguments and return values. Safe to import from anywhere.
"""
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Intent detection constants (used by api_chat to decide when to call search)
# --------------------------------------------------------------------------

SEARCH_INTENT_PHRASES = (
    'search online', 'search the web', 'search for', 'find online',
    'find more information', 'look up online', 'look this up',
    'google this', 'google for', 'search google', 'browse the web',
    'find information online', 'search the internet', 'go online',
    'find more about', 'research online', 'search news',
)


# --------------------------------------------------------------------------
# Session web-context persistence
# --------------------------------------------------------------------------

def load_session_web_context(session_id: str) -> dict:
    """Load persisted URL/search context for a chat session."""
    import json as _j
    try:
        from analyzer.db import _get_conn
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT web_context FROM chat_sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row and row['web_context']:
            return _j.loads(row['web_context']) or {}
    except Exception:
        pass
    return {}


def save_session_web_context(session_id: str, ctx: dict):
    """Persist URL/search context back to the session row."""
    import json as _j
    try:
        from analyzer.db import _get_conn
        with _get_conn() as conn:
            conn.execute(
                "UPDATE chat_sessions SET web_context=? WHERE id=?",
                (_j.dumps(ctx), session_id)
            )
    except Exception as e:
        logger.debug(f"save_session_web_context failed: {e}")


# --------------------------------------------------------------------------
# DuckDuckGo search (no API key)
# --------------------------------------------------------------------------

def ddg_search(query: str, max_results: int = 6) -> list:
    """
    DuckDuckGo web search — returns [{title, excerpt, url}].
    No API key required. Self-contained (no web_researcher import needed).
    """
    import re as _re
    import urllib.request as _ur
    import urllib.parse as _up
    try:
        encoded = _up.quote(query[:200])
        url = f'https://html.duckduckgo.com/html/?q={encoded}&kl=us-en'
        req = _ur.Request(url, headers={
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36'),
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        with _ur.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')

        link_re = _re.compile(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            _re.DOTALL | _re.IGNORECASE)
        snip_re = _re.compile(
            r'class="result__snippet"[^>]*>(.*?)</(?:span|td|div)',
            _re.DOTALL | _re.IGNORECASE)

        links    = link_re.findall(html)
        snippets = [_re.sub(r'<[^>]+>', '', s).strip() for s in snip_re.findall(html)]
        out = []
        snip_i = 0
        for raw_url, title_html in links:
            if len(out) >= max_results:
                break
            real_url = raw_url
            m = _re.search(r'uddg=([^&]+)', raw_url)
            if m:
                real_url = _up.unquote(m.group(1))
            if 'y.js?' in real_url or 'duckduckgo.com/y.js' in real_url:
                continue
            out.append({
                'title':   _re.sub(r'<[^>]+>', '', title_html).strip(),
                'excerpt': snippets[snip_i] if snip_i < len(snippets) else '',
                'url':     real_url,
            })
            snip_i += 1
        return out
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


# --------------------------------------------------------------------------
# Court docket URL resolution (Justia → CourtListener)
# --------------------------------------------------------------------------

def resolve_court_docket_url(url: str) -> tuple:
    """
    Detect Justia or PACER docket URLs and resolve them via CourtListener's
    free public API instead of scraping the blocked page directly.

    Returns (text_content, error_message) — same contract as fetch_url_text.
    Justia format: https://dockets.justia.com/docket/{state}/{court-id}/{case-num}/{pacer-id}
    """
    import re as _re
    import urllib.request as _ur
    import json as _json

    # Only handle known court docket hosts
    justia_pat = _re.compile(
        r'dockets\.justia\.com/docket/[^/]+/([a-z0-9]+)/([\w:%-]+?)(?:/(\d+))?/?$',
        _re.IGNORECASE
    )
    m = justia_pat.search(url)
    if not m:
        return ('', '')   # Not a Justia docket URL — caller should use regular fetch

    raw_court, raw_case, _pacer_id = m.groups()

    # Map Justia court code to CourtListener court slug:
    # nysdce → nysd  (strip trailing 'ce' civil / 'cr' criminal)
    cl_court = _re.sub(r'(ce|cr|bk|mj|mc|po|ap)$', '', raw_court.lower())

    # Normalise case number: '1:2025cv10573' → '1:25-cv-10573'
    case_num = (raw_case or '').replace('%3A', ':').replace('%2B', '+')
    cn_m = _re.match(r'^(\d+):(20)?(\d{2})(cv|cr|bk|mj|mc|po|ap)(\d+)$', case_num, _re.I)
    if cn_m:
        div, _, yr2, tp, num = cn_m.groups()
        case_num = f"{div}:{yr2}-{tp.lower()}-{num}"

    # Query CourtListener public search API (no auth required)
    search_url = (
        'https://www.courtlistener.com/api/rest/v4/search/'
        f'?type=d&docket_number={case_num}&court={cl_court}&page_size=1'
    )
    try:
        req = _ur.Request(search_url, headers={
            'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)'
        })
        with _ur.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode())
    except Exception as e:
        return ('', f'CourtListener lookup failed: {e}')

    results = data.get('results', [])
    if not results:
        # Broader fallback: search by last component of case number only
        num_only = case_num.split('-')[-1].lstrip('0') or case_num
        fallback_url = (
            f'https://www.courtlistener.com/api/rest/v4/search/'
            f'?type=d&q={num_only}&court={cl_court}&page_size=3'
        )
        try:
            req2 = _ur.Request(fallback_url, headers={
                'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)'
            })
            with _ur.urlopen(req2, timeout=10) as resp2:
                data = _json.loads(resp2.read().decode())
            results = data.get('results', [])
        except Exception:
            pass

    if not results:
        return ('', f'No results found in CourtListener for case {case_num} in {cl_court}')

    r = results[0]

    def _fmt_list(v):
        if isinstance(v, list):
            return ', '.join(str(x) for x in v[:10])
        return str(v) if v else ''

    lines = [
        f"COURT DOCKET — fetched via CourtListener (Justia was blocked)",
        f"",
        f"Case Name:     {r.get('caseName', '')}",
        f"Docket No.:    {r.get('docketNumber', '')}",
        f"Court:         {r.get('court', '')} ({r.get('court_citation_string', '')})",
        f"Filed:         {r.get('dateFiled', '')}",
        f"Judge:         {r.get('assignedTo', '')}",
        f"Cause:         {r.get('cause', '')}",
        f"Nature of Suit:{r.get('suitNature', '')}",
        f"Jurisdiction:  {r.get('jurisdictionType', '')}",
        f"Jury Demand:   {r.get('juryDemand', '')}",
        f"",
        f"Parties:       {_fmt_list(r.get('party', []))}",
        f"Attorneys:     {_fmt_list(r.get('attorney', []))}",
        f"Firms:         {_fmt_list(r.get('firm', []))}",
        f"",
        f"CourtListener: https://www.courtlistener.com{r.get('docket_absolute_url', '')}",
    ]
    return ('\n'.join(lines), '')


# --------------------------------------------------------------------------
# Generic URL text fetch
# --------------------------------------------------------------------------

def fetch_url_text(url: str, max_chars: int = 4000) -> tuple:
    """
    Fetch a URL and return (text_content, error_message).
    Strips HTML tags and truncates. Returns ('', error) on failure.
    """
    import re as _re
    try:
        import requests as _req
        resp = _req.get(url, timeout=12, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; PaperlessAI/3.5; +https://paperless-ai.local)',
        }, allow_redirects=True)
        resp.raise_for_status()
        raw = resp.text
    except Exception as e:
        return ('', str(e))

    # Strip script/style blocks, then all tags, collapse whitespace
    raw = _re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', ' ', raw,
                  flags=_re.DOTALL | _re.IGNORECASE)
    raw = _re.sub(r'<[^>]+>', ' ', raw)
    raw = _re.sub(r'[ \t]{2,}', ' ', raw)
    raw = _re.sub(r'\n{3,}', '\n\n', raw)
    raw = raw.strip()

    if not raw:
        return ('', 'Page fetched but contained no readable text (possibly JS-rendered).')

    if len(raw) > max_chars:
        raw = raw[:max_chars] + f'\n\n[Content truncated — {len(raw):,} chars total, showing first {max_chars:,}]'
    return (raw, '')
