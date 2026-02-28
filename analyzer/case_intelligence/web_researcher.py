"""
Web Research module for Case Intelligence.

Provides web-based legal research to augment the CI pipeline with live sources
beyond what the LLM was trained on.

Free sources (no API key required):
  - CourtListener     : Federal case law opinions and party history
  - Harvard Caselaw   : Comprehensive US case law corpus (6.7 M cases)
  - DuckDuckGo        : General web search for news and background
  - GDELT             : Global news event and entity mentions (real-time)
  - BOP Inmate Locator: Bureau of Prisons federal inmate search
  - OFAC Sanctions    : Treasury SDN / Consolidated Sanctions list
  - SEC EDGAR         : Securities filings and enforcement actions
  - FEC OpenData      : Campaign finance contributions (free key from api.open.fec.gov)

Optional paid sources (require API key in web_research_config):
  Web Search / AI:
  - Brave Search      : Independent web index ($5/1,000 queries)
  - Google CSE        : Google web search (100/day free, $5/1,000 beyond)
  - Exa AI            : Neural/semantic search ($7/1,000 queries)
  - Perplexity Sonar  : AI-synthesized answers with citations (~$5-14/1,000)
  - Tavily            : AI-optimized search (free tier: 1,000/mo)
  - Serper.dev        : Google results API ($50/50k queries)

  News:
  - NewsAPI           : 150,000-source news archive ($449/month business)

  Court Dockets:
  - Docket Alarm      : 675M+ federal and state dockets ($99/month flat)
  - UniCourt          : Normalized federal + state court data ($49-299/month)

  Public Records / Background:
  - OpenSanctions     : Consolidated sanctions/PEP entity database (~€0.10/call)
  - OpenCorporates    : 200M+ global business entity records (paid key)

  Enterprise Legal:
  - Lexis-Nexis       : Enterprise case law and research API
  - vLex              : Global case law from 100+ countries (subscription)
  - Westlaw Edge      : Thomson Reuters flagship legal research (enterprise)
  - CLEAR             : Thomson Reuters comprehensive background intelligence (enterprise)

Role-aware behavior:
  - defense     : queries skew toward dismissal, suppression, impeachment material
  - plaintiff   : queries skew toward prior bad acts, civil judgments, pattern evidence
  - prosecution : queries skew toward prior convictions, criminal history, modus operandi
"""

import json
import logging
import time
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# ── Rate limits (min seconds between calls per source) ───────────────────────
_RATE = {
    'courtlistener': 1.2,
    'caselaw':       0.8,
    'ddg':           2.5,
    'brave':         0.5,
    'google_cse':    0.3,
    'exa':           0.5,
    'perplexity':    1.5,
    'gdelt':         1.2,
    'bop':           3.0,
    'ofac':          1.0,
    'fec':           1.2,
    'sec_edgar':     1.0,
    'opensanctions': 0.8,
    'opencorporates':1.0,
    'newsapi':       0.5,
    'docket_alarm':  2.0,
    'unicourt':      2.0,
    'vlex':          1.0,
    'westlaw':       1.0,
    'clear':         2.0,
    'tavily':        0.5,
    'serper':        0.5,
    'lexisnexis':    1.0,
}

# CourtListener court abbreviations for US states
_STATE_TO_CL: Dict[str, str] = {
    'alabama': 'ala', 'alaska': 'alaska', 'arizona': 'ariz', 'arkansas': 'ark',
    'california': 'cal', 'colorado': 'colo', 'connecticut': 'conn', 'delaware': 'del',
    'florida': 'fla', 'georgia': 'ga', 'hawaii': 'haw', 'idaho': 'idaho',
    'illinois': 'ill', 'indiana': 'ind', 'iowa': 'iowa', 'kansas': 'kan',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'mass', 'michigan': 'mich', 'minnesota': 'minn',
    'mississippi': 'miss', 'missouri': 'mo', 'montana': 'mont', 'nebraska': 'neb',
    'nevada': 'nev', 'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm',
    'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'ohio',
    'oklahoma': 'okla', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri',
    'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tenn', 'texas': 'tex',
    'utah': 'utah', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wash',
    'west virginia': 'wva', 'wisconsin': 'wis', 'wyoming': 'wyo',
}

# Role-aware query prefixes for legal authority search
_ROLE_AUTHORITY_PREFIX: Dict[str, str] = {
    'plaintiff':    'plaintiff prevailing similar facts judgment damages',
    'prosecution':  'prosecution successful conviction precedent element proof',
    'defense':      'defense prevailing acquittal dismissal suppression reversal',
    'neutral':      'relevant case law precedent',
}

# Role-aware entity character research keywords
_ROLE_ENTITY_PERSON: Dict[str, str] = {
    'plaintiff':    'credibility reliability expert bias civil history',
    'prosecution':  'prior conviction criminal history arrest modus operandi',
    'defense':      'impeachment false testimony inconsistency credibility bias financial motivation',
    'neutral':      'background history credibility',
}
_ROLE_ENTITY_ORG: Dict[str, str] = {
    'plaintiff':    'prior complaints regulatory violations consumer fraud lawsuit',
    'prosecution':  'prior offenses regulatory action criminal enterprise pattern',
    'defense':      'reputation litigation history regulatory compliance',
    'neutral':      'lawsuit regulatory history',
}


class WebResearcher:
    """
    Orchestrates web-based legal research across multiple free and paid sources.

    Config dict keys (passed as web_research_config in CI run):
      --- Free / no key ---
      courtlistener    bool  CourtListener opinions search (default True)
      caselaw_api      bool  Harvard Caselaw API (default True)
      general_search   bool  DuckDuckGo web search (default True)
      entity_research  bool  Research named entities (default True)
      gdelt_news       bool  GDELT global news (default True)
      bop_search       bool  BOP federal inmate search for persons (default True)
      ofac_search      bool  OFAC/Treasury sanctions check (default True)
      sec_edgar        bool  SEC EDGAR filings search (default True)
      fec_key          str   FEC OpenFEC API key (free from api.open.fec.gov)
      --- Web search (paid keys) ---
      brave_key        str   Brave Search API key ($5/1,000 queries)
      google_cse_key   str   Google Custom Search API key (100/day free)
      google_cse_cx    str   Google CSE search engine ID (cx parameter)
      exa_key          str   Exa AI neural search key ($7/1,000 queries)
      perplexity_key   str   Perplexity Sonar key (~$5-14/1,000 + tokens)
      tavily_key       str   Tavily AI search key (free tier: 1,000/mo)
      serper_key       str   Serper.dev Google results key ($50/50k)
      --- News (paid) ---
      newsapi_key      str   NewsAPI.org key ($449/mo business)
      --- Court dockets (paid) ---
      docket_alarm_user str  Docket Alarm username ($99/mo flat)
      docket_alarm_pass str  Docket Alarm password
      unicourt_id      str   UniCourt OAuth client ID ($49-299/mo)
      unicourt_secret  str   UniCourt OAuth client secret
      --- Public records (paid) ---
      opensanctions_key str  OpenSanctions API key (~€0.10/call)
      opencorporates_key str OpenCorporates API key (paid, business entity)
      --- Enterprise legal ---
      lexisnexis_key   str   Lexis-Nexis enterprise API key
      vlex_key         str   vLex API key (subscription)
      westlaw_key      str   Westlaw Edge API key (Thomson Reuters enterprise)
      clear_key        str   CLEAR by Thomson Reuters key (enterprise)
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self._last_call: Dict[str, float] = {}
        self._docket_alarm_token: Optional[str] = None
        self._unicourt_token: Optional[str] = None

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _throttle(self, source: str):
        limit = _RATE.get(source, 1.0)
        elapsed = time.monotonic() - self._last_call.get(source, 0.0)
        if elapsed < limit:
            time.sleep(limit - elapsed)
        self._last_call[source] = time.monotonic()

    # ── Public API ────────────────────────────────────────────────────────────

    def search_legal_authorities(
        self,
        query: str,
        jurisdiction: str = None,
        role: str = 'neutral',
        max_results: int = 8,
    ) -> List[Dict]:
        """Search for relevant case law and legal authorities across all configured sources."""
        prefix = _ROLE_AUTHORITY_PREFIX.get(role, _ROLE_AUTHORITY_PREFIX['neutral'])
        enriched = f"{prefix} {query[:250]}"
        results: List[Dict] = []

        # Free case law
        if self.config.get('courtlistener', True):
            results.extend(self._cl_opinions(enriched, jurisdiction,
                                             max_results=min(5, max_results)))
        if self.config.get('caselaw_api', True):
            results.extend(self._caselaw_search(enriched, jurisdiction,
                                                max_results=min(5, max_results)))

        # Court dockets (paid)
        if self.config.get('docket_alarm_user') and self.config.get('docket_alarm_pass'):
            results.extend(self._docket_alarm(enriched, max_results=3))
        if self.config.get('unicourt_id') and self.config.get('unicourt_secret'):
            results.extend(self._unicourt_party(enriched))

        # AI/web search for legal content
        if self.config.get('brave_key'):
            results.extend(self._brave(f'legal precedent {enriched}', max_results=3))
        if self.config.get('google_cse_key') and self.config.get('google_cse_cx'):
            results.extend(self._google_cse(f'case law {enriched}', max_results=3))
        if self.config.get('exa_key'):
            results.extend(self._exa(f'legal case {enriched}', max_results=3))
        if self.config.get('perplexity_key'):
            results.extend(self._perplexity(f'Relevant legal precedents for: {query}'))
        if self.config.get('tavily_key'):
            results.extend(self._tavily(f'legal case law {enriched}', max_results=3))
        if self.config.get('serper_key'):
            results.extend(self._serper(f'legal precedent {enriched}', max_results=3))

        # Enterprise legal
        if self.config.get('lexisnexis_key'):
            results.extend(self._lexisnexis(query, jurisdiction, max_results=5))
        if self.config.get('vlex_key'):
            results.extend(self._vlex(enriched, jurisdiction, max_results=5))
        if self.config.get('westlaw_key'):
            results.extend(self._westlaw(enriched, jurisdiction, max_results=5))

        return self._dedup(results)[:max_results]

    def research_entity(
        self,
        name: str,
        entity_type: str = 'person',
        role_in_case: str = '',
        run_role: str = 'neutral',
    ) -> Dict:
        """
        Research a named person or organization for background, court history,
        public records, and character information relevant to the litigation role.
        """
        result: Dict[str, Any] = {
            'name': name,
            'entity_type': entity_type,
            'court_history': [],
            'news_mentions': [],
            'public_records': [],
            'summary': '',
        }

        if not self.config.get('entity_research', True):
            return result

        is_person = entity_type.lower() in (
            'person', 'individual', 'plaintiff', 'defendant', 'witness',
            'attorney', 'judge', 'expert', 'trustee', 'receiver',
        )

        # ── Court history ─────────────────────────────────────────────────────
        if self.config.get('courtlistener', True):
            result['court_history'].extend(self._cl_party_search(name))
        if self.config.get('unicourt_id') and self.config.get('unicourt_secret'):
            result['court_history'].extend(self._unicourt_party(name))
        if self.config.get('docket_alarm_user') and self.config.get('docket_alarm_pass'):
            result['court_history'].extend(self._docket_alarm(f'"{name}"', max_results=3))

        # ── Free public records ───────────────────────────────────────────────
        if is_person and self.config.get('bop_search', True):
            result['public_records'].extend(self._bop_inmate(name))
        if self.config.get('ofac_search', True):
            result['public_records'].extend(self._ofac_search(name))
        if self.config.get('opensanctions_key'):
            result['public_records'].extend(self._opensanctions(name))
        if self.config.get('sec_edgar', True):
            result['public_records'].extend(
                self._sec_edgar(f'"{name}"', max_results=3))
        if is_person and self.config.get('fec_key'):
            result['public_records'].extend(self._fec_contributions(name))
        if not is_person and self.config.get('opencorporates_key'):
            result['public_records'].extend(self._opencorporates(name))
        if self.config.get('clear_key'):
            result['public_records'].extend(self._clear(name))

        # ── Web news / background ─────────────────────────────────────────────
        if self.config.get('general_search', True):
            kw = (_ROLE_ENTITY_PERSON if is_person else _ROLE_ENTITY_ORG).get(
                run_role, 'background history credibility')
            web_q = f'"{name}" {kw}'
            result['news_mentions'].extend(self._ddg(web_q, max_results=4))
        if self.config.get('gdelt_news', True):
            result['news_mentions'].extend(self._gdelt_news(f'"{name}"', max_results=4))
        if self.config.get('newsapi_key'):
            result['news_mentions'].extend(self._newsapi(f'"{name}"', max_results=3))
        if self.config.get('brave_key'):
            kw = (_ROLE_ENTITY_PERSON if is_person else _ROLE_ENTITY_ORG).get(
                run_role, 'background')
            result['news_mentions'].extend(self._brave(f'"{name}" {kw}', max_results=3))
        if self.config.get('google_cse_key') and self.config.get('google_cse_cx'):
            result['news_mentions'].extend(self._google_cse(f'"{name}"', max_results=2))
        if self.config.get('exa_key'):
            result['news_mentions'].extend(self._exa(f'"{name}"', max_results=2))
        if self.config.get('tavily_key'):
            result['news_mentions'].extend(
                self._tavily(f'"{name}" {kw if "kw" in dir() else ""}', max_results=2))
        if self.config.get('serper_key'):
            result['news_mentions'].extend(self._serper(f'"{name}"', max_results=2))

        # ── Plain-text summary for prompt injection ───────────────────────────
        parts: List[str] = []
        if result['court_history']:
            cases = [
                f"{c.get('case_name', '?')} ({(c.get('date') or '')[:4]})"
                for c in result['court_history'][:3]
            ]
            parts.append(f"Court records: {'; '.join(cases)}")
        if result['public_records']:
            flags = [r.get('title') or r.get('name', '') for r in result['public_records'][:3]
                     if (r.get('title') or r.get('name'))]
            if flags:
                parts.append(f"Public records: {'; '.join(flags)}")
        if result['news_mentions']:
            headlines = [m.get('title', '') for m in result['news_mentions'][:2]
                         if m.get('title')]
            if headlines:
                parts.append(f"Web: {'; '.join(headlines)}")
        result['summary'] = ' | '.join(parts)
        return result

    def search_general(self, query: str, max_results: int = 5) -> List[Dict]:
        """General web search for news and public information."""
        results: List[Dict] = []
        if self.config.get('general_search', True):
            results.extend(self._ddg(query, max_results=max_results))
        if self.config.get('gdelt_news', True):
            results.extend(self._gdelt_news(query, max_results=5))
        if self.config.get('brave_key'):
            results.extend(self._brave(query, max_results=3))
        if self.config.get('google_cse_key') and self.config.get('google_cse_cx'):
            results.extend(self._google_cse(query, max_results=3))
        if self.config.get('exa_key'):
            results.extend(self._exa(query, max_results=3))
        if self.config.get('perplexity_key'):
            results.extend(self._perplexity(query))
        if self.config.get('newsapi_key'):
            results.extend(self._newsapi(query, max_results=3))
        if self.config.get('tavily_key'):
            results.extend(self._tavily(query, max_results=3))
        if self.config.get('serper_key'):
            results.extend(self._serper(query, max_results=3))
        return self._dedup(results)[:max_results]

    # ── CourtListener ─────────────────────────────────────────────────────────

    def _cl_opinions(self, query: str, jurisdiction: str,
                     max_results: int = 5) -> List[Dict]:
        """Search CourtListener for case opinions (free, no key required)."""
        self._throttle('courtlistener')
        try:
            params: Dict[str, Any] = {
                'type': 'o',
                'q': query[:300],
                'stat_Precedential': 'on',
                'order_by': 'score desc',
            }
            cl_court = self._jur_to_cl(jurisdiction)
            if cl_court:
                params['court'] = cl_court
            data = _http_get('https://www.courtlistener.com/api/rest/v4/search/',
                             params=params, timeout=12)
            if not data:
                return []
            out = []
            for item in (data.get('results') or [])[:max_results]:
                out.append({
                    'citation':       item.get('citation', item.get('caseName', 'Unknown')),
                    'title':          item.get('caseName', ''),
                    'court':          item.get('court', ''),
                    'date':           (item.get('dateFiled') or '')[:10],
                    'excerpt':        (item.get('snippet') or '')[:500],
                    'url':            'https://www.courtlistener.com' + item.get('absolute_url', ''),
                    'source':         'courtlistener',
                    'authority_type': 'binding' if 'scotus' in str(item.get('court', '')).lower()
                                      else 'persuasive',
                    'reliability':    'official',
                })
            return out
        except Exception as e:
            logger.warning(f"CourtListener opinions search failed: {e}")
            return []

    def _cl_party_search(self, name: str) -> List[Dict]:
        """Search CourtListener for federal cases naming a specific party."""
        self._throttle('courtlistener')
        try:
            data = _http_get('https://www.courtlistener.com/api/rest/v4/search/',
                             params={'type': 'o', 'q': f'"{name}"',
                                     'order_by': 'score desc'},
                             timeout=12)
            if not data:
                return []
            out = []
            for item in (data.get('results') or [])[:5]:
                out.append({
                    'case_name': item.get('caseName', ''),
                    'court':     item.get('court', ''),
                    'date':      (item.get('dateFiled') or '')[:10],
                    'excerpt':   (item.get('snippet') or '')[:300],
                    'url':       'https://www.courtlistener.com' + item.get('absolute_url', ''),
                    'source':    'courtlistener',
                })
            return out
        except Exception as e:
            logger.warning(f"CourtListener party search failed: {e}")
            return []

    # ── Harvard Caselaw Access Project ────────────────────────────────────────

    def _caselaw_search(self, query: str, jurisdiction: str,
                        max_results: int = 5) -> List[Dict]:
        """Search the Harvard Caselaw Access Project (free, no key required)."""
        self._throttle('caselaw')
        try:
            params: Dict[str, Any] = {
                'search':    query[:200],
                'page_size': min(max_results, 10),
                'full_case': 'false',
            }
            jur = self._jur_to_caselaw(jurisdiction)
            if jur:
                params['jurisdiction'] = jur
            data = _http_get('https://api.case.law/v1/cases/', params=params, timeout=12)
            if not data:
                return []
            out = []
            for item in (data.get('results') or [])[:max_results]:
                citations = item.get('citations') or []
                cite = citations[0].get('cite', '') if citations else ''
                out.append({
                    'citation':       cite or item.get('name', 'Unknown'),
                    'title':          item.get('name', ''),
                    'court':          (item.get('court') or {}).get('name', ''),
                    'date':           (item.get('decision_date') or '')[:10],
                    'url':            item.get('url', ''),
                    'source':         'caselaw_access',
                    'authority_type': 'persuasive',
                    'reliability':    'official',
                    'excerpt':        '',
                })
            return out
        except Exception as e:
            logger.warning(f"Caselaw API search failed: {e}")
            return []

    # ── DuckDuckGo ────────────────────────────────────────────────────────────

    def _ddg(self, query: str, max_results: int = 5) -> List[Dict]:
        """General web search via DuckDuckGo HTML (free, no API key)."""
        self._throttle('ddg')
        try:
            encoded = urllib.parse.quote(query[:200])
            url = f'https://html.duckduckgo.com/html/?q={encoded}&kl=us-en'
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                                   'Chrome/120.0.0.0 Safari/537.36'),
                    'Accept': 'text/html,application/xhtml+xml',
                    'Accept-Language': 'en-US,en;q=0.9',
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode('utf-8', errors='replace')

            link_re = re.compile(
                r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                re.DOTALL | re.IGNORECASE)
            snip_re = re.compile(
                r'class="result__snippet"[^>]*>(.*?)</(?:span|td|div)',
                re.DOTALL | re.IGNORECASE)

            links    = link_re.findall(html)
            snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snip_re.findall(html)]
            out: List[Dict] = []
            snip_idx = 0
            for raw_url, title_html in links:
                if len(out) >= max_results:
                    break
                real_url = raw_url
                uddg_m = re.search(r'uddg=([^&]+)', raw_url)
                if uddg_m:
                    real_url = urllib.parse.unquote(uddg_m.group(1))
                if 'y.js?' in real_url or '/y.js' in real_url or 'duckduckgo.com/y.js' in real_url:
                    continue
                out.append({
                    'title':   re.sub(r'<[^>]+>', '', title_html).strip(),
                    'excerpt': snippets[snip_idx] if snip_idx < len(snippets) else '',
                    'url':     real_url,
                    'source':  'web_search',
                })
                snip_idx += 1
            return out
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []

    # ── GDELT Global News ─────────────────────────────────────────────────────

    def _gdelt_news(self, query: str, max_results: int = 8) -> List[Dict]:
        """Search GDELT global news events (free, no key required)."""
        self._throttle('gdelt')
        try:
            q = urllib.parse.quote(f'{query[:150]} sourcelang:english')
            url = (f'https://api.gdeltproject.org/api/v2/doc/doc'
                   f'?query={q}&mode=artlist&maxrecords={min(max_results, 25)}'
                   f'&format=json&sort=DateDesc')
            data = _http_get(url, timeout=12)
            if not data:
                return []
            out = []
            for art in (data.get('articles') or [])[:max_results]:
                out.append({
                    'title':   art.get('title', ''),
                    'excerpt': '',
                    'url':     art.get('url', ''),
                    'source':  'gdelt_news',
                    'date':    (art.get('seendate') or '')[:8],
                    'domain':  art.get('domain', ''),
                })
            return out
        except Exception as e:
            logger.warning(f"GDELT news search failed: {e}")
            return []

    # ── BOP Federal Inmate Locator ────────────────────────────────────────────

    def _bop_inmate(self, name: str) -> List[Dict]:
        """Search Bureau of Prisons inmate locator (free, no key required).
        Returns federal inmates matching the name from 1982 to present."""
        self._throttle('bop')
        try:
            parts = name.strip().split()
            first = parts[0] if parts else ''
            last  = parts[-1] if len(parts) > 1 else ''
            if not last:
                return []
            params = {
                'todo':          'query',
                'output':        'json',
                'nameFirst':     first[:25],
                'nameLast':      last[:25],
                'inmateNumType': 'IRN',
            }
            data = _http_get(
                'https://www.bop.gov/PublicInfo/execute/inmateloc',
                params=params, timeout=15)
            if not data:
                return []
            rows = (data.get('InmateLocator') or {}).get('row') or []
            if isinstance(rows, dict):
                rows = [rows]
            out = []
            for r in rows[:5]:
                inmate_num = r.get('inmateNum', '')
                facility   = r.get('faclName', '')
                rel_date   = r.get('projRelDate') or r.get('actRelDate', 'UNKNOWN')
                full_name  = f"{r.get('nameFirst','')} {r.get('nameLast','')}".strip()
                out.append({
                    'name':    full_name,
                    'title':   f'BOP Federal Inmate: {full_name} #{inmate_num}',
                    'excerpt': f'Facility: {facility}. Projected release: {rel_date}',
                    'url':     'https://www.bop.gov/inmateloc/',
                    'source':  'bop_inmate',
                    'flag':    'federal_inmate',
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"BOP inmate search failed: {e}")
            return []

    # ── OFAC Sanctions ────────────────────────────────────────────────────────

    def _ofac_search(self, name: str) -> List[Dict]:
        """Search OFAC/Treasury Consolidated Sanctions list (free, no key required)."""
        self._throttle('ofac')
        try:
            # Try OFAC sanctions search API (unofficial web endpoint)
            params = {
                'search':     name[:80],
                'program':    '',
                'listType':   'SDN',
                'score':      '85',
                'searchType': 'Combined',
            }
            data = _http_get(
                'https://sanctionssearch.ofac.treas.gov/api/Search/GetSearchResults',
                params=params, timeout=12)
            results = []
            if data:
                items = data.get('SDNList') or data.get('results') or data.get('matches') or []
                for item in items[:5]:
                    entity_name = (item.get('name') or item.get('firstName', '') + ' ' +
                                   item.get('lastName', '')).strip()
                    program     = item.get('program', item.get('remarks', ''))
                    results.append({
                        'name':    entity_name,
                        'title':   f'OFAC SDN Match: {entity_name}',
                        'excerpt': f'Sanctions program: {program}',
                        'url':     'https://sanctionssearch.ofac.treas.gov/',
                        'source':  'ofac_sanctions',
                        'flag':    'sanctions',
                        'reliability': 'official',
                    })
            return results
        except Exception as e:
            logger.debug(f"OFAC search failed (non-critical): {e}")
            return []

    # ── SEC EDGAR ─────────────────────────────────────────────────────────────

    def _sec_edgar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search SEC EDGAR full-text filing search (free, no key required)."""
        self._throttle('sec_edgar')
        try:
            params = {
                'q':         query[:200],
                'dateRange': 'custom',
                'startdt':   '2000-01-01',
                'forms':     '8-K,10-K,DEF 14A,S-1,10-Q',
            }
            data = _http_get(
                'https://efts.sec.gov/LATEST/search-index',
                params=params, timeout=12)
            if not data:
                return []
            hits = (data.get('hits') or {}).get('hits') or []
            out = []
            for h in hits[:max_results]:
                src = h.get('_source') or {}
                entity = src.get('entity_name', '')
                form   = src.get('form_type', '')
                date   = src.get('file_date', '')[:10]
                cik    = src.get('entity_id', '')
                out.append({
                    'title':   f'SEC EDGAR {form}: {entity}',
                    'excerpt': f'Filed: {date}',
                    'url':     f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}',
                    'source':  'sec_edgar',
                    'date':    date,
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"SEC EDGAR search failed: {e}")
            return []

    # ── FEC Campaign Finance ──────────────────────────────────────────────────

    def _fec_contributions(self, name: str) -> List[Dict]:
        """Search FEC campaign finance contributions (free API key from api.open.fec.gov)."""
        self._throttle('fec')
        try:
            api_key = self.config.get('fec_key', 'DEMO_KEY')
            params = {
                'contributor_name': name[:80],
                'api_key':          api_key,
                'per_page':         5,
                'sort':             '-contribution_receipt_date',
            }
            data = _http_get(
                'https://api.open.fec.gov/v1/schedules/schedule_a/',
                params=params, timeout=12)
            if not data:
                return []
            results = data.get('results') or []
            out = []
            for r in results[:5]:
                amount     = r.get('contribution_receipt_amount', 0)
                committee  = r.get('committee', {}).get('name', '')
                date       = (r.get('contribution_receipt_date') or '')[:10]
                contributor= r.get('contributor_name', name)
                out.append({
                    'title':   f'FEC Contribution: ${amount:,.0f} to {committee}',
                    'excerpt': f'{contributor} contributed ${amount:,.0f} on {date}',
                    'url':     'https://www.fec.gov/data/receipts/individual-contributions/',
                    'source':  'fec_contributions',
                    'date':    date,
                    'flag':    'campaign_finance',
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"FEC contributions search failed: {e}")
            return []

    # ── Brave Search ─────────────────────────────────────────────────────────

    def _brave(self, query: str, max_results: int = 5) -> List[Dict]:
        """Web search via Brave Search API ($5/1,000 queries)."""
        api_key = self.config.get('brave_key', '')
        if not api_key:
            return []
        self._throttle('brave')
        try:
            data = _http_get(
                'https://api.search.brave.com/res/v1/web/search',
                params={'q': query[:400], 'count': min(max_results, 20)},
                timeout=12,
                extra_headers={'X-Subscription-Token': api_key,
                               'Accept-Encoding': 'gzip'},
            )
            if not data:
                return []
            return [{
                'title':   r.get('title', ''),
                'excerpt': r.get('description', ''),
                'url':     r.get('url', ''),
                'source':  'brave_search',
            } for r in (data.get('web', {}).get('results') or [])[:max_results]]
        except Exception as e:
            logger.warning(f"Brave search failed: {e}")
            return []

    # ── Google Custom Search ──────────────────────────────────────────────────

    def _google_cse(self, query: str, max_results: int = 5) -> List[Dict]:
        """Google web search via Custom Search JSON API (100/day free, $5/1,000 beyond)."""
        api_key = self.config.get('google_cse_key', '')
        cx      = self.config.get('google_cse_cx', '')
        if not api_key or not cx:
            return []
        self._throttle('google_cse')
        try:
            data = _http_get(
                'https://www.googleapis.com/customsearch/v1',
                params={'key': api_key, 'cx': cx, 'q': query[:400],
                        'num': min(max_results, 10)},
                timeout=12,
            )
            if not data:
                return []
            return [{
                'title':   item.get('title', ''),
                'excerpt': item.get('snippet', ''),
                'url':     item.get('link', ''),
                'source':  'google_cse',
            } for item in (data.get('items') or [])[:max_results]]
        except Exception as e:
            logger.warning(f"Google CSE search failed: {e}")
            return []

    # ── Exa AI Neural Search ──────────────────────────────────────────────────

    def _exa(self, query: str, max_results: int = 5) -> List[Dict]:
        """Neural/semantic web search via Exa AI ($7/1,000 queries)."""
        api_key = self.config.get('exa_key', '')
        if not api_key:
            return []
        self._throttle('exa')
        try:
            data = _http_post_json(
                'https://api.exa.ai/search',
                payload={
                    'query':      query[:500],
                    'numResults': min(max_results, 10),
                    'type':       'neural',
                },
                headers={'x-api-key': api_key},
                timeout=15,
            )
            if not data:
                return []
            return [{
                'title':   r.get('title', ''),
                'excerpt': (r.get('highlights') or [''])[0][:400],
                'url':     r.get('url', ''),
                'source':  'exa_ai',
            } for r in (data.get('results') or [])[:max_results]]
        except Exception as e:
            logger.warning(f"Exa AI search failed: {e}")
            return []

    # ── Perplexity Sonar ─────────────────────────────────────────────────────

    def _perplexity(self, query: str) -> List[Dict]:
        """AI-synthesized answer with web citations via Perplexity Sonar API."""
        api_key = self.config.get('perplexity_key', '')
        if not api_key:
            return []
        self._throttle('perplexity')
        try:
            data = _http_post_json(
                'https://api.perplexity.ai/chat/completions',
                payload={
                    'model': 'sonar',
                    'messages': [
                        {'role': 'system', 'content':
                         'You are a legal research assistant. Be concise and cite sources.'},
                        {'role': 'user', 'content': query[:800]},
                    ],
                    'max_tokens': 512,
                },
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=20,
            )
            if not data:
                return []
            content = ((data.get('choices') or [{}])[0]
                       .get('message', {}).get('content', ''))
            citations = data.get('citations') or []
            out = []
            if content:
                out.append({
                    'title':   'Perplexity Sonar: ' + query[:60],
                    'excerpt': content[:600],
                    'url':     citations[0] if citations else '',
                    'source':  'perplexity',
                })
            # Also add citation URLs as individual results
            for cite_url in citations[1:4]:
                out.append({
                    'title':   cite_url,
                    'excerpt': '',
                    'url':     cite_url,
                    'source':  'perplexity_citation',
                })
            return out
        except Exception as e:
            logger.warning(f"Perplexity search failed: {e}")
            return []

    # ── NewsAPI ──────────────────────────────────────────────────────────────

    def _newsapi(self, query: str, max_results: int = 5) -> List[Dict]:
        """News article search via NewsAPI.org ($449/month business tier)."""
        api_key = self.config.get('newsapi_key', '')
        if not api_key:
            return []
        self._throttle('newsapi')
        try:
            data = _http_get(
                'https://newsapi.org/v2/everything',
                params={
                    'q':        query[:500],
                    'sortBy':   'relevancy',
                    'pageSize': min(max_results, 20),
                    'apiKey':   api_key,
                    'language': 'en',
                },
                timeout=12,
            )
            if not data:
                return []
            return [{
                'title':   a.get('title', ''),
                'excerpt': a.get('description', ''),
                'url':     a.get('url', ''),
                'source':  'newsapi',
                'date':    (a.get('publishedAt') or '')[:10],
            } for a in (data.get('articles') or [])[:max_results]]
        except Exception as e:
            logger.warning(f"NewsAPI search failed: {e}")
            return []

    # ── OpenSanctions ─────────────────────────────────────────────────────────

    def _opensanctions(self, name: str) -> List[Dict]:
        """Search OpenSanctions consolidated sanctions/PEP database (~€0.10/call)."""
        api_key = self.config.get('opensanctions_key', '')
        if not api_key:
            return []
        self._throttle('opensanctions')
        try:
            data = _http_get(
                'https://api.opensanctions.org/search/default',
                params={'q': name[:80], 'schema': 'Person', 'limit': 5},
                extra_headers={'Authorization': f'ApiKey {api_key}'},
                timeout=12,
            )
            if not data:
                return []
            out = []
            for r in (data.get('results') or [])[:5]:
                entity_name = r.get('caption', name)
                datasets    = ', '.join(r.get('datasets') or [])
                out.append({
                    'name':    entity_name,
                    'title':   f'OpenSanctions: {entity_name}',
                    'excerpt': f'Found in: {datasets}',
                    'url':     f"https://www.opensanctions.org/entities/{r.get('id','')}",
                    'source':  'opensanctions',
                    'flag':    'sanctions_pep',
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"OpenSanctions search failed: {e}")
            return []

    # ── OpenCorporates ────────────────────────────────────────────────────────

    def _opencorporates(self, name: str) -> List[Dict]:
        """Search OpenCorporates for business entity records (paid key required)."""
        api_key = self.config.get('opencorporates_key', '')
        if not api_key:
            return []
        self._throttle('opencorporates')
        try:
            # Search companies
            data = _http_get(
                'https://api.opencorporates.com/v0.4/companies/search',
                params={'q': name[:80], 'api_token': api_key,
                        'jurisdiction_code': 'us', 'inactive': 'false',
                        'per_page': 5},
                timeout=12,
            )
            if not data:
                return []
            out = []
            companies = (data.get('results') or {}).get('companies') or []
            for item in companies[:5]:
                co   = item.get('company') or item
                corp = co.get('name', '')
                jur  = co.get('jurisdiction_code', '')
                status = co.get('current_status', '')
                url  = co.get('opencorporates_url', '')
                out.append({
                    'title':   f'OpenCorporates: {corp}',
                    'excerpt': f'Jurisdiction: {jur.upper()}, Status: {status}',
                    'url':     url,
                    'source':  'opencorporates',
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"OpenCorporates search failed: {e}")
            return []

    # ── Docket Alarm ─────────────────────────────────────────────────────────

    def _docket_alarm_login(self) -> Optional[str]:
        """Log in to Docket Alarm and return a session token."""
        if self._docket_alarm_token:
            return self._docket_alarm_token
        username = self.config.get('docket_alarm_user', '')
        password = self.config.get('docket_alarm_pass', '')
        if not username or not password:
            return None
        try:
            data = _http_post_json(
                'https://www.docketalarm.com/api/v1/login/',
                payload={'username': username, 'password': password},
                timeout=12,
            )
            if data and data.get('login_token'):
                self._docket_alarm_token = data['login_token']
                return self._docket_alarm_token
        except Exception as e:
            logger.warning(f"Docket Alarm login failed: {e}")
        return None

    def _docket_alarm(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Docket Alarm for federal and state court dockets ($99/month flat)."""
        self._throttle('docket_alarm')
        token = self._docket_alarm_login()
        if not token:
            return []
        try:
            data = _http_get(
                'https://www.docketalarm.com/api/v1/search/',
                params={'login_token': token, 'q': query[:300], 'limit': max_results},
                timeout=15,
            )
            if not data:
                return []
            out = []
            for r in (data.get('search_results') or [])[:max_results]:
                case_name = r.get('case_name', r.get('name', ''))
                court     = r.get('court', '')
                date      = (r.get('date_filed') or '')[:10]
                out.append({
                    'case_name': case_name,
                    'court':     court,
                    'date':      date,
                    'excerpt':   r.get('description', ''),
                    'url':       r.get('link', ''),
                    'source':    'docket_alarm',
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"Docket Alarm search failed: {e}")
            return []

    # ── UniCourt ──────────────────────────────────────────────────────────────

    def _unicourt_login(self) -> Optional[str]:
        """Authenticate to UniCourt via OAuth client credentials."""
        if self._unicourt_token:
            return self._unicourt_token
        client_id     = self.config.get('unicourt_id', '')
        client_secret = self.config.get('unicourt_secret', '')
        if not client_id or not client_secret:
            return None
        try:
            data = _http_post_json(
                'https://auth.unicourt.com/oauth2/token',
                payload={
                    'grant_type':    'client_credentials',
                    'client_id':     client_id,
                    'client_secret': client_secret,
                },
                timeout=12,
            )
            if data and data.get('access_token'):
                self._unicourt_token = data['access_token']
                return self._unicourt_token
        except Exception as e:
            logger.warning(f"UniCourt auth failed: {e}")
        return None

    def _unicourt_party(self, name: str) -> List[Dict]:
        """Search UniCourt for court cases by party name ($49-299/month)."""
        self._throttle('unicourt')
        token = self._unicourt_login()
        if not token:
            return []
        try:
            data = _http_get(
                'https://api.unicourt.com/caseSearch',
                params={'partyName': name[:80], 'pageNumber': 1, 'pageSize': 5},
                extra_headers={'Authorization': f'Bearer {token}'},
                timeout=15,
            )
            if not data:
                return []
            out = []
            for r in (data.get('caseSearchResult') or {}).get('cases', [])[:5]:
                out.append({
                    'case_name': r.get('caseTitle', ''),
                    'court':     (r.get('court') or {}).get('name', ''),
                    'date':      (r.get('caseFiledDate') or '')[:10],
                    'excerpt':   '',
                    'url':       r.get('caseUrl', ''),
                    'source':    'unicourt',
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"UniCourt search failed: {e}")
            return []

    # ── Tavily (paid, AI-optimized) ───────────────────────────────────────────

    def _tavily(self, query: str, max_results: int = 5) -> List[Dict]:
        """AI-optimized search via Tavily API (free tier: 1,000/mo)."""
        api_key = self.config.get('tavily_key', '')
        if not api_key:
            return []
        self._throttle('tavily')
        try:
            data = _http_post_json(
                'https://api.tavily.com/search',
                payload={
                    'api_key':      api_key,
                    'query':        query[:500],
                    'search_depth': 'basic',
                    'max_results':  max_results,
                },
                timeout=15,
            )
            if not data:
                return []
            return [{
                'title':   r.get('title', ''),
                'excerpt': (r.get('content') or '')[:500],
                'url':     r.get('url', ''),
                'source':  'tavily',
            } for r in (data.get('results') or [])[:max_results]]
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return []

    # ── Serper.dev (Google results) ───────────────────────────────────────────

    def _serper(self, query: str, max_results: int = 5) -> List[Dict]:
        """Google search results via Serper.dev API ($50/50k queries)."""
        api_key = self.config.get('serper_key', '')
        if not api_key:
            return []
        self._throttle('serper')
        try:
            data = _http_post_json(
                'https://google.serper.dev/search',
                payload={'q': query[:500], 'num': max_results},
                headers={'X-API-KEY': api_key},
                timeout=12,
            )
            if not data:
                return []
            return [{
                'title':   r.get('title', ''),
                'excerpt': r.get('snippet', ''),
                'url':     r.get('link', ''),
                'source':  'serper_google',
            } for r in (data.get('organic') or [])[:max_results]]
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
            return []

    # ── Lexis-Nexis (enterprise) ──────────────────────────────────────────────

    def _lexisnexis(self, query: str, jurisdiction: str,
                    max_results: int = 5) -> List[Dict]:
        """Lexis-Nexis Research API (requires enterprise subscription)."""
        api_key = self.config.get('lexisnexis_key', '')
        if not api_key:
            return []
        self._throttle('lexisnexis')
        try:
            payload: Dict[str, Any] = {
                'query': query[:500],
                'pageSize': max_results,
                'pageNumber': 1,
            }
            if jurisdiction:
                payload['jurisdiction'] = jurisdiction
            data = _http_post_json(
                'https://api.lexisnexis.com/research/cases/v1/search',
                payload=payload,
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=15,
            )
            if not data:
                return []
            out = []
            for item in (data.get('value') or data.get('cases') or [])[:max_results]:
                out.append({
                    'citation':       item.get('citation', ''),
                    'title':          item.get('caseName', item.get('title', '')),
                    'court':          item.get('court', ''),
                    'date':           (item.get('decisionDate') or '')[:10],
                    'excerpt':        (item.get('headnotes') or item.get('snippet', ''))[:400],
                    'url':            item.get('url', ''),
                    'source':         'lexisnexis',
                    'authority_type': 'binding',
                    'reliability':    'official',
                })
            return out
        except Exception as e:
            logger.warning(f"Lexis-Nexis search failed: {e}")
            return []

    # ── vLex (global case law, subscription) ─────────────────────────────────

    def _vlex(self, query: str, jurisdiction: str,
              max_results: int = 5) -> List[Dict]:
        """Search vLex global case law and statutes (subscription required)."""
        api_key = self.config.get('vlex_key', '')
        if not api_key:
            return []
        self._throttle('vlex')
        try:
            params: Dict[str, Any] = {
                'q':     query[:300],
                'count': max_results,
            }
            if jurisdiction:
                params['jurisdiction'] = jurisdiction
            data = _http_get(
                'https://api.vlex.com/v1/search',
                params=params,
                extra_headers={'Authorization': f'Bearer {api_key}'},
                timeout=15,
            )
            if not data:
                return []
            out = []
            for item in (data.get('results') or data.get('hits') or [])[:max_results]:
                out.append({
                    'citation':       item.get('citation', ''),
                    'title':          item.get('title', item.get('name', '')),
                    'court':          item.get('court', item.get('jurisdiction', '')),
                    'date':           (item.get('decision_date') or item.get('date', ''))[:10],
                    'excerpt':        (item.get('snippet') or '')[:400],
                    'url':            item.get('url', item.get('vlex_url', '')),
                    'source':         'vlex',
                    'authority_type': 'persuasive',
                    'reliability':    'official',
                })
            return out
        except Exception as e:
            logger.warning(f"vLex search failed: {e}")
            return []

    # ── Westlaw Edge (Thomson Reuters, enterprise) ────────────────────────────

    def _westlaw(self, query: str, jurisdiction: str,
                 max_results: int = 5) -> List[Dict]:
        """Search Westlaw Edge case law via Thomson Reuters API (enterprise)."""
        api_key = self.config.get('westlaw_key', '')
        if not api_key:
            return []
        self._throttle('westlaw')
        try:
            data = _http_post_json(
                'https://api.thomsonreuters.com/westlaw/v1/search',
                payload={
                    'query':        query[:500],
                    'jurisdiction': jurisdiction or 'US',
                    'pageSize':     max_results,
                    'contentType':  'cases',
                },
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=15,
            )
            if not data:
                return []
            out = []
            for item in (data.get('results') or [])[:max_results]:
                out.append({
                    'citation':       item.get('citation', ''),
                    'title':          item.get('title', ''),
                    'court':          item.get('court', ''),
                    'date':           (item.get('decidedDate') or '')[:10],
                    'excerpt':        (item.get('synopsis') or '')[:400],
                    'url':            item.get('url', ''),
                    'source':         'westlaw',
                    'authority_type': 'binding',
                    'reliability':    'official',
                })
            return out
        except Exception as e:
            logger.warning(f"Westlaw search failed: {e}")
            return []

    # ── CLEAR (Thomson Reuters, enterprise) ───────────────────────────────────

    def _clear(self, name: str) -> List[Dict]:
        """Comprehensive background check via CLEAR by Thomson Reuters (enterprise)."""
        api_key = self.config.get('clear_key', '')
        if not api_key:
            return []
        self._throttle('clear')
        try:
            data = _http_post_json(
                'https://api.thomsonreuters.com/clear/v1/search',
                payload={'name': name[:80], 'sources': ['criminal', 'civil', 'liens']},
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=20,
            )
            if not data:
                return []
            out = []
            for r in (data.get('results') or [])[:5]:
                out.append({
                    'name':    r.get('name', name),
                    'title':   f"CLEAR: {r.get('record_type', 'Record')} — {r.get('name', name)}",
                    'excerpt': r.get('summary', ''),
                    'url':     r.get('source_url', ''),
                    'source':  'clear_tr',
                    'flag':    r.get('record_type', 'record'),
                    'reliability': 'official',
                })
            return out
        except Exception as e:
            logger.warning(f"CLEAR search failed: {e}")
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _jur_to_cl(self, jurisdiction: str) -> Optional[str]:
        if not jurisdiction:
            return None
        j = jurisdiction.lower()
        if 'supreme court' in j and ('us' in j or 'united states' in j):
            return 'scotus'
        if 'circuit' in j or 'federal' in j:
            return None
        for state, code in _STATE_TO_CL.items():
            if state in j:
                return code
        return None

    def _jur_to_caselaw(self, jurisdiction: str) -> Optional[str]:
        if not jurisdiction:
            return None
        j = jurisdiction.lower()
        if 'federal' in j or 'circuit' in j or 'united states' in j:
            return 'us'
        for state in _STATE_TO_CL:
            if state in j:
                return state.replace(' ', '-')
        return None

    @staticmethod
    def _dedup(results: List[Dict]) -> List[Dict]:
        seen: set = set()
        out: List[Dict] = []
        for r in results:
            key = (r.get('url') or r.get('citation') or r.get('title', ''))[:80].lower()
            if key and key not in seen:
                seen.add(key)
                out.append(r)
        return out


# ── Module-level HTTP helpers ─────────────────────────────────────────────────

def _http_get(url: str, params: dict = None, timeout: int = 10,
              extra_headers: dict = None) -> Optional[dict]:
    """HTTP GET returning parsed JSON, or None on failure."""
    headers = {'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)'}
    if extra_headers:
        headers.update(extra_headers)

    if _HAS_REQUESTS:
        try:
            r = _requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r.json()
            logger.debug(f"GET {url} returned {r.status_code}")
        except Exception as e:
            logger.debug(f"requests GET {url} failed: {e}")

    # urllib fallback
    try:
        full_url = url
        if params:
            full_url = url + '?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(full_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8', errors='replace'))
    except Exception as e:
        logger.debug(f"urllib GET {url} failed: {e}")
    return None


def _http_post_json(url: str, payload: dict,
                    headers: dict = None, timeout: int = 10) -> Optional[dict]:
    """HTTP POST with JSON body, returning parsed JSON or None."""
    if _HAS_REQUESTS:
        try:
            h = {'Content-Type': 'application/json',
                 'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)',
                 **(headers or {})}
            r = _requests.post(url, json=payload, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            logger.debug(f"POST {url} returned {r.status_code}")
        except Exception as e:
            logger.debug(f"requests POST {url} failed: {e}")
    return None
