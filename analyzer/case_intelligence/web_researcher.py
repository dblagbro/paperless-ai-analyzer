"""
Web Research module for Case Intelligence.

Provides web-based legal research to augment the CI pipeline with live sources
beyond what the LLM was trained on:

Free sources (no API key):
  - CourtListener     : Federal case law, opinions, party history
  - Harvard Caselaw   : Comprehensive US case law corpus
  - DuckDuckGo        : General web search for news, background

Optional paid sources (require API key in web_research_config):
  - Tavily            : AI-optimized search (free tier: 1000/mo, then $0.005/req)
  - Serper.dev        : Google search results ($50/50k queries)
  - Lexis-Nexis       : Enterprise legal research API

Role-aware behavior:
  - defense     : queries skew toward case-dismissal, suppression, impeachment material
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

# Rate limits: minimum seconds between calls to each source
_RATE = {
    'courtlistener': 1.2,
    'caselaw':       0.8,
    'ddg':           2.5,
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

    config dict keys:
      courtlistener  bool  CourtListener case law search (free, default True)
      caselaw_api    bool  Harvard Caselaw API (free, default True)
      general_search bool  DuckDuckGo web search (free, default True)
      entity_research bool Research named entities for background (default True)
      tavily_key     str   Tavily API key (optional)
      serper_key     str   Serper.dev API key (optional)
      lexisnexis_key str   Lexis-Nexis API key (optional)
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self._last_call: Dict[str, float] = {}

    # ── Rate limiting ────────────────────────────────────────────────────────

    def _throttle(self, source: str):
        limit = _RATE.get(source, 1.0)
        elapsed = time.monotonic() - self._last_call.get(source, 0.0)
        if elapsed < limit:
            time.sleep(limit - elapsed)
        self._last_call[source] = time.monotonic()

    # ── Public API ───────────────────────────────────────────────────────────

    def search_legal_authorities(
        self,
        query: str,
        jurisdiction: str = None,
        role: str = 'neutral',
        max_results: int = 8,
    ) -> List[Dict]:
        """
        Search for relevant case law and legal authorities.

        Applies a role-aware prefix to bias results toward the user's litigation
        position (e.g. defense sees dismissals/suppression; plaintiff sees judgments).
        Results are de-duplicated across all sources.
        """
        prefix = _ROLE_AUTHORITY_PREFIX.get(role, _ROLE_AUTHORITY_PREFIX['neutral'])
        enriched = f"{prefix} {query[:250]}"
        results: List[Dict] = []

        if self.config.get('courtlistener', True):
            results.extend(self._cl_opinions(enriched, jurisdiction,
                                             max_results=min(5, max_results)))
        if self.config.get('caselaw_api', True):
            results.extend(self._caselaw_search(enriched, jurisdiction,
                                                max_results=min(5, max_results)))
        if self.config.get('tavily_key'):
            results.extend(self._tavily(f'legal case law {enriched}', max_results=3))
        if self.config.get('serper_key'):
            results.extend(self._serper(f'legal precedent {enriched}', max_results=3))
        if self.config.get('lexisnexis_key'):
            results.extend(self._lexisnexis(query, jurisdiction, max_results=5))

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
        and character information relevant to the current litigation role.

        Returns a dict:
          name, entity_type, court_history (list), news_mentions (list), summary (str)
        """
        result: Dict[str, Any] = {
            'name': name,
            'entity_type': entity_type,
            'court_history': [],
            'news_mentions': [],
            'summary': '',
        }

        if not self.config.get('entity_research', True):
            return result

        # Court history via CourtListener party search
        if self.config.get('courtlistener', True):
            result['court_history'] = self._cl_party_search(name)

        # Web news/background with role-aware focus
        if self.config.get('general_search', True):
            if entity_type in ('person', 'individual'):
                kw = _ROLE_ENTITY_PERSON.get(run_role, _ROLE_ENTITY_PERSON['neutral'])
            else:
                kw = _ROLE_ENTITY_ORG.get(run_role, _ROLE_ENTITY_ORG['neutral'])
            web_q = f'"{name}" {kw}'
            result['news_mentions'] = self._ddg(web_q, max_results=4)
            if self.config.get('tavily_key'):
                result['news_mentions'].extend(
                    self._tavily(web_q, max_results=2))
            if self.config.get('serper_key'):
                result['news_mentions'].extend(
                    self._serper(web_q, max_results=2))

        # Plain-text summary for prompt injection
        parts: List[str] = []
        if result['court_history']:
            cases = [
                f"{c.get('case_name', '?')} ({(c.get('date') or '')[:4]})"
                for c in result['court_history'][:3]
            ]
            parts.append(f"Court records: {'; '.join(cases)}")
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
        if self.config.get('tavily_key'):
            results.extend(self._tavily(query, max_results=3))
        if self.config.get('serper_key'):
            results.extend(self._serper(query, max_results=3))
        return self._dedup(results)[:max_results]

    # ── CourtListener ────────────────────────────────────────────────────────

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

    # ── Harvard Caselaw Access Project ───────────────────────────────────────

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

    # ── DuckDuckGo Lite ──────────────────────────────────────────────────────

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

            # Parse DuckDuckGo HTML (class="result__a" links + class="result__snippet")
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
                # DDG wraps real URLs: //duckduckgo.com/l/?uddg=<url-encoded>
                real_url = raw_url
                uddg_m = re.search(r'uddg=([^&]+)', raw_url)
                if uddg_m:
                    real_url = urllib.parse.unquote(uddg_m.group(1))
                # Skip ads (decoded URL goes to duckduckgo.com/y.js)
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

    # ── Tavily (paid, AI-optimized) ──────────────────────────────────────────

    def _tavily(self, query: str, max_results: int = 5) -> List[Dict]:
        """AI-optimized search via Tavily API."""
        api_key = self.config.get('tavily_key', '')
        if not api_key:
            return []
        self._throttle('tavily')
        try:
            data = _http_post_json(
                'https://api.tavily.com/search',
                payload={
                    'api_key': api_key,
                    'query': query[:500],
                    'search_depth': 'basic',
                    'max_results': max_results,
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

    # ── Serper.dev (Google results) ──────────────────────────────────────────

    def _serper(self, query: str, max_results: int = 5) -> List[Dict]:
        """Google search results via Serper.dev API."""
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

    # ── Lexis-Nexis (enterprise) ─────────────────────────────────────────────

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

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _jur_to_cl(self, jurisdiction: str) -> Optional[str]:
        """Map a jurisdiction display name to a CourtListener court filter."""
        if not jurisdiction:
            return None
        j = jurisdiction.lower()
        if 'supreme court' in j and ('us' in j or 'united states' in j):
            return 'scotus'
        if 'circuit' in j or 'federal' in j:
            return None  # search all federal courts
        for state, code in _STATE_TO_CL.items():
            if state in j:
                return code
        return None

    def _jur_to_caselaw(self, jurisdiction: str) -> Optional[str]:
        """Map a jurisdiction display name to a Harvard Caselaw jurisdiction slug."""
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
        """De-duplicate by URL or citation."""
        seen: set = set()
        out: List[Dict] = []
        for r in results:
            key = (r.get('url') or r.get('citation') or r.get('title', ''))[:80].lower()
            if key and key not in seen:
                seen.add(key)
                out.append(r)
        return out


# ── Module-level HTTP helpers ─────────────────────────────────────────────────

def _http_get(url: str, params: dict = None, timeout: int = 10) -> Optional[dict]:
    """HTTP GET returning parsed JSON, or None on failure."""
    if _HAS_REQUESTS:
        try:
            r = _requests.get(
                url, params=params, timeout=timeout,
                headers={'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)'},
            )
            if r.status_code == 200:
                return r.json()
            logger.debug(f"GET {url} returned {r.status_code}")
        except Exception as e:
            logger.debug(f"requests GET {url} failed: {e}")

    # urllib fallback
    try:
        if params:
            url = url + '?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Paperless-AI-Analyzer/3.6 (legal-research)'},
        )
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
            h = {'Content-Type': 'application/json', **(headers or {})}
            r = _requests.post(url, json=payload, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            logger.debug(f"POST {url} returned {r.status_code}")
        except Exception as e:
            logger.debug(f"requests POST {url} failed: {e}")
    return None
