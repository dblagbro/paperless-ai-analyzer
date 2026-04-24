"""General web search + news providers: DuckDuckGo, GDELT, Brave, Google CSE,
Exa, Perplexity, Tavily, Serper, NewsAPI.

Extracted from web_researcher.py during the v3.9.8 split. Assumes the host
class mixes in WebResearcherBaseMixin."""
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List

from .http_utils import _http_get, _http_post_json

logger = logging.getLogger(__name__)


class GeneralSearchProvidersMixin:
    """DDG, GDELT, Brave, Google CSE, Exa, Perplexity, Tavily, Serper, NewsAPI."""

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

