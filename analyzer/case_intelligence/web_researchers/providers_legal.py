"""Case law + docket providers: CourtListener, Harvard Caselaw,
Lexis-Nexis, vLex, Westlaw, Docket Alarm, UniCourt.

Extracted from web_researcher.py during the v3.9.8 split. Assumes the host
class mixes in WebResearcherBaseMixin (for `_throttle`, `_jur_to_cl`, etc.)."""
import json
import logging
import re
import time
import urllib.parse
from typing import Dict, List, Optional

from .http_utils import _http_get, _http_post_json

logger = logging.getLogger(__name__)


class LegalProvidersMixin:
    """CourtListener, Caselaw, Lexis-Nexis, vLex, Westlaw, Docket Alarm, UniCourt."""

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

