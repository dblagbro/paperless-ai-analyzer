"""web_researchers package — WebResearcher class composed of provider mixins.

Package layout (see refactor-log Entry 011):
    constants.py         # _RATE, _STATE_TO_CL, _ROLE_* mappings
    http_utils.py        # _http_get, _http_post_json
    base.py              # WebResearcherBaseMixin (__init__, throttle, helpers)
    providers_legal.py   # LegalProvidersMixin (7 sources)
    providers_general.py # GeneralSearchProvidersMixin (9 sources)
    providers_entities.py# EntityResearchProvidersMixin (7 sources)

External import surface unchanged:
    from analyzer.case_intelligence.web_researcher import WebResearcher
still works — `web_researcher.py` is now a re-export pointing here.
"""
import logging
from typing import Any, Dict, List

from .base import WebResearcherBaseMixin
from .constants import _ROLE_AUTHORITY_PREFIX, _ROLE_ENTITY_ORG, _ROLE_ENTITY_PERSON
from .providers_entities import EntityResearchProvidersMixin
from .providers_general import GeneralSearchProvidersMixin
from .providers_legal import LegalProvidersMixin

logger = logging.getLogger(__name__)


class WebResearcher(
    LegalProvidersMixin,
    GeneralSearchProvidersMixin,
    EntityResearchProvidersMixin,
    WebResearcherBaseMixin,
):
    """Orchestrates web-based legal research across every configured source.

    See constants.py docstrings and config-key tables in the README for the
    full list of supported providers and their API-key configuration.
    """

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
