"""Public-records / entity-background providers: BOP inmate locator,
OFAC sanctions, SEC EDGAR, FEC OpenData, OpenSanctions, OpenCorporates, CLEAR.

Extracted from web_researcher.py during the v3.9.8 split. Assumes the host
class mixes in WebResearcherBaseMixin."""
import logging
import re
import urllib.parse
from typing import Dict, List

from .http_utils import _http_get

logger = logging.getLogger(__name__)


class EntityResearchProvidersMixin:
    """BOP, OFAC, SEC EDGAR, FEC, OpenSanctions, OpenCorporates, CLEAR."""

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

