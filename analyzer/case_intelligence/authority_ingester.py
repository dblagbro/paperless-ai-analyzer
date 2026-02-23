"""
Authority Ingester — fetches legal authorities from external APIs.

Sources (in priority order):
  1. NY Senate Open Legislation API (NYS statutes)
  2. eCFR API v1 (federal regulations)
  3. CourtListener REST API (court opinions)
  4. Federal Register API
  5. GovInfo API
"""

import logging
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import requests

from analyzer.case_intelligence.db import upsert_authority_corpus_entry

logger = logging.getLogger(__name__)

# API base URLs
NYSENATE_BASE = 'https://legislation.nysenate.gov/api/3'
ECFR_BASE = 'https://www.ecfr.gov/api/versioner/v1'
COURTLISTENER_BASE = 'https://www.courtlistener.com/api/rest/v3'

# NYS statute laws to ingest (CPLR, DRL, EPTL, SCPA, UCC, RPL, FCA)
NYS_STATUTE_IDS = ['CVP', 'DOM', 'EPT', 'SCP', 'UCC', 'RPP', 'FCT']

# Federal regulation titles relevant to legal proceedings
FEDERAL_REGULATION_TITLES = {
    '28': 'Judicial Administration',
    '11': 'Federal Elections (Bankruptcy Code in Title 11 USC)',
}


class AuthorityIngester:
    """
    Fetches and stores legal authorities in the ci_authority_corpus table.
    Does NOT embed — embedding is handled separately by AuthorityRetriever.
    """

    def __init__(self, courtlistener_token: Optional[str] = None,
                 nysenate_token: Optional[str] = None,
                 request_timeout: int = 30):
        self.courtlistener_token = courtlistener_token
        self.nysenate_token = nysenate_token
        self.timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'PaperlessAIAnalyzer/3.0 (legal research)'})
        if courtlistener_token:
            self.session.headers['Authorization'] = f'Token {courtlistener_token}'

    def ingest_nys_statutes(self, law_ids: List[str] = None,
                             jurisdiction: str = 'NYS') -> int:
        """
        Ingest NYS statutes from the NY Senate Open Legislation API.
        Returns count of authorities ingested.
        """
        if law_ids is None:
            law_ids = NYS_STATUTE_IDS

        count = 0
        for law_id in law_ids:
            try:
                count += self._ingest_nys_law(law_id, jurisdiction)
                time.sleep(0.5)  # Be polite to the API
            except Exception as e:
                logger.error(f"NYS statute ingestion failed for {law_id}: {e}")

        logger.info(f"NYS statute ingestion complete: {count} articles ingested")
        return count

    def _ingest_nys_law(self, law_id: str, jurisdiction: str) -> int:
        """Ingest a single NYS law (all articles/sections)."""
        count = 0
        try:
            # Fetch law tree (list of articles)
            params = {'limit': 50}
            if self.nysenate_token:
                params['key'] = self.nysenate_token

            url = f'{NYSENATE_BASE}/laws/{law_id}/tree'
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if not resp.ok:
                logger.warning(f"NYS law {law_id} tree fetch failed: {resp.status_code}")
                return 0

            data = resp.json()
            law_name = data.get('result', {}).get('info', {}).get('name', law_id)
            documents = data.get('result', {}).get('documents', {})
            sections = self._flatten_law_tree(documents)

            for section in sections[:200]:  # Limit per law
                try:
                    section_id = section.get('locationId', '')
                    section_title = section.get('title', '')
                    citation = f"{law_id} § {section_id}"

                    corpus_id = upsert_authority_corpus_entry(
                        citation=citation,
                        source='nysenate',
                        jurisdiction=jurisdiction,
                        authority_type='statute',
                        title=f"{law_name} {section_title}",
                        content_text=section.get('text', ''),
                        source_url=f'https://www.nysenate.gov/legislation/laws/{law_id}/{section_id}',
                        reliability='official',
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to ingest {law_id} section: {e}")

        except Exception as e:
            logger.error(f"_ingest_nys_law failed for {law_id}: {e}")

        return count

    def _flatten_law_tree(self, documents: dict, result: list = None) -> list:
        """Recursively flatten the NY Senate law tree into a list of sections."""
        if result is None:
            result = []
        if not documents:
            return result
        if documents.get('docType') == 'SECTION':
            result.append(documents)
        for child in documents.get('documents', []):
            self._flatten_law_tree(child, result)
        return result

    def ingest_ecfr(self, title_numbers: List[str] = None,
                    jurisdiction: str = 'US') -> int:
        """
        Ingest federal regulations from the eCFR API.
        Returns count of regulations ingested.
        """
        if title_numbers is None:
            title_numbers = ['28']  # Judicial Administration by default

        count = 0
        for title_num in title_numbers:
            try:
                count += self._ingest_ecfr_title(title_num, jurisdiction)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"eCFR ingestion failed for title {title_num}: {e}")

        logger.info(f"eCFR ingestion complete: {count} regulations ingested")
        return count

    def _ingest_ecfr_title(self, title_num: str, jurisdiction: str) -> int:
        """Ingest a single eCFR title."""
        count = 0
        try:
            url = f'{ECFR_BASE}/full/{datetime.now().strftime("%Y-%m-%d")}/title-{title_num}.json'
            resp = self.session.get(url, timeout=self.timeout)
            if not resp.ok:
                logger.warning(f"eCFR title {title_num} fetch failed: {resp.status_code}")
                return 0

            data = resp.json()
            sections = self._extract_ecfr_sections(data, title_num)

            for sec in sections[:100]:
                try:
                    citation = f"{title_num} C.F.R. § {sec.get('identifier', '')}"
                    corpus_id = upsert_authority_corpus_entry(
                        citation=citation,
                        source='ecfr',
                        jurisdiction=jurisdiction,
                        authority_type='regulation',
                        title=sec.get('label', citation),
                        content_text=sec.get('text', ''),
                        source_url=f'https://www.ecfr.gov/current/title-{title_num}/section-{sec.get("identifier","")}',
                        reliability='official',
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to ingest eCFR section: {e}")

        except Exception as e:
            logger.error(f"_ingest_ecfr_title failed for title {title_num}: {e}")

        return count

    def _extract_ecfr_sections(self, data: dict, title_num: str,
                                result: list = None) -> list:
        """Recursively extract sections from eCFR JSON tree."""
        if result is None:
            result = []
        if data.get('type') == 'SECTION':
            text = ' '.join(
                p.get('text', '') for p in data.get('children', [])
                if isinstance(p, dict) and p.get('type') == 'P'
            )
            result.append({
                'identifier': data.get('identifier', ''),
                'label': data.get('label', ''),
                'text': text,
            })
        for child in data.get('children', []):
            if isinstance(child, dict):
                self._extract_ecfr_sections(child, title_num, result)
        return result

    def ingest_courtlistener_opinions(self, jurisdictions: List[str],
                                       query: str = 'CPLR OR bankruptcy OR fraud',
                                       max_results: int = 50) -> int:
        """
        Ingest court opinions from CourtListener.
        Returns count of opinions ingested.
        """
        count = 0
        try:
            params = {
                'q': query,
                'type': 'o',  # opinions
                'order_by': 'score desc',
                'format': 'json',
                'page_size': min(max_results, 20),
            }

            # Filter by jurisdiction if specific courts specified
            if jurisdictions and 'NYS' in jurisdictions:
                params['q'] = f"({query}) AND court:ny*"

            resp = self.session.get(
                f'{COURTLISTENER_BASE}/search/',
                params=params,
                timeout=self.timeout,
            )
            if not resp.ok:
                logger.warning(f"CourtListener search failed: {resp.status_code}")
                return 0

            results = resp.json().get('results', [])
            for opinion in results[:max_results]:
                try:
                    citation = opinion.get('citation', '') or opinion.get('caseName', 'Unknown')
                    corpus_id = upsert_authority_corpus_entry(
                        citation=citation,
                        source='courtlistener',
                        jurisdiction=self._map_court_to_jurisdiction(
                            opinion.get('court_id', '')
                        ),
                        authority_type='case_law',
                        title=opinion.get('caseName', citation),
                        content_text=(opinion.get('snippet', '') or ''),
                        source_url=f"https://www.courtlistener.com{opinion.get('absolute_url', '')}",
                        reliability='official',
                    )
                    count += 1
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Failed to ingest CourtListener opinion: {e}")

        except Exception as e:
            logger.error(f"CourtListener ingestion failed: {e}")

        logger.info(f"CourtListener ingestion complete: {count} opinions ingested")
        return count

    def _map_court_to_jurisdiction(self, court_id: str) -> str:
        """Map CourtListener court_id to our jurisdiction string."""
        if 'ny' in court_id.lower():
            return 'NYS'
        if 'sdny' in court_id.lower():
            return 'SDNY'
        if 'edny' in court_id.lower():
            return 'EDNY'
        if '2d' in court_id.lower() or 'ca2' in court_id.lower():
            return '2nd Circuit'
        return 'US'

    def ingest_all(self, jurisdiction_profile=None, sources: List[str] = None) -> dict:
        """
        Run full ingestion across all configured sources.

        Args:
            jurisdiction_profile: JurisdictionProfile to filter relevance
            sources: List of source names to use (default: all)

        Returns:
            Dict with counts per source.
        """
        if sources is None:
            sources = ['nysenate', 'ecfr', 'courtlistener']

        results = {}

        if 'nysenate' in sources:
            results['nysenate'] = self.ingest_nys_statutes()

        if 'ecfr' in sources:
            results['ecfr'] = self.ingest_ecfr()

        if 'courtlistener' in sources:
            jurisdictions = ['NYS', 'SDNY', 'US']
            if jurisdiction_profile:
                jurisdictions = jurisdiction_profile.authority_jurisdictions or jurisdictions
            results['courtlistener'] = self.ingest_courtlistener_opinions(jurisdictions)

        total = sum(results.values())
        logger.info(f"Authority ingestion complete: {total} total entries — {results}")
        return results
