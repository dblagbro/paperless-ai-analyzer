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
ECFR_SEARCH_BASE = 'https://www.ecfr.gov/api/search/v1'
FEDERAL_REGISTER_BASE = 'https://www.federalregister.gov/api/v1'
COURTLISTENER_BASE = 'https://www.courtlistener.com/api/rest/v4'

# NYS statute laws to ingest (CPLR, DRL, EPTL, SCPA, UCC, RPL, FCA)
NYS_STATUTE_IDS = ['CVP', 'DOM', 'EPT', 'SCP', 'UCC', 'RPP', 'FCT']

# eCFR search queries: legally relevant topics across all titles
ECFR_SEARCH_QUERIES = [
    'fraud criminal evidence procedure',
    'money laundering financial crimes',
    'securities fraud insider trading',
    'bank secrecy anti-money laundering',
    'criminal forfeiture asset seizure',
    'judicial administration court rules',
    'tax evasion IRS civil penalties',
    'whistleblower retaliation discrimination',
]


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
        Requires a free API token from https://legislation.nysenate.gov/register
        Returns count of authorities ingested.
        """
        if not self.nysenate_token:
            logger.warning(
                "NYS Senate ingestion skipped — API token required. "
                "Register for a free key at https://legislation.nysenate.gov/register "
                "and set NYSENATE_API_TOKEN in the container environment."
            )
            return 0

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
        Ingest federal regulations using the eCFR Search API (free, no key).
        Runs targeted queries across legally relevant topics and titles.
        Returns count of regulations ingested.
        """
        count = 0
        seen_citations: set = set()

        for query in ECFR_SEARCH_QUERIES:
            try:
                params = {
                    'query':    query,
                    'per_page': 20,
                    'page':     1,
                }
                resp = self.session.get(
                    f'{ECFR_SEARCH_BASE}/results',
                    params=params,
                    timeout=self.timeout,
                )
                if not resp.ok:
                    logger.warning(f"eCFR search '{query}' failed: {resp.status_code}")
                    continue

                results = resp.json().get('results') or []
                for sec in results:
                    try:
                        h   = sec.get('hierarchy') or {}
                        hh  = sec.get('hierarchy_headings') or {}
                        title_num = h.get('title', '')
                        section   = h.get('section', '') or h.get('part', '')
                        citation  = f"{title_num} C.F.R. § {section}" if section else f"CFR Title {title_num}"
                        if citation in seen_citations:
                            continue
                        seen_citations.add(citation)

                        sec_label = (sec.get('headings') or {}).get('section', '') or citation
                        text      = sec.get('full_text_excerpt') or ''
                        source_url = (
                            f"https://www.ecfr.gov/current/title-{title_num}/section-{section}"
                            if section else f"https://www.ecfr.gov/current/title-{title_num}"
                        )

                        upsert_authority_corpus_entry(
                            citation=citation,
                            source='ecfr',
                            jurisdiction=jurisdiction,
                            authority_type='regulation',
                            title=sec_label,
                            content_text=text,
                            source_url=source_url,
                            reliability='official',
                        )
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to ingest eCFR section: {e}")

                time.sleep(0.3)

            except Exception as e:
                logger.error(f"eCFR ingestion query '{query}' failed: {e}")

        logger.info(f"eCFR ingestion complete: {count} regulations ingested")
        return count

    def ingest_federal_register(self, jurisdiction: str = 'US',
                                 max_results: int = 120) -> int:
        """
        Ingest regulatory documents from the Federal Register API (free, no key).
        Searches legally-relevant topics: fraud, enforcement, AML, securities, tax.
        """
        count = 0
        topics = [
            'fraud criminal enforcement penalty',
            'anti-money laundering bank secrecy',
            'securities fraud insider trading',
            'tax evasion civil penalty',
            'consumer protection unfair deceptive',
            'civil rights discrimination employment',
        ]
        per_topic = max(10, max_results // len(topics))

        for topic in topics:
            try:
                # Build URL manually to properly encode array params
                base = f'{FEDERAL_REGISTER_BASE}/documents.json'
                url  = (f'{base}?conditions[term]={requests.utils.quote(topic)}'
                        f'&per_page={per_topic}'
                        f'&fields[]=title&fields[]=citation&fields[]=html_url&fields[]=abstract'
                        f'&order=relevance')
                resp = self.session.get(url, timeout=self.timeout)
                if not resp.ok:
                    logger.warning(f"Federal Register topic '{topic}' failed: {resp.status_code}")
                    continue

                for doc in (resp.json().get('results') or []):
                    try:
                        citation = doc.get('citation') or doc.get('document_number', '')
                        title    = doc.get('title', '')
                        text     = doc.get('abstract') or ''
                        url_doc  = doc.get('html_url', '')
                        upsert_authority_corpus_entry(
                            citation=citation,
                            source='federal_register',
                            jurisdiction=jurisdiction,
                            authority_type='regulation',
                            title=title,
                            content_text=text,
                            source_url=url_doc,
                            reliability='official',
                        )
                        count += 1
                    except Exception as e:
                        logger.warning(f"Federal Register doc ingest failed: {e}")

                time.sleep(0.3)

            except Exception as e:
                logger.error(f"Federal Register topic '{topic}' failed: {e}")

        logger.info(f"Federal Register ingestion complete: {count} documents ingested")
        return count

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
                params={**params, 'format': 'json'},
                timeout=self.timeout,
            )
            if not resp.ok:
                logger.warning(f"CourtListener search failed: {resp.status_code}")
                return 0

            results = resp.json().get('results', [])
            for opinion in results[:max_results]:
                try:
                    # v4 API returns citation as a list; join or take first
                    raw_cite = opinion.get('citation') or opinion.get('caseName', 'Unknown')
                    if isinstance(raw_cite, list):
                        citation = raw_cite[0] if raw_cite else opinion.get('caseName', 'Unknown')
                    else:
                        citation = raw_cite or opinion.get('caseName', 'Unknown')
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
                     Valid values: nysenate, ecfr, federal_register, courtlistener

        Returns:
            Dict with counts per source.
        """
        if sources is None:
            sources = ['nysenate', 'ecfr', 'federal_register', 'courtlistener']

        results = {}

        if 'nysenate' in sources:
            results['nysenate'] = self.ingest_nys_statutes()

        if 'ecfr' in sources:
            results['ecfr'] = self.ingest_ecfr()

        if 'federal_register' in sources:
            results['federal_register'] = self.ingest_federal_register()

        if 'courtlistener' in sources:
            jurisdictions = ['NYS', 'SDNY', 'US']
            if jurisdiction_profile:
                jurisdictions = jurisdiction_profile.authority_jurisdictions or jurisdictions
            results['courtlistener'] = self.ingest_courtlistener_opinions(jurisdictions)

        total = sum(results.values())
        logger.info(f"Authority ingestion complete: {total} total entries — {results}")
        return results
