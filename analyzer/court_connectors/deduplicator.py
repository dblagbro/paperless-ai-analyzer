"""
3-tier deduplication for court document imports.

Tier 1 — URL match:   check source_url in court_imported_docs
Tier 2 — Hash match:  SHA-256 of downloaded bytes vs court_imported_docs
Tier 3 — Title match: Paperless title search via paperless_client
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CourtDeduplicator:
    """
    Checks a candidate document against all three deduplication tiers.

    Usage:
        dedup = CourtDeduplicator(project_slug, paperless_client)
        skip, reason = dedup.check_url(source_url)
        if skip:
            ...continue...
        tmp_path = connector.download_document(entry)
        skip, reason = dedup.check_hash(tmp_path)
        if skip:
            tmp_path.unlink(missing_ok=True)
            ...continue...
        skip, reason = dedup.check_title(title)
        if skip:
            tmp_path.unlink(missing_ok=True)
            ...continue...
        # upload tmp_path
    """

    def __init__(self, project_slug: str, paperless_client=None):
        self.project_slug = project_slug
        self.paperless_client = paperless_client

    def check_url(self, source_url: str) -> Tuple[bool, str]:
        """
        Tier 1: URL-based dedup.

        Returns:
            (should_skip: bool, reason: str)
        """
        if not source_url:
            return False, ''
        try:
            from analyzer.court_db import url_already_imported
            if url_already_imported(self.project_slug, source_url):
                return True, 'url_match'
        except Exception as e:
            logger.warning(f"URL dedup check failed: {e}")
        return False, ''

    def hash_file(self, file_path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()

    def check_hash(self, file_path: Path) -> Tuple[bool, str, str]:
        """
        Tier 2: Hash-based dedup.

        Returns:
            (should_skip: bool, reason: str, sha256_hex: str)
        """
        try:
            digest = self.hash_file(file_path)
            from analyzer.court_db import hash_already_imported
            if hash_already_imported(self.project_slug, digest):
                return True, 'hash_match', digest
            return False, '', digest
        except Exception as e:
            logger.warning(f"Hash dedup check failed: {e}")
            return False, '', ''

    def check_title(self, title: str) -> Tuple[bool, str]:
        """
        Tier 3: Paperless title search dedup.

        Returns:
            (should_skip: bool, reason: str)
        """
        if not self.paperless_client or not title:
            return False, ''
        try:
            # Use a prefix of the title to avoid false positives from truncation
            search_fragment = title[:80]
            result = self.paperless_client.get_documents(
                title__icontains=search_fragment, page_size=5
            )
            if result and result.get('count', 0) > 0:
                return True, 'title_match'
        except Exception as e:
            logger.debug(f"Title dedup check failed (non-fatal): {e}")
        return False, ''
