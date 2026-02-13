"""
Profile Loader and Matcher

Loads document type profiles and matches documents to profiles.
"""

import re
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    """Document type profile."""
    profile_id: str
    display_name: str
    version: int
    description: str
    match: Dict[str, Any]
    extraction: Dict[str, Any]
    validation: Dict[str, Any]
    checks_enabled: List[str]
    forensics: Dict[str, Any]
    tagging: Dict[str, Any]
    raw_config: Dict[str, Any]  # Store full config for reference


class ProfileLoader:
    """Loads and manages document type profiles."""

    def __init__(self, profiles_dir: str = '/app/profiles'):
        """
        Initialize profile loader.

        Args:
            profiles_dir: Path to profiles directory
        """
        self.profiles_dir = Path(profiles_dir)
        self.active_dir = self.profiles_dir / 'active'
        self.staging_dir = self.profiles_dir / 'staging'
        self.examples_dir = self.profiles_dir / 'examples'

        # Create directories if they don't exist
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)

        self.profiles: List[Profile] = []
        self.load_profiles()

    def load_profiles(self) -> None:
        """Load all active profiles from disk."""
        self.profiles = []

        if not list(self.active_dir.glob('*.yaml')):
            logger.warning(f"No active profiles found in {self.active_dir}")
            logger.info("Copy example profiles to active/ directory to enable profile matching")
            return

        for profile_file in self.active_dir.glob('*.yaml'):
            try:
                with open(profile_file, 'r') as f:
                    config = yaml.safe_load(f)

                profile = Profile(
                    profile_id=config['profile_id'],
                    display_name=config.get('display_name', config['profile_id']),
                    version=config.get('version', 1),
                    description=config.get('description', ''),
                    match=config.get('match', {}),
                    extraction=config.get('extraction', {}),
                    validation=config.get('validation', {}),
                    checks_enabled=config.get('checks_enabled', []),
                    forensics=config.get('forensics', {}),
                    tagging=config.get('tagging', {}),
                    raw_config=config
                )

                self.profiles.append(profile)
                logger.info(f"Loaded profile: {profile.profile_id} v{profile.version}")

            except Exception as e:
                logger.error(f"Failed to load profile {profile_file}: {e}")

        logger.info(f"Loaded {len(self.profiles)} active profiles")

    def match_profile(self, document: Dict[str, Any]) -> Tuple[Optional[Profile], float]:
        """
        Find the best matching profile for a document.

        Args:
            document: Document data from Paperless API

        Returns:
            Tuple of (matching profile, score) or (None, 0.0) if no match
        """
        if not self.profiles:
            logger.debug("No profiles loaded, cannot match")
            return None, 0.0

        content = document.get('content', '').lower()
        mime_type = document.get('mime_type', '')

        best_profile = None
        best_score = 0.0

        for profile in self.profiles:
            score = self._calculate_match_score(profile, content, mime_type)
            min_score = profile.match.get('min_score', 0.6)

            logger.debug(f"Profile {profile.profile_id}: score={score:.2f} (min={min_score})")

            if score >= min_score and score > best_score:
                best_score = score
                best_profile = profile

        if best_profile:
            logger.info(f"Matched document to profile: {best_profile.profile_id} "
                       f"(score={best_score:.2f})")
        else:
            logger.info(f"No profile matched (best score: {best_score:.2f})")

        return best_profile, best_score

    def _calculate_match_score(self, profile: Profile, content: str, mime_type: str) -> float:
        """
        Calculate match score for a profile against document content.

        Args:
            profile: Profile to test
            content: Document content (lowercased)
            mime_type: Document MIME type

        Returns:
            Match score (0.0 to 1.0)
        """
        match_config = profile.match
        score_points = []

        # Keyword matching
        keywords_config = match_config.get('keywords', {})

        # ANY keywords
        any_keywords = keywords_config.get('any', [])
        if any_keywords:
            any_matches = sum(1 for kw in any_keywords if kw.lower() in content)
            if any_matches > 0:
                score_points.append(min(any_matches * 0.1, 0.5))  # Cap at 0.5

        # ALL keywords
        all_keywords = keywords_config.get('all', [])
        if all_keywords:
            all_match = all(kw.lower() in content for kw in all_keywords)
            if all_match:
                score_points.append(0.3)

        # Regex matching
        regex_patterns = match_config.get('regex', {}).get('any', [])
        for pattern in regex_patterns:
            try:
                if re.search(pattern, content, re.IGNORECASE):
                    score_points.append(0.15)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        # MIME type matching
        mime_types = match_config.get('mime_types', [])
        if mime_type in mime_types:
            score_points.append(0.2)

        # Calculate final score (normalized to 0-1)
        if not score_points:
            return 0.0

        total_score = sum(score_points)
        # Normalize: cap at 1.0
        return min(total_score, 1.0)

    def generate_staging_profile(self,
                                 document: Dict[str, Any],
                                 extracted_data: Optional[Dict] = None) -> str:
        """
        Generate a suggested profile based on document analysis.

        Args:
            document: Document data from Paperless API
            extracted_data: Optional extracted data to inform profile

        Returns:
            Path to generated profile file
        """
        from datetime import datetime

        doc_id = document['id']
        content = document.get('content', '')
        mime_type = document.get('mime_type', '')

        # Extract potential keywords (top 10 most common multi-word phrases)
        keywords = self._extract_keywords(content)

        # Generate profile ID
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        profile_id = f"suggested_{timestamp}_doc_{doc_id}"

        # Build profile config
        profile_config = {
            'profile_id': profile_id,
            'display_name': f"Suggested Profile for Document {doc_id}",
            'version': 1,
            'description': f"Auto-generated profile suggestion based on document {doc_id}",
            'match': {
                'keywords': {
                    'any': keywords[:10],  # Top 10 keywords
                    'all': []
                },
                'regex': {
                    'any': []
                },
                'mime_types': [mime_type],
                'min_score': 0.5
            },
            'extraction': {
                'engine': 'unstructured',
                'mode': 'elements',
                'table_hints': {
                    'column_keywords': ['date', 'description', 'amount', 'balance'],
                    'date_regex': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                    'currency_symbols': ['$', 'USD']
                }
            },
            'validation': {
                'currency': 'USD',
                'running_balance_tolerance': 0.01,
                'allow_future_dates': False
            },
            'checks_enabled': [
                'running_balance',
                'page_totals',
                'continuity',
                'duplicates',
                'date_order',
                'image_forensics'
            ],
            'forensics': {
                'enabled': True,
                'dpi': 300
            },
            'tagging': {
                'deterministic_prefix': 'anomaly:',
                'ai_prefix': 'aianomaly:',
                'auto_tags': ['analyzed:deterministic:v1']
            }
        }

        # Write to staging directory
        staging_file = self.staging_dir / f"{profile_id}.yaml"
        with open(staging_file, 'w') as f:
            yaml.dump(profile_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated staging profile: {staging_file}")
        return str(staging_file)

    def _extract_keywords(self, content: str, max_keywords: int = 20) -> List[str]:
        """
        Extract potential keywords from document content.

        Args:
            content: Document text content
            max_keywords: Maximum number of keywords to return

        Returns:
            List of keyword phrases
        """
        # Simple keyword extraction: find common phrases
        words = re.findall(r'\b[a-z]{3,}\b', content.lower())

        # Common financial/document terms
        common_terms = [
            'statement period', 'account number', 'beginning balance', 'ending balance',
            'total amount', 'transaction', 'merchant', 'invoice', 'due date',
            'payment', 'balance', 'deposit', 'withdrawal', 'credit', 'debit'
        ]

        # Find which common terms exist in content
        found_keywords = [term for term in common_terms if term in content.lower()]

        # Add most frequent words (simple frequency count)
        from collections import Counter
        word_freq = Counter(words)
        frequent_words = [word for word, count in word_freq.most_common(30)
                         if len(word) > 4 and count > 2]

        # Combine and deduplicate
        all_keywords = found_keywords + frequent_words
        return list(dict.fromkeys(all_keywords))[:max_keywords]  # Deduplicate and limit
