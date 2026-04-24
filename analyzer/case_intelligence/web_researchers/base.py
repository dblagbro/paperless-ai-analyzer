"""Base WebResearcher mixin: __init__, rate limiting, jurisdiction helpers,
dedup. All provider mixins assume `self.config`, `self._last_call`, and the
throttle machinery defined here.

Extracted from web_researcher.py during the v3.9.8 split."""
import logging
import time
from typing import Dict, List, Optional

from .constants import _RATE, _STATE_TO_CL

logger = logging.getLogger(__name__)


class WebResearcherBaseMixin:
    """State + utility methods shared by every provider mixin."""

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
