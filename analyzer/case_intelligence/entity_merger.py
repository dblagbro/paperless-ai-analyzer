"""
Entity Merger — Phase 1M CI Pipeline Stage.

Runs after Phase 1 extraction to deduplicate and merge entities that refer
to the same person or organization. Uses a two-pass approach:

  Pass 1 — Deterministic: case-insensitive exact match, punctuation
           normalization, middle-initial collapsing.
  Pass 2 — AI-assisted: fuzzy merge using LLM for abbreviations, aliases,
           maiden names, entity shorthand.

Merged duplicates are flagged (merged_into FK) but NOT deleted.
"""

import json
import logging
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Any

from analyzer.case_intelligence.db import (
    get_ci_entities, upsert_ci_entity,
    mark_entity_merged, update_entity_aliases, get_ci_entities_active,
)

logger = logging.getLogger(__name__)

# Max entities to send to LLM in one merge call (to stay within token limits)
_AI_MERGE_BATCH = 50


def _normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not name:
        return ''
    name = unicodedata.normalize('NFKD', name)
    name = name.lower()
    # Remove common legal suffixes for comparison only
    name = re.sub(r'\b(llc|llp|lp|inc|corp|co|ltd|plc|pllc|pc|pa|esq|jr|sr|ii|iii|iv)\b\.?', '', name)
    # Normalize & → and
    name = name.replace('&', 'and')
    # Strip remaining punctuation except spaces
    name = re.sub(r"[^\w\s]", '', name)
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _middle_initial_key(name: str) -> str:
    """Remove middle initials for comparison: 'John A. Smith' → 'john smith'."""
    parts = _normalize(name).split()
    # Remove single-char parts (middle initials)
    parts = [p for p in parts if len(p) > 1]
    return ' '.join(parts)


class EntityMerger:
    """
    Deduplicates and merges ci_entities for a given run.
    """

    def __init__(self, llm_clients: dict = None, usage_tracker=None):
        self.llm_clients = llm_clients or {}
        self.usage_tracker = usage_tracker

    def merge_run_entities(self, run_id: str) -> Dict[str, int]:
        """
        Run the full merge pipeline for a CI run.

        Returns: {merged: N, total: M}
        """
        entities = [dict(e) for e in get_ci_entities(run_id)]
        if len(entities) < 2:
            return {'merged': 0, 'total': len(entities)}

        merged_count = 0
        merged_count += self._deterministic_merge(run_id, entities)

        # Re-fetch active entities after deterministic pass
        active = [dict(e) for e in get_ci_entities_active(run_id)]
        if len(active) >= 2 and self.llm_clients:
            merged_count += self._ai_merge(run_id, active)

        total_active = len([dict(e) for e in get_ci_entities_active(run_id)])
        if merged_count:
            logger.info(f"EntityMerger: run {run_id} — merged {merged_count} duplicates, "
                        f"{total_active} active entities remain")
        return {'merged': merged_count, 'total': total_active}

    # -------------------------------------------------------------------
    # Pass 1 — Deterministic merge
    # -------------------------------------------------------------------

    def _deterministic_merge(self, run_id: str, entities: List[Dict]) -> int:
        """Merge entities by exact/normalized name match within each type."""
        merged = 0
        # Group by type
        by_type: Dict[str, List[Dict]] = {}
        for e in entities:
            by_type.setdefault(e['entity_type'], []).append(e)

        for etype, group in by_type.items():
            # Build canonical buckets keyed by normalized name
            # The canonical is the first entity seen with that key
            buckets: Dict[str, Dict] = {}  # norm_key → canonical entity

            for entity in sorted(group, key=lambda x: x['id']):
                if entity.get('merged_into'):
                    continue
                mid_key = _middle_initial_key(entity['name'])
                norm_key = _normalize(entity['name'])

                # Try exact normalized match first, then middle-initial match
                canon_key = None
                if norm_key in buckets:
                    canon_key = norm_key
                elif mid_key in buckets:
                    canon_key = mid_key

                if canon_key:
                    canonical = buckets[canon_key]
                    if canonical['id'] != entity['id']:
                        # Merge entity into canonical
                        self._do_merge(canonical, entity)
                        merged += 1
                else:
                    # Register as canonical under both keys
                    buckets[norm_key] = entity
                    if mid_key and mid_key != norm_key:
                        buckets[mid_key] = entity

        return merged

    def _do_merge(self, canonical: Dict, duplicate: Dict):
        """Flag duplicate as merged into canonical; union aliases + provenance."""
        # Union aliases — guard against 'null' stored as JSON (json.loads('null') → None)
        canon_aliases = set(json.loads(canonical.get('aliases') or '[]') or [])
        dup_aliases = set(json.loads(duplicate.get('aliases') or '[]') or [])
        # Add duplicate's name as alias on canonical
        new_aliases = canon_aliases | dup_aliases | {duplicate['name']}
        new_aliases.discard(canonical['name'])  # don't alias self

        # Union provenance — guard against 'null' JSON value same as aliases
        try:
            canon_prov = json.loads(canonical.get('provenance') or '[]') or []
        except Exception:
            canon_prov = []
        try:
            dup_prov = json.loads(duplicate.get('provenance') or '[]') or []
        except Exception:
            dup_prov = []
        # Deduplicate provenance by doc_id
        seen_ids = {p.get('paperless_doc_id') for p in canon_prov}
        for p in dup_prov:
            if p.get('paperless_doc_id') not in seen_ids:
                canon_prov.append(p)
                seen_ids.add(p.get('paperless_doc_id'))

        update_entity_aliases(canonical['id'], json.dumps(sorted(new_aliases)),
                              json.dumps(canon_prov))
        mark_entity_merged(duplicate['id'], canonical['id'])

        # Update canonical's in-memory aliases for subsequent iterations
        canonical['aliases'] = json.dumps(sorted(new_aliases))
        canonical['provenance'] = json.dumps(canon_prov)

    # -------------------------------------------------------------------
    # Pass 2 — AI-assisted fuzzy merge
    # -------------------------------------------------------------------

    _AI_MERGE_PROMPT = """You are a legal entity deduplication expert.

Below is a list of named entities extracted from legal documents.
Identify groups of entities that refer to THE SAME person or organization.
Consider: typos, abbreviations, middle names, maiden names, name variations,
common legal entity shorthand (MoFo = Morrison Foerster, etc.)

ENTITIES (type: id — name):
{entity_list}

Return ONLY valid JSON — an array of merge groups.
Only include groups with 2+ members (do NOT include singletons):
[
  {{
    "canonical_name": "The most complete/formal name",
    "canonical_id": <id of the canonical entity>,
    "merge_ids": [<ids of duplicates to merge INTO canonical>]
  }}
]
If there are no merge groups, return: []
"""

    def _ai_merge(self, run_id: str, entities: List[Dict]) -> int:
        """Use LLM to identify fuzzy-match merge groups."""
        merged = 0

        # Process in batches of _AI_MERGE_BATCH
        for batch_start in range(0, len(entities), _AI_MERGE_BATCH):
            batch = entities[batch_start:batch_start + _AI_MERGE_BATCH]
            entity_list = '\n'.join(
                f"  {e['entity_type']}: {e['id']} — {e['name']}"
                for e in batch
                if not e.get('merged_into')
            )
            if not entity_list:
                continue

            prompt = self._AI_MERGE_PROMPT.format(entity_list=entity_list)
            result = self._call_llm(prompt)
            if not result:
                continue

            # Build id→entity map for this batch
            id_map = {e['id']: e for e in batch}

            for group in result:
                canon_id = group.get('canonical_id')
                merge_ids = group.get('merge_ids', [])
                if not canon_id or not merge_ids:
                    continue
                canonical = id_map.get(canon_id)
                if not canonical:
                    continue
                for mid in merge_ids:
                    dup = id_map.get(mid)
                    if not dup or dup.get('merged_into') or dup['id'] == canon_id:
                        continue
                    self._do_merge(canonical, dup)
                    merged += 1

        return merged

    def _call_llm(self, prompt: str) -> Optional[List[Dict]]:
        """Call LLM for AI merge pass. Returns parsed list or None.
        Routed through proxy pool with direct-provider fallback."""
        from analyzer.llm.proxy_call import call_llm_json

        # Determine direct-provider hint: prefer openai then anthropic
        provider = 'openai' if 'openai' in self.llm_clients else (
            'anthropic' if 'anthropic' in self.llm_clients else None
        )
        client = self.llm_clients.get(provider) if provider else None
        default_model = 'gpt-4o-mini' if provider == 'openai' else 'claude-haiku-4-5-20251001'

        parsed = call_llm_json(
            prompt,
            task='entity',
            max_tokens=2000,
            provider=provider,
            api_key=getattr(client, 'api_key', None) if client else None,
            model=default_model,
            operation='ci:entity_merge',
            usage_tracker=self.usage_tracker,
        )
        if parsed is None:
            return None
        if isinstance(parsed, list):
            return parsed
        # Some models return {groups: [...]}
        for key in ('groups', 'merge_groups', 'entities'):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return []
