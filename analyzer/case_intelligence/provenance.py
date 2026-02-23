"""
Provenance dataclasses for Case Intelligence AI.

Every finding, entity, timeline event, and theory must cite the documents
that support it. Provenance objects are stored as JSON in TEXT fields.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json


@dataclass
class Provenance:
    """A citation back to a specific location in a case document."""
    paperless_doc_id: int
    page_number: Optional[int] = None
    offset: Optional[int] = None
    bates_label: Optional[str] = None
    excerpt: str = ""
    extraction_version: Optional[str] = None  # ISO timestamp of run
    prompt_version: Optional[str] = None      # e.g., "entity_v1"
    model_used: Optional[str] = None          # e.g., "gpt-4o-mini"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class HypothesisProvenance:
    """A provenance entry for a hypothesis (no document source)."""
    hypothesis: bool = True
    excerpt: str = ""
    what_would_confirm: str = ""
    extraction_version: Optional[str] = None
    prompt_version: Optional[str] = None
    model_used: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class AuthorityProvenance:
    """A citation to an external legal authority."""
    citation: str                          # e.g., "CPLR § 3212"
    authority_type: str                    # statute|regulation|case_law|rule|secondary
    jurisdiction: str                      # "NYS", "SDNY", "US"
    source: str                            # courtlistener|ecfr|nysenate|web_search|manual
    source_url: Optional[str] = None
    retrieval_date: Optional[str] = None
    reliability: str = "official"          # official|unofficial|unknown
    excerpt: Optional[str] = None
    relevance_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def provenance_list_to_json(provenance_list: List) -> str:
    """Serialize a list of provenance objects to a JSON string."""
    if not provenance_list:
        return "[]"
    return json.dumps([p.to_dict() if hasattr(p, 'to_dict') else p for p in provenance_list])


def json_to_provenance_list(json_str: str) -> List[dict]:
    """Deserialize a JSON string to a list of provenance dicts."""
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def missing_required_citations(result: dict) -> bool:
    """
    Check if an LLM result is missing mandatory citations.
    Used to decide whether to escalate to a stronger model.
    """
    if not isinstance(result, dict):
        return True
    # Check for provenance in common locations
    for key in ('provenance', 'evidence', 'supporting_evidence', 'citations'):
        if key in result:
            val = result[key]
            if isinstance(val, list) and len(val) > 0:
                return False
    # If the result has findings/entities/events, each should have provenance
    for key in ('findings', 'entities', 'events'):
        if key in result and isinstance(result[key], list):
            for item in result[key]:
                if isinstance(item, dict) and not item.get('provenance'):
                    return True  # At least one item missing provenance
    return False
