"""
Jurisdiction Profile model for Case Intelligence AI.

Pre-built profiles for NYS, SDNY, EDNY, and related courts.
Each profile specifies the procedural framework, local rules, and
which authority sources to search during a CI run.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


@dataclass
class JurisdictionProfile:
    """Complete description of a case's procedural context."""
    jurisdiction_id: str               # e.g., "nys-supreme-civil"
    display_name: str
    baseline_framework: str            # "CPLR", "FRBP", "FRCP", "FCA", "SCPA"
    court: str
    county: Optional[str] = None       # "New York", "Kings", "Queens", etc.
    judge_part: Optional[str] = None   # "Part 61"
    part_rules_notes: Optional[str] = None
    bankruptcy_overlay: Optional[str] = None  # "Chapter 7", "Chapter 11", None
    local_rules: List[str] = field(default_factory=list)
    authority_jurisdictions: List[str] = field(default_factory=list)
    custom_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'JurisdictionProfile':
        d = json.loads(json_str)
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict) -> 'JurisdictionProfile':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Pre-built profiles (NYS-first, SDNY, EDNY, FCA, SCPA)
JURISDICTION_PROFILES: Dict[str, JurisdictionProfile] = {
    "nys-supreme-civil": JurisdictionProfile(
        jurisdiction_id="nys-supreme-civil",
        display_name="NYS Supreme Court (Civil)",
        baseline_framework="CPLR",
        court="NYS Supreme Court",
        local_rules=["22 NYCRR Part 202"],
        authority_jurisdictions=["NYS", "2nd Circuit", "US"],
    ),
    "nys-commercial-division": JurisdictionProfile(
        jurisdiction_id="nys-commercial-division",
        display_name="NYS Supreme Court – Commercial Division",
        baseline_framework="CPLR",
        court="NYS Supreme Court - Commercial Division",
        local_rules=["22 NYCRR Part 202", "Commercial Division Rules (22 NYCRR Part 202-a)"],
        authority_jurisdictions=["NYS", "2nd Circuit", "US"],
    ),
    "sdny-civil": JurisdictionProfile(
        jurisdiction_id="sdny-civil",
        display_name="SDNY (Civil)",
        baseline_framework="FRCP",
        court="S.D.N.Y.",
        local_rules=["S.D.N.Y. Local Civil Rules"],
        authority_jurisdictions=["2nd Circuit", "SDNY", "US"],
    ),
    "sdny-bankruptcy-ch7": JurisdictionProfile(
        jurisdiction_id="sdny-bankruptcy-ch7",
        display_name="SDNY Bankruptcy Court – Chapter 7",
        baseline_framework="FRBP",
        court="S.D.N.Y. Bankruptcy Court",
        bankruptcy_overlay="Chapter 7",
        local_rules=["S.D.N.Y. Local Bankruptcy Rules"],
        authority_jurisdictions=["2nd Circuit", "SDNY", "US"],
    ),
    "sdny-bankruptcy-ch11": JurisdictionProfile(
        jurisdiction_id="sdny-bankruptcy-ch11",
        display_name="SDNY Bankruptcy Court – Chapter 11",
        baseline_framework="FRBP",
        court="S.D.N.Y. Bankruptcy Court",
        bankruptcy_overlay="Chapter 11",
        local_rules=["S.D.N.Y. Local Bankruptcy Rules"],
        authority_jurisdictions=["2nd Circuit", "SDNY", "US"],
    ),
    "nys-family-court": JurisdictionProfile(
        jurisdiction_id="nys-family-court",
        display_name="NYS Family Court",
        baseline_framework="FCA",
        court="NYS Family Court",
        local_rules=["Family Court Act", "22 NYCRR Part 205"],
        authority_jurisdictions=["NYS", "2nd Circuit", "US"],
    ),
    "nys-surrogate": JurisdictionProfile(
        jurisdiction_id="nys-surrogate",
        display_name="NYS Surrogate's Court",
        baseline_framework="SCPA",
        court="NYS Surrogate's Court",
        local_rules=["Surrogate's Court Procedure Act", "22 NYCRR Part 207"],
        authority_jurisdictions=["NYS", "2nd Circuit", "US"],
    ),
    "edny-civil": JurisdictionProfile(
        jurisdiction_id="edny-civil",
        display_name="EDNY (Civil)",
        baseline_framework="FRCP",
        court="E.D.N.Y.",
        local_rules=["E.D.N.Y. Local Civil Rules"],
        authority_jurisdictions=["2nd Circuit", "EDNY", "US"],
    ),
    "edny-bankruptcy": JurisdictionProfile(
        jurisdiction_id="edny-bankruptcy",
        display_name="EDNY Bankruptcy Court",
        baseline_framework="FRBP",
        court="E.D.N.Y. Bankruptcy Court",
        local_rules=["E.D.N.Y. Local Bankruptcy Rules"],
        authority_jurisdictions=["2nd Circuit", "EDNY", "US"],
    ),
    "nys-appellate-div-1": JurisdictionProfile(
        jurisdiction_id="nys-appellate-div-1",
        display_name="NYS Appellate Division – 1st Department",
        baseline_framework="CPLR",
        court="NYS Appellate Division, 1st Department",
        local_rules=["22 NYCRR Part 1250"],
        authority_jurisdictions=["NYS", "2nd Circuit", "US"],
    ),
    "nys-appellate-div-2": JurisdictionProfile(
        jurisdiction_id="nys-appellate-div-2",
        display_name="NYS Appellate Division – 2nd Department",
        baseline_framework="CPLR",
        court="NYS Appellate Division, 2nd Department",
        local_rules=["22 NYCRR Part 1250"],
        authority_jurisdictions=["NYS", "2nd Circuit", "US"],
    ),
    "custom": JurisdictionProfile(
        jurisdiction_id="custom",
        display_name="Custom / Other",
        baseline_framework="",
        court="",
        local_rules=[],
        authority_jurisdictions=["US"],
    ),
}


def get_jurisdiction_profile(jurisdiction_id: str) -> Optional[JurisdictionProfile]:
    """Return a pre-built profile, or None if not found."""
    return JURISDICTION_PROFILES.get(jurisdiction_id)


def list_jurisdiction_profiles() -> List[dict]:
    """Return all profiles as dicts for the API."""
    return [
        {
            "jurisdiction_id": p.jurisdiction_id,
            "display_name": p.display_name,
            "baseline_framework": p.baseline_framework,
            "court": p.court,
            "bankruptcy_overlay": p.bankruptcy_overlay,
        }
        for p in JURISDICTION_PROFILES.values()
    ]
