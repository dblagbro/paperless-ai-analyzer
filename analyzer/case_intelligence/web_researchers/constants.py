"""Constants shared by every WebResearcher provider — rate limits, jurisdiction
codes, role-aware keyword prefixes.

Extracted from web_researcher.py during the v3.9.8 split."""
from typing import Dict

# ── Rate limits (min seconds between calls per source) ───────────────────────
_RATE = {
    'courtlistener': 1.2,
    'caselaw':       0.8,
    'ddg':           2.5,
    'brave':         0.5,
    'google_cse':    0.3,
    'exa':           0.5,
    'perplexity':    1.5,
    'gdelt':         1.2,
    'bop':           3.0,
    'ofac':          1.0,
    'fec':           1.2,
    'sec_edgar':     1.0,
    'opensanctions': 0.8,
    'opencorporates':1.0,
    'newsapi':       0.5,
    'docket_alarm':  2.0,
    'unicourt':      2.0,
    'vlex':          1.0,
    'westlaw':       1.0,
    'clear':         2.0,
    'tavily':        0.5,
    'serper':        0.5,
    'lexisnexis':    1.0,
}


# CourtListener court abbreviations for US states
_STATE_TO_CL: Dict[str, str] = {
    'alabama': 'ala', 'alaska': 'alaska', 'arizona': 'ariz', 'arkansas': 'ark',
    'california': 'cal', 'colorado': 'colo', 'connecticut': 'conn', 'delaware': 'del',
    'florida': 'fla', 'georgia': 'ga', 'hawaii': 'haw', 'idaho': 'idaho',
    'illinois': 'ill', 'indiana': 'ind', 'iowa': 'iowa', 'kansas': 'kan',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'mass', 'michigan': 'mich', 'minnesota': 'minn',
    'mississippi': 'miss', 'missouri': 'mo', 'montana': 'mont', 'nebraska': 'neb',
    'nevada': 'nev', 'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm',
    'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'ohio',
    'oklahoma': 'okla', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri',
    'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tenn', 'texas': 'tex',
    'utah': 'utah', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wash',
    'west virginia': 'wva', 'wisconsin': 'wis', 'wyoming': 'wyo',
}


# Role-aware query prefixes for legal authority search
_ROLE_AUTHORITY_PREFIX: Dict[str, str] = {
    'plaintiff':    'plaintiff prevailing similar facts judgment damages',
    'prosecution':  'prosecution successful conviction precedent element proof',
    'defense':      'defense prevailing acquittal dismissal suppression reversal',
    'neutral':      'relevant case law precedent',
}


# Role-aware entity character research keywords
_ROLE_ENTITY_PERSON: Dict[str, str] = {
    'plaintiff':    'credibility reliability expert bias civil history',
    'prosecution':  'prior conviction criminal history arrest modus operandi',
    'defense':      'impeachment false testimony inconsistency credibility bias financial motivation',
    'neutral':      'background history credibility',
}
_ROLE_ENTITY_ORG: Dict[str, str] = {
    'plaintiff':    'prior complaints regulatory violations consumer fraud lawsuit',
    'prosecution':  'prior offenses regulatory action criminal enterprise pattern',
    'defense':      'reputation litigation history regulatory compliance',
    'neutral':      'lawsuit regulatory history',
}
