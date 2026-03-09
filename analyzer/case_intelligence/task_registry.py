"""
Task Registry for Case Intelligence AI.

Defines all LLM tasks, their tier assignments, model routing, and
estimated cost per document. Used by BudgetManager and the orchestrator.

5-Tier system (v3.6.6):
  Tier 1 — Junior Associate:   Extraction only
  Tier 2 — Senior Associate:   + Contradictions + Basic theories + Web
  Tier 3 — Partner:            + Adversarial + Forensic + Discovery + Paid web
  Tier 4 — Senior Partner:     + Witnesses + War Room + Senior partner review
  Tier 5 — White Glove:        + Deep forensics + Trial strategy + Multi-model
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TaskDefinition:
    """Definition of a single LLM task within the CI pipeline."""
    task_type: str
    tier: int                          # 1–5 (new 5-tier system)
    primary_model: str
    primary_provider: str
    escalate_model: str
    escalate_provider: str
    estimated_cost_per_doc: float      # USD estimate
    description: str
    requires_citation: bool = True     # whether output must have provenance
    max_output_tokens: int = 2000
    is_per_run: bool = False           # True = fixed per run (not scaled by doc count)


# Task registry — maps task_type → TaskDefinition
TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "generate_questions": TaskDefinition(
        task_type="generate_questions",
        tier=1,
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        escalate_model="claude-haiku-4-5-20251001",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.001,
        description="Generate clarifying questions for a CI run",
        requires_citation=False,
        max_output_tokens=1000,
    ),
    "entity_extraction": TaskDefinition(
        task_type="entity_extraction",
        tier=1,
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        escalate_model="claude-haiku-4-5-20251001",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.002,
        description="Extract people, orgs, accounts, properties from a document",
        max_output_tokens=8000,
    ),
    "timeline_extraction": TaskDefinition(
        task_type="timeline_extraction",
        tier=1,
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        escalate_model="claude-haiku-4-5-20251001",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.002,
        description="Extract dated events and their significance",
        max_output_tokens=8000,
    ),
    "financial_extraction": TaskDefinition(
        task_type="financial_extraction",
        tier=1,
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        escalate_model="claude-haiku-4-5-20251001",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.003,
        description="Extract financial amounts, balances, and transfers",
        max_output_tokens=8000,
    ),
    "contradiction_detection": TaskDefinition(
        task_type="contradiction_detection",
        tier=2,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.015,
        description="Detect conflicting facts across documents",
        max_output_tokens=8000,
    ),
    "disputed_facts_matrix": TaskDefinition(
        task_type="disputed_facts_matrix",
        tier=2,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.010,
        description="Build opposing positions on key facts",
        max_output_tokens=8000,
    ),
    "authority_relevance": TaskDefinition(
        task_type="authority_relevance",
        tier=2,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.008,
        description="Assess relevance of retrieved legal authorities",
        max_output_tokens=2000,
    ),
    "theory_generation": TaskDefinition(
        task_type="theory_generation",
        tier=3,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.05,
        description="Generate role-aware factual and legal theories",
        max_output_tokens=8000,
    ),
    "adversarial_testing": TaskDefinition(
        task_type="adversarial_testing",
        tier=3,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.05,
        description="Attempt to falsify generated theories",
        max_output_tokens=8000,
    ),
    "opposing_theory_generation": TaskDefinition(
        task_type="opposing_theory_generation",
        tier=3,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.05,
        description="Generate opposing-perspective theories for counter-argument awareness",
        max_output_tokens=8000,
    ),
    "report_generation": TaskDefinition(
        task_type="report_generation",
        tier=2,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.02,
        description="Generate attorney-ready memo or report",
        requires_citation=False,
        max_output_tokens=8000,
    ),
    "findings_summary": TaskDefinition(
        task_type="findings_summary",
        tier=2,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.010,
        description="Compile key findings summary",
        requires_citation=False,
        max_output_tokens=3000,
        is_per_run=True,
    ),

    # ── Tier 3: Partner tasks (v3.6.6) ──────────────────────────────────────
    "forensic_accounting": TaskDefinition(
        task_type="forensic_accounting",
        tier=3,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.025,
        description="Forensic accounting: structuring, round-number anomalies, cash flow gaps",
        requires_citation=True,
        max_output_tokens=8000,
        is_per_run=True,
    ),
    "discovery_gap_analysis": TaskDefinition(
        task_type="discovery_gap_analysis",
        tier=3,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.020,
        description="Discovery gap analysis: missing docs, spoliation, custodian gaps, RFP list",
        requires_citation=True,
        max_output_tokens=8000,
        is_per_run=True,
    ),

    # ── Tier 4: Senior Partner tasks (v3.6.6) ───────────────────────────────
    "witness_intelligence": TaskDefinition(
        task_type="witness_intelligence",
        tier=4,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.030,
        description="Witness intelligence: impeachment points, financial interest, deposition prep",
        requires_citation=True,
        max_output_tokens=6000,
        is_per_run=False,   # per-person
    ),
    "war_room_strategy": TaskDefinition(
        task_type="war_room_strategy",
        tier=4,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.060,
        description="War room: opposing case theory, dangerous arguments, settlement valuation",
        requires_citation=True,
        max_output_tokens=8000,
        is_per_run=True,
    ),
    "senior_partner_review": TaskDefinition(
        task_type="senior_partner_review",
        tier=4,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-sonnet-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.030,
        description="Senior partner review: challenge D2 output, find missed issues",
        requires_citation=False,
        max_output_tokens=4000,
        is_per_run=True,
    ),

    # ── Tier 5: White Glove tasks (v3.6.6) ──────────────────────────────────
    "deep_financial_forensics": TaskDefinition(
        task_type="deep_financial_forensics",
        tier=5,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-opus-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.050,
        description="Deep financial forensics: Benford's law, beneficial ownership, structuring",
        requires_citation=True,
        max_output_tokens=8000,
        is_per_run=True,
    ),
    "trial_strategy": TaskDefinition(
        task_type="trial_strategy",
        tier=5,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-opus-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.040,
        description="Trial strategy memo: opening, theme, witness order, key exhibits",
        requires_citation=False,
        max_output_tokens=6000,
        is_per_run=True,
    ),
    "settlement_valuation": TaskDefinition(
        task_type="settlement_valuation",
        tier=5,
        primary_model="gpt-4o",
        primary_provider="openai",
        escalate_model="claude-opus-4-6",
        escalate_provider="anthropic",
        estimated_cost_per_doc=0.030,
        description="Settlement valuation: range, leverage, walk-away, comparable verdicts",
        requires_citation=False,
        max_output_tokens=4000,
        is_per_run=True,
    ),
    "multi_model_synthesis": TaskDefinition(
        task_type="multi_model_synthesis",
        tier=5,
        primary_model="claude-opus-4-6",
        primary_provider="anthropic",
        escalate_model="gpt-4o",
        escalate_provider="openai",
        estimated_cost_per_doc=0.080,
        description="Multi-model comparison: merge Anthropic + OpenAI theories, flag disagreements",
        requires_citation=False,
        max_output_tokens=8000,
        is_per_run=True,
    ),
}


def get_task(task_type: str) -> Optional[TaskDefinition]:
    """Return a TaskDefinition by task_type, or None."""
    return TASK_REGISTRY.get(task_type)


def estimate_run_cost(num_docs: int, max_tier: int = 3) -> dict:
    """
    Estimate total cost for a CI run given document count and max tier.
    Returns dict with 'total_usd' and 'breakdown_by_task'.
    """
    breakdown = {}
    total = 0.0
    for task_def in TASK_REGISTRY.values():
        if task_def.tier > max_tier:
            continue
        if task_def.is_per_run:
            cost = task_def.estimated_cost_per_doc
        else:
            cost = task_def.estimated_cost_per_doc * max(num_docs, 1)
        breakdown[task_def.task_type] = round(cost, 4)
        total += cost

    # witness_intelligence scales by person count (estimate 5 top witnesses)
    if max_tier >= 4 and 'witness_intelligence' in breakdown:
        witness_cost = TASK_REGISTRY['witness_intelligence'].estimated_cost_per_doc * 5
        breakdown['witness_intelligence'] = round(witness_cost, 4)
        # Adjust total for the correction (we already added per-doc above)
        total = sum(breakdown.values())

    return {
        'total_usd': round(total, 4),
        'breakdown_by_task': breakdown,
    }


def estimate_run_cost_simple(num_docs: int, max_tier: int = 3) -> float:
    """Simple float estimate (for backward compat)."""
    return estimate_run_cost(num_docs, max_tier)['total_usd']


# Tier metadata for UI display
TIER_INFO = {
    1: {
        'name': 'Junior Associate',
        'subtitle': 'Extraction only',
        'details': 'Entities, timeline, financial facts',
        'cost_range': '$0.05–1',
        'time_range': '5–30 min',
        'badge_color': '#6c757d',
    },
    2: {
        'name': 'Senior Associate',
        'subtitle': '+ Contradictions + Basic theories',
        'details': 'Cross-doc analysis, web research (free)',
        'cost_range': '$0.50–5',
        'time_range': '15–60 min',
        'badge_color': '#17a2b8',
    },
    3: {
        'name': 'Partner',
        'subtitle': '+ Forensic + Discovery',
        'details': 'Adversarial testing, forensic accounting, discovery gap analysis',
        'cost_range': '$5–50',
        'time_range': '30–120 min',
        'badge_color': '#28a745',
        'recommended': True,
    },
    4: {
        'name': 'Senior Partner',
        'subtitle': '+ Witnesses + War Room',
        'details': 'Witness dossiers, war room strategy, senior partner review',
        'cost_range': '$25–200',
        'time_range': '1–4 hrs',
        'badge_color': '#fd7e14',
    },
    5: {
        'name': 'White Glove',
        'subtitle': '+ Multi-model + Trial strategy',
        'details': 'Deep forensics, trial memo, settlement valuation, multi-model compare',
        'cost_range': '$100–500',
        'time_range': '2–8 hrs',
        'badge_color': '#6f42c1',
    },
}
