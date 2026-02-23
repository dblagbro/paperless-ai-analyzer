"""
Task Registry for Case Intelligence AI.

Defines all LLM tasks, their tier assignments, model routing, and
estimated cost per document. Used by BudgetManager and the orchestrator.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TaskDefinition:
    """Definition of a single LLM task within the CI pipeline."""
    task_type: str
    tier: int                          # 1=cheap, 2=mid, 3=expensive
    primary_model: str
    primary_provider: str
    escalate_model: str
    escalate_provider: str
    estimated_cost_per_doc: float      # USD estimate
    description: str
    requires_citation: bool = True     # whether output must have provenance
    max_output_tokens: int = 2000


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
    ),
}


def get_task(task_type: str) -> Optional[TaskDefinition]:
    """Return a TaskDefinition by task_type, or None."""
    return TASK_REGISTRY.get(task_type)


def estimate_run_cost(num_docs: int, max_tier: int = 3) -> float:
    """
    Estimate total cost for a CI run given document count and max tier.
    Uses per-doc estimates from the task registry.
    """
    total = 0.0
    for task_def in TASK_REGISTRY.values():
        if task_def.tier > max_tier:
            continue
        if task_def.task_type in ('generate_questions', 'findings_summary', 'report_generation'):
            # Fixed-cost tasks (done once per run, not per doc)
            total += task_def.estimated_cost_per_doc
        else:
            total += task_def.estimated_cost_per_doc * num_docs
    return round(total, 4)
