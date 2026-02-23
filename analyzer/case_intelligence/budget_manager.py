"""
Budget Manager for Case Intelligence AI.

Enforces per-run and monthly budget ceilings. Called before every LLM task
to check whether the run has budget remaining, and after every task to record
the actual cost.
"""

import logging
from typing import Optional
from analyzer.case_intelligence.db import get_ci_run, update_ci_run, increment_ci_run_cost

logger = logging.getLogger(__name__)


class BudgetManager:
    """
    Budget enforcement for CI runs.

    Usage pattern:
        if budget_manager.check_and_charge(run_id, 'entity_extraction', 0.002):
            # proceed with task
            actual_cost = run_llm_task(...)
            budget_manager.record_actual_cost(run_id, actual_cost)
        else:
            # run is budget_blocked — stop
    """

    def check_and_charge(self, run_id: str, task_type: str,
                         estimated_cost: float) -> bool:
        """
        Check if the task can proceed within budget.

        Returns True if allowed (and reserves the estimated cost).
        Returns False if budget would be exceeded (sets budget_blocked on run).
        """
        run = get_ci_run(run_id)
        if not run:
            logger.error(f"BudgetManager: run {run_id} not found")
            return False

        cost_so_far = run['cost_so_far_usd'] or 0.0
        budget = run['budget_per_run_usd'] or 10.0

        if cost_so_far + estimated_cost > budget:
            note = (
                f"Budget ceiling ${budget:.2f} reached at ${cost_so_far:.4f}. "
                f"Task '{task_type}' (est. ${estimated_cost:.4f}) would exceed budget. "
                f"To continue, increase the budget per run and start a new run."
            )
            update_ci_run(run_id, budget_blocked=1, budget_blocked_note=note,
                          status='budget_blocked')
            logger.warning(f"CI run {run_id} budget blocked: {note}")
            return False

        # Reserve the estimated cost now (actual may differ)
        increment_ci_run_cost(run_id, estimated_cost)
        return True

    def record_actual_cost(self, run_id: str, actual_cost: float,
                           estimated_cost: float = 0.0):
        """
        Called after an LLM task completes with the actual token cost.
        Adjusts cost_so_far by (actual - estimated) to reconcile the reservation.
        """
        adjustment = actual_cost - estimated_cost
        if abs(adjustment) > 0.000001:
            increment_ci_run_cost(run_id, adjustment)

    def get_remaining_budget(self, run_id: str) -> Optional[float]:
        """Return remaining budget in USD, or None if run not found."""
        run = get_ci_run(run_id)
        if not run:
            return None
        budget = run['budget_per_run_usd'] or 10.0
        cost = run['cost_so_far_usd'] or 0.0
        return max(0.0, budget - cost)

    def is_blocked(self, run_id: str) -> bool:
        """Return True if the run is currently budget_blocked."""
        run = get_ci_run(run_id)
        if not run:
            return True
        return bool(run['budget_blocked'])
