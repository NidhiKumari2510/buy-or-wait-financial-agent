"""
code/ranker.py

Deterministic ranking of safe, eligible candidates per problem_statement.md's
"Choosing Between Safe Plans" 6-rule order:
  1. Complete the full request by desired_completion_date.
  2. Require no spending changes.
  3. Minimize the total amount paid.
  4. Start payment earlier.
  5. Use fewer payments.
  6. Lowest payment_option_id as final tie-breaker.
No LLM involved -- pure sort.
"""

from __future__ import annotations

from datetime import date

from models import FinancialState
from planner import Candidate


def _completion_date(candidate: Candidate) -> date:
    if candidate.method == "wait":
        return candidate.earliest_full_payment_date or date.max
    if not candidate.payments:
        return date.max
    return candidate.payments[-1][0]


def is_eligible(candidate: Candidate, state: FinancialState) -> bool:
    accepted = set(state.profile.payment_methods_user_will_consider)
    if candidate.method in ("full_payment", "partial_payment", "installments"):
        return candidate.method in accepted
    if candidate.method == "wait":
        return "full_payment" in accepted
    return False


def _sort_key(candidate: Candidate, state: FinancialState):
    completes_by_deadline = _completion_date(candidate) <= state.request.desired_completion_date
    start_date = candidate.payments[0][0] if candidate.payments else date.max
    option_id = candidate.payment_option_id or "zzzzzzzzzz"
    return (
        0 if completes_by_deadline else 1,   # rule 1
        len(candidate.spending_changes),      # rule 2
        candidate.total_paid,                 # rule 3
        start_date,                           # rule 4
        len(candidate.payments),              # rule 5
        option_id,                            # rule 6
    )


def rank_candidates(candidates: list[Candidate], state: FinancialState) -> list[Candidate]:
    """Returns only eligible+safe candidates, best first."""
    eligible_safe = [c for c in candidates if c.is_safe and is_eligible(c, state)]
    eligible_safe.sort(key=lambda c: _sort_key(c, state))
    return eligible_safe