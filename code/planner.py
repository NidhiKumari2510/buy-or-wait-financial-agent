"""
code/planner.py

Generates every candidate plan for a request: full_payment, partial_payment,
installments (one per supplied payment_option), wait, and spending-changes
variants of full_payment. Each candidate is tested for safety via
simulator.simulate(). Deterministic -- no LLM calls here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import config
from data_loader import DataContext
from models import FinancialEvent, FinancialState, PaymentOption
from simulator import simulate

FLEXIBLE_KINDS = {
    config.EVENT_FLEXIBILITY_STOPPABLE,
    config.EVENT_FLEXIBILITY_REDUCIBLE,
    config.EVENT_FLEXIBILITY_REDUCIBLE_OR_STOPPABLE,
}


@dataclass
class SpendingChange:
    change_type: str              # "stop" or "reduce_to"
    event_id: str
    new_amount: Optional[float]   # None for stop
    category: str
    saving: float                 # amount freed up (home currency)


@dataclass
class Candidate:
    method: str                               # full_payment | partial_payment | installments | wait | not_recommended
    payments: list[tuple[date, float]]        # chronological (date, amount)
    total_paid: float
    is_safe: bool
    payment_option_id: Optional[str] = None   # set for installments
    spending_changes: list[SpendingChange] = field(default_factory=list)
    earliest_full_payment_date: Optional[date] = None
    amount_safe_to_pay: float = 0.0


# ---------------------------------------------------------------------------
# amount_safe_to_pay via bisection (no spending changes)
# ---------------------------------------------------------------------------

def compute_amount_safe_to_pay(state: FinancialState) -> float:
    requested = state.request.requested_amount
    request_date = state.request.request_date

    full = simulate(state, [(request_date, requested)])
    if full.is_safe:
        return requested

    zero_check = simulate(state, [(request_date, 0.0)])
    if not zero_check.is_safe:
        return 0.0  # user is already below minimum balance regardless of this purchase

    lo, hi = 0.0, requested
    for _ in range(60):
        mid = (lo + hi) / 2
        result = simulate(state, [(request_date, mid)])
        if result.is_safe:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


# ---------------------------------------------------------------------------
# earliest_date_for_full_payment (no spending changes)
# ---------------------------------------------------------------------------

def compute_earliest_full_payment_date(state: FinancialState) -> Optional[date]:
    request_date = state.request.request_date
    window_end = request_date + timedelta(days=config.FORECAST_DAYS)
    requested = state.request.requested_amount

    d = request_date
    while d <= window_end:
        result = simulate(state, [(d, requested)])
        if result.is_safe:
            return d
        d += timedelta(days=1)
    return None


# ---------------------------------------------------------------------------
# Spending changes: pick the most recent occurrence per flexible category,
# largest-saving first, up to config.MAX_SPENDING_CHANGES, skipping protected
# categories.
# ---------------------------------------------------------------------------

def _flexible_candidates(state: FinancialState) -> list[SpendingChange]:
    protected = set(state.profile.expense_categories_to_protect)
    stop_categories = set(state.profile.expense_categories_user_is_willing_to_stop)
    reduce_categories = set(state.profile.expense_categories_user_is_willing_to_reduce)

    best_per_category: dict[str, FinancialEvent] = {}
    for e in state.events:
        if e.flexibility not in FLEXIBLE_KINDS:
            continue
        if e.category in protected:
            continue
        if e.direction != config.EVENT_DIRECTION_DEBIT:
            continue
        if e.status in config.EVENT_STATUSES_ALWAYS_EXCLUDE:
            continue
        amt = state.home_amounts.get(e.event_id)
        if amt is None:
            continue
        current = best_per_category.get(e.category)
        if current is None or e.event_date > current.event_date:
            best_per_category[e.category] = e

    changes: list[SpendingChange] = []
    for category, event in best_per_category.items():
        amt = state.home_amounts[event.event_id]
        can_stop = category in stop_categories and event.flexibility in (
            config.EVENT_FLEXIBILITY_STOPPABLE, config.EVENT_FLEXIBILITY_REDUCIBLE_OR_STOPPABLE
        )
        can_reduce = category in reduce_categories and event.flexibility in (
            config.EVENT_FLEXIBILITY_REDUCIBLE, config.EVENT_FLEXIBILITY_REDUCIBLE_OR_STOPPABLE
        )

        if can_stop:
            changes.append(SpendingChange("stop", event.event_id, None, category, saving=amt))
        elif can_reduce:
            floor = event.minimum_allowed_amount if event.minimum_allowed_amount is not None else 0.0
            saving = max(0.0, amt - floor)
            if saving > 0:
                changes.append(SpendingChange("reduce_to", event.event_id, floor, category, saving=saving))

    changes.sort(key=lambda c: c.saving, reverse=True)
    return changes[: config.MAX_SPENDING_CHANGES]


def _apply_changes_to_state_copy(state: FinancialState, changes: list[SpendingChange]) -> FinancialState:
    """Simulate() reads state.events and state.home_amounts directly, so we
    build a shallow-modified copy: same object, but with adjusted home_amounts
    for the chosen events. We restore afterward -- see try/finally callers."""
    for change in changes:
        if change.change_type == "stop":
            state.home_amounts[change.event_id] = 0.0
        else:
            state.home_amounts[change.event_id] = change.new_amount
    return state


def _restore_amounts(state: FinancialState, changes: list[SpendingChange], original: dict[str, Optional[float]]) -> None:
    for change in changes:
        state.home_amounts[change.event_id] = original[change.event_id]


def try_full_payment_with_spending_changes(state: FinancialState) -> Optional[Candidate]:
    """Only called when plain full_payment already failed. Applies flexible
    spending changes one at a time (largest saving first) until safe or
    until MAX_SPENDING_CHANGES is exhausted."""
    available = _flexible_candidates(state)
    if not available:
        return None

    requested = state.request.requested_amount
    request_date = state.request.request_date

    for n in range(1, len(available) + 1):
        chosen = available[:n]
        original = {c.event_id: state.home_amounts[c.event_id] for c in chosen}
        _apply_changes_to_state_copy(state, chosen)
        try:
            result = simulate(state, [(request_date, requested)])
            if result.is_safe:
                return Candidate(
                    method="full_payment",
                    payments=[(request_date, requested)],
                    total_paid=requested,
                    is_safe=True,
                    spending_changes=chosen,
                    earliest_full_payment_date=request_date,
                    amount_safe_to_pay=requested,
                )
        finally:
            _restore_amounts(state, chosen, original)
    return None


# ---------------------------------------------------------------------------
# Candidate builders
# ---------------------------------------------------------------------------

def build_full_payment_candidate(state: FinancialState) -> Candidate:
    requested = state.request.requested_amount
    request_date = state.request.request_date
    result = simulate(state, [(request_date, requested)])
    return Candidate(
        method="full_payment",
        payments=[(request_date, requested)],
        total_paid=requested,
        is_safe=result.is_safe,
        earliest_full_payment_date=request_date if result.is_safe else None,
        amount_safe_to_pay=requested if result.is_safe else 0.0,
    )


def build_partial_payment_candidate(
    state: FinancialState, amount_safe_to_pay: float, earliest_full_date: Optional[date]
) -> Optional[Candidate]:
    req = state.request
    if not req.allows_partial_payment:
        return None
    if not (0 < amount_safe_to_pay < req.requested_amount):
        return None
    if earliest_full_date is None or earliest_full_date > req.desired_completion_date:
        return None

    remainder = round(req.requested_amount - amount_safe_to_pay, 2)
    payments = [(req.request_date, amount_safe_to_pay), (earliest_full_date, remainder)]
    result = simulate(state, payments)
    return Candidate(
        method="partial_payment",
        payments=payments,
        total_paid=req.requested_amount,
        is_safe=result.is_safe,
        earliest_full_payment_date=earliest_full_date,
        amount_safe_to_pay=amount_safe_to_pay,
    )


def build_installment_candidates(state: FinancialState) -> list[Candidate]:
    out = []
    for opt in state.payment_options:
        if opt.payment_method != "installments":
            continue
        if state.profile.max_installment_months is not None:
            span_days = (opt.schedule()[-1][0] - opt.first_payment_date).days if opt.number_of_payments > 1 else 0
            if span_days > state.profile.max_installment_months * 31:
                continue  # exceeds user's max_installment_months preference
        payments = opt.schedule()
        result = simulate(state, payments)
        out.append(
            Candidate(
                method="installments",
                payments=payments,
                total_paid=opt.total_payable_amount,
                is_safe=result.is_safe,
                payment_option_id=opt.payment_option_id,
                earliest_full_payment_date=payments[-1][0] if result.is_safe else None,
                amount_safe_to_pay=payments[0][1] if result.is_safe else 0.0,
            )
        )
    return out


def build_wait_candidate(state: FinancialState, earliest_full_date: Optional[date], full_payment_safe_today: bool) -> Optional[Candidate]:
    """Only a distinct candidate when full payment becomes safe LATER than
    request_date -- if it's already safe today, 'wait' is redundant with
    full_payment and must not be offered as a separate option."""
    if earliest_full_date is None:
        return None
    if full_payment_safe_today:
        return None
    if earliest_full_date <= state.request.request_date:
        return None
    requested = state.request.requested_amount
    return Candidate(
        method="wait",
        payments=[(earliest_full_date, requested)],
        total_paid=requested,
        is_safe=True,
        earliest_full_payment_date=earliest_full_date,
        amount_safe_to_pay=0.0,  # nothing is paid today under "wait"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_candidates(state: FinancialState) -> tuple[list[Candidate], float, Optional[date]]:
    """Returns (all_generated_candidates, amount_safe_to_pay, earliest_date_for_full_payment).
    The two scalars are computed once, without spending changes, per spec
    ("before optional spending changes") -- they're used for output fields
    regardless of which payment method ends up recommended."""
    amount_safe_to_pay = compute_amount_safe_to_pay(state)
    earliest_full_date = compute_earliest_full_payment_date(state)

    candidates: list[Candidate] = []

    full = build_full_payment_candidate(state)
    candidates.append(full)

    if not full.is_safe:
        with_changes = try_full_payment_with_spending_changes(state)
        if with_changes is not None:
            candidates.append(with_changes)

    partial = build_partial_payment_candidate(state, amount_safe_to_pay, earliest_full_date)
    if partial is not None:
        candidates.append(partial)

    candidates.extend(build_installment_candidates(state))

    wait = build_wait_candidate(state, earliest_full_date, full_payment_safe_today=full.is_safe)
    if wait is not None:
        candidates.append(wait)

    return candidates, amount_safe_to_pay, earliest_full_date


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import data_loader
    from financial_state import build_financial_state

    ctx = data_loader.load_all()
    sample_request_id = next(iter(ctx.requests))
    state = build_financial_state(ctx, sample_request_id)

    candidates, amount_safe, earliest_date = generate_candidates(state)
    print(f"Candidates for {sample_request_id} (requested_amount={state.request.requested_amount}):")
    print(f"  amount_safe_to_pay (no changes): {amount_safe}")
    print(f"  earliest_date_for_full_payment:  {earliest_date}")
    for c in candidates:
        print(f"  - {c.method:14s} safe={c.is_safe!s:5s} total_paid={c.total_paid:.2f} "
              f"payments={c.payments} spending_changes={[sc.event_id for sc in c.spending_changes]}")