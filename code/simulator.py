"""
code/simulator.py

The 90-Day Safety Check.

Projects a user's balance forward from request_date across FORECAST_DAYS,
applying:
  - settled/scheduled events within the window (home-currency amounts)
  - pending DEBITS: reserved (still subtracted)
  - pending CREDITS: excluded entirely (problem_statement.md: "ignore
    pending credits")
  - cancelled / failed / unrealized statuses: always excluded
  - non_cash direction: always excluded
  - recurring series projected beyond their last explicit occurrence in the
    data (e.g. rent only listed 1 month out still gets projected for the
    rest of the 90-day window)

Candidate payments (the purchase itself, or a payment plan) are injected as
additional debits at specific dates.

CAVEAT: recurring-series detection is a heuristic (longest trailing regular
chain by category+direction, tolerant of one genuine rate-change but not
noise). NOT yet validated against dataset/sample_requests.csv -- that
validation is the next step, and any accuracy issue there should be
diagnosed by printing detect_recurring_series() output for the affected
user before assuming anything else in the pipeline is wrong.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median
from typing import Optional

import config
from financial_state import home_amount
from models import FinancialEvent, FinancialState


# ---------------------------------------------------------------------------
# Recurring series detection
# ---------------------------------------------------------------------------

@dataclass
class RecurringSeries:
    category: str
    direction: str
    interval_days: int
    amount: float          # home-currency, trailing-consistent amount
    last_date: date        # last known occurrence (event_date) used to project forward
    currency: str


INTERVAL_TOLERANCE_DAYS = 4  # allow real-world jitter (weekends, month-length variance)


def _trailing_consistent_amount(amounts: list[float]) -> Optional[float]:
    """Walk backward from the most recent occurrence, keeping a trailing run
    of values within ~5% of each other. Lets a genuine rate change (raise,
    pay cut) win over older history without one noisy outlier derailing the
    whole series."""
    if not amounts:
        return None
    if len(amounts) == 1:
        return amounts[0]

    tail = [amounts[-1]]
    for amt in reversed(amounts[:-1]):
        ref = median(tail)
        if ref == 0:
            break
        if abs(amt - ref) / abs(ref) <= 0.05:
            tail.append(amt)
        else:
            break
    return median(tail)


def detect_recurring_series(
    events: list[FinancialEvent], home_amounts: dict[str, Optional[float]]
) -> list[RecurringSeries]:
    """Group settled/scheduled events by (category, direction) and find
    groups occurring at a roughly regular interval -- treated as recurring
    (salary, rent, subscriptions) and projected forward beyond their last
    known occurrence."""
    groups: dict[tuple[str, str], list[FinancialEvent]] = defaultdict(list)
    for e in events:
        if e.status not in (config.EVENT_STATUS_SETTLED, config.EVENT_STATUS_SCHEDULED):
            continue
        if e.direction == config.EVENT_DIRECTION_NON_CASH:
            continue
        if home_amounts.get(e.event_id) is None:
            continue  # can't use an unresolved amount for pattern detection
        groups[(e.category, e.direction)].append(e)

    series_list: list[RecurringSeries] = []
    for (category, direction), group_events in groups.items():
        if len(group_events) < 2:
            continue
        group_events.sort(key=lambda e: e.event_date)
        dates = [e.event_date for e in group_events]
        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        if not intervals:
            continue
        med_interval = median(intervals)
        if med_interval < 1:
            continue  # same-day duplicates, not a recurring cadence

        # longest trailing chain of intervals close to the median
        chain = [group_events[-1]]
        for i in range(len(intervals) - 1, -1, -1):
            if abs(intervals[i] - med_interval) <= INTERVAL_TOLERANCE_DAYS:
                chain.append(group_events[i])
            else:
                break
        chain.reverse()
        if len(chain) < 2:
            continue  # no genuine regular pattern in the trailing history

        chain_amounts = [home_amounts[e.event_id] for e in chain]
        amount = _trailing_consistent_amount(chain_amounts)
        if amount is None:
            continue

        series_list.append(
            RecurringSeries(
                category=category,
                direction=direction,
                interval_days=round(med_interval),
                amount=amount,
                last_date=chain[-1].event_date,
                currency=chain[-1].currency,
            )
        )

    return series_list


def project_recurring(
    series: RecurringSeries, window_start: date, window_end: date
) -> list[tuple[date, float, str]]:
    """Synthetic (date, home_amount, direction) occurrences for a recurring
    series, strictly after its last known date, up to window_end."""
    out = []
    next_date = series.last_date + timedelta(days=series.interval_days)
    while next_date <= window_end:
        if next_date >= window_start:
            out.append((next_date, series.amount, series.direction))
        next_date += timedelta(days=series.interval_days)
    return out


# ---------------------------------------------------------------------------
# Building the full set of cash-flow events for the forecast window
# ---------------------------------------------------------------------------

@dataclass
class CashFlowEvent:
    on_date: date
    signed_amount: float  # positive = credit to balance, negative = debit
    source: str           # event_id, "recurring:<category>", or "candidate_payment"


def _signed_home_amount(direction: str, amt: float) -> float:
    return amt if direction == config.EVENT_DIRECTION_CREDIT else -amt


def build_cash_flow_events(
    state: FinancialState,
    window_start: date,
    window_end: date,
) -> list[CashFlowEvent]:
    events_out: list[CashFlowEvent] = []
    covered_recurring_dates: set[tuple[str, str, date]] = set()

    for e in state.events:
        if e.status in config.EVENT_STATUSES_ALWAYS_EXCLUDE:
            continue
        if e.direction == config.EVENT_DIRECTION_NON_CASH:
            continue

        amt = home_amount(state, e.event_id)
        if amt is None:
            # Not yet resolved via image/message evidence -- excluded rather
            # than guessed. evidence_resolver.py should run before simulate()
            # so this ideally never triggers by the time we're evaluating.
            continue

        # Pending CREDITS excluded per problem_statement.md; pending DEBITS
        # still reserved (subtracted), matching "reserve pending transactions".
        if e.status == config.EVENT_STATUS_PENDING and e.direction == config.EVENT_DIRECTION_CREDIT:
            continue

        occurrence_date = e.settlement_date or e.event_date
        if not (window_start <= occurrence_date <= window_end):
            continue

        events_out.append(
            CashFlowEvent(
                on_date=occurrence_date,
                signed_amount=_signed_home_amount(e.direction, amt),
                source=e.event_id,
            )
        )
        covered_recurring_dates.add((e.category, e.direction, occurrence_date))

    recurring_series = detect_recurring_series(state.events, state.home_amounts)
    for series in recurring_series:
        for occ_date, amt, direction in project_recurring(series, window_start, window_end):
            key = (series.category, direction, occ_date)
            if key in covered_recurring_dates:
                continue  # already explicitly represented in financial_events.csv
            events_out.append(
                CashFlowEvent(
                    on_date=occ_date,
                    signed_amount=_signed_home_amount(direction, amt),
                    source=f"recurring:{series.category}",
                )
            )

    return events_out


# ---------------------------------------------------------------------------
# The simulation itself
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    is_safe: bool
    min_balance: float
    min_balance_date: Optional[date]
    final_balance: float
    balance_on_date: dict[date, float]


def simulate(
    state: FinancialState,
    candidate_payments: list[tuple[date, float]],
    forecast_days: int = config.FORECAST_DAYS,
) -> SimulationResult:
    """candidate_payments: list of (date, amount) DEBITS representing the
    purchase/plan being tested, e.g. [(request_date, amount_safe_to_pay)]
    or a full installment schedule. Amounts must already be in home_currency."""
    request_date = state.request.request_date
    window_end = request_date + timedelta(days=forecast_days)

    cash_flow = build_cash_flow_events(state, request_date, window_end)
    for pay_date, pay_amount in candidate_payments:
        cash_flow.append(
            CashFlowEvent(on_date=pay_date, signed_amount=-pay_amount, source="candidate_payment")
        )

    cash_flow.sort(key=lambda cf: cf.on_date)

    balance = state.profile.current_available_balance
    min_balance = balance
    min_balance_date: Optional[date] = request_date
    balance_on_date: dict[date, float] = {request_date: balance}

    for cf in cash_flow:
        balance += cf.signed_amount
        balance_on_date[cf.on_date] = balance
        if balance < min_balance:
            min_balance = balance
            min_balance_date = cf.on_date

    is_safe = min_balance >= state.profile.minimum_balance_to_keep

    return SimulationResult(
        is_safe=is_safe,
        min_balance=min_balance,
        min_balance_date=min_balance_date,
        final_balance=balance,
        balance_on_date=balance_on_date,
    )


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import data_loader
    from financial_state import build_financial_state

    ctx = data_loader.load_all()
    sample_request_id = next(iter(ctx.requests))
    state = build_financial_state(ctx, sample_request_id)

    result = simulate(state, candidate_payments=[])
    print(f"Baseline (no purchase) simulation for {sample_request_id}:")
    print(f"  starting balance:        {state.profile.current_available_balance}")
    print(f"  minimum_balance_to_keep: {state.profile.minimum_balance_to_keep}")
    print(f"  min_balance reached:     {result.min_balance} on {result.min_balance_date}")
    print(f"  final_balance:           {result.final_balance}")
    print(f"  is_safe:                 {result.is_safe}")

    recurring = detect_recurring_series(state.events, state.home_amounts)
    print(f"  recurring series detected: {len(recurring)}")
    for s in recurring:
        print(f"    {s.category}/{s.direction}: every {s.interval_days}d, amount={s.amount:.2f}, last={s.last_date}")