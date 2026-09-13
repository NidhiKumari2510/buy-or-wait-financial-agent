"""
code/financial_state.py

Builds one FinancialState per request_id: joins the user's profile with
their events, this request's payment options, and any relevant
messages/images -- then normalizes every event's amount into the user's
home_currency.

Events with a blank amount (need image resolution) are left as None in
home_amounts. evidence_resolver.py fills those in later, before the 90-day
simulator runs -- this file does not decide inclusion/exclusion by status;
that's simulator.py's concern.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from data_loader import DataContext
from models import FinancialEvent, FinancialState
from utils.currency import convert


def _rate_lookup_date(event: FinancialEvent) -> date:
    """Prefer settlement_date (when money actually moves); some pending
    events have a blank settlement_date, so fall back to event_date."""
    return event.settlement_date or event.event_date


def _to_home_currency(ctx: DataContext, event: FinancialEvent, home_currency: str) -> Optional[float]:
    if event.amount is None:
        return None  # still needs image resolution -- never default to 0
    if event.currency == home_currency:
        return event.amount
    return convert(ctx, event.amount, event.currency, home_currency, _rate_lookup_date(event))


def build_financial_state(ctx: DataContext, request_id: str) -> FinancialState:
    request = ctx.requests[request_id]
    profile = ctx.profile_for(request.user_id)
    events = ctx.events_for_user(request.user_id)
    payment_options = ctx.payment_options_for_request(request_id)
    messages, images = ctx.evidence_for_request(request.user_id, request_id)

    state = FinancialState(
        request=request,
        profile=profile,
        events=events,
        messages=messages,
        images=images,
        payment_options=payment_options,
    )

    for event in events:
        state.home_amounts[event.event_id] = _to_home_currency(ctx, event, profile.home_currency)

    return state


def set_resolved_amount(ctx: DataContext, state: FinancialState, event: FinancialEvent) -> None:
    """Call after evidence_resolver.py sets event.amount from an image/message.
    Converts to home currency and updates state.home_amounts in place."""
    if event.amount is None:
        raise ValueError(f"Cannot resolve {event.event_id}: amount is still None")
    state.home_amounts[event.event_id] = _to_home_currency(ctx, event, state.profile.home_currency)


def home_amount(state: FinancialState, event_id: str) -> Optional[float]:
    """None means the event's amount hasn't been resolved yet."""
    return state.home_amounts.get(event_id)


# ---------------------------------------------------------------------------
# Debug helper -- python3 code/financial_state.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import data_loader

    ctx = data_loader.load_all()
    sample_request_id = next(iter(ctx.requests))
    state = build_financial_state(ctx, sample_request_id)

    print(f"Built FinancialState for {sample_request_id}")
    print(f"  user_id:         {state.request.user_id}")
    print(f"  home_currency:   {state.profile.home_currency}")
    print(f"  events:          {len(state.events)}")
    print(f"  payment_options: {len(state.payment_options)}")
    print(f"  messages:        {len(state.messages)}")
    print(f"  images:          {len(state.images)}")
    unresolved = [eid for eid, amt in state.home_amounts.items() if amt is None]
    print(f"  events still needing image-resolved amount: {len(unresolved)}")