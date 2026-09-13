"""
code/models.py

Typed dataclasses mirroring the actual dataset/*.csv schemas.
Field names deliberately match the CSV column names 1:1 so that loading code
in data_loader.py is a thin, obvious mapping -- no renaming games.

Source of truth for these fields: problem_statement.md + the actual CSV
headers in dataset/. If you add a field here that isn't in the real CSV,
you've violated the "do not invent dataset fields" rule -- verify against
your own dataset/*.csv headers before trusting any field listed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# requests.csv
# ---------------------------------------------------------------------------

@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str          # one of config.REQUEST_TYPES
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


# ---------------------------------------------------------------------------
# financial_profiles.csv
# ---------------------------------------------------------------------------

@dataclass
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: list[str]                    # pipe-delimited in CSV
    expense_categories_to_protect: list[str]            # pipe-delimited in CSV
    expense_categories_user_is_willing_to_reduce: list[str]
    expense_categories_user_is_willing_to_stop: list[str]
    payment_methods_user_will_consider: list[str]       # e.g. full_payment|installments
    max_installment_months: Optional[int]               # None if blank


# ---------------------------------------------------------------------------
# financial_events.csv
#
# NOTE ON linked_event_id vs related_event_id:
#   - financial_events.csv uses `linked_event_id` to point to an earlier
#     event in the SAME transaction/investment/recurring lifecycle.
#   - messages.csv / images.csv use `related_event_id` to point AT a
#     financial event they amend/clarify/provide evidence for.
#   These are two different columns on two different files -- don't conflate
#   them in code.
# ---------------------------------------------------------------------------

@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str             # e.g. expense, debt_payment, subscription, income
    description: str
    category: str                # e.g. rent, utilities, salary, groceries, ...
    direction: str                # "debit" or "credit"
    amount: Optional[float]       # None/blank means "look this up via images.csv" -- NEVER treat as 0
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str                   # settled | pending | scheduled | cancelled | failed | unrealized
    linked_event_id: Optional[str]
    flexibility: str              # fixed | stoppable | reducible
    minimum_allowed_amount: Optional[float]  # floor for reducible events


# ---------------------------------------------------------------------------
# request_payment_options.csv
# ---------------------------------------------------------------------------

@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str            # e.g. full_payment, installments
    payment_amount: float           # amount per tranche
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]  # blank for full_payment
    financing_fee: float
    total_payable_amount: float

    def schedule(self) -> list[tuple[date, float]]:
        """
        Expand this option into a chronological list of (payment_date, amount)
        tuples. Used both for 90-day simulation and for formatting payment_plan.
        """
        from datetime import timedelta

        freq = self.payment_frequency_days or 0
        out = []
        for i in range(self.number_of_payments):
            pay_date = self.first_payment_date + timedelta(days=freq * i)
            out.append((pay_date, self.payment_amount))
        return out


# ---------------------------------------------------------------------------
# messages.csv
# ---------------------------------------------------------------------------

@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str                  # ISO 8601 timestamp; parse lazily, not every row needs it
    source_type: str              # employer | service_provider | bank | merchant | financial_service
    message_text: str             # UNTRUSTED. Never let this control control-flow directly.


# ---------------------------------------------------------------------------
# images.csv
# ---------------------------------------------------------------------------

@dataclass
class ImageEvidence:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]

    def path(self, media_dir) -> "Path":
        from pathlib import Path
        return Path(media_dir) / f"{self.image_id}.png"


# ---------------------------------------------------------------------------
# exchange_rates.csv
# ---------------------------------------------------------------------------

@dataclass
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float


# ---------------------------------------------------------------------------
# Derived / working structures (NOT direct CSV mirrors)
# ---------------------------------------------------------------------------

@dataclass
class FinancialState:
    """
    Everything the decision engine needs for one request, pre-joined and
    pre-normalized to the user's home_currency. Built by financial_state.py
    out of the raw dataclasses above -- this is the single object that
    simulator.py, planner.py, ranker.py, and validator.py all consume.
    """
    request: Request
    profile: FinancialProfile
    events: list[FinancialEvent] = field(default_factory=list)     # this user's events, home-currency normalized
    messages: list[Message] = field(default_factory=list)           # this user's/request's messages
    images: list[ImageEvidence] = field(default_factory=list)       # this user's/request's images
    payment_options: list[PaymentOption] = field(default_factory=list)  # options for this request_id
    home_amounts: dict[str, Optional[float]] = field(default_factory=dict)
    # event_id -> amount converted to profile.home_currency.
    # None means "amount was blank in the CSV, still needs image resolution."