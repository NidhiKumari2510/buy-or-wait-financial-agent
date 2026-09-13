"""
code/data_loader.py

Loads every dataset/*.csv into typed objects (models.py) and builds
indexes for O(1) retrieval by user_id / request_id / event_id.

Design principles:
  - Parse once, up front. Everything downstream (financial_state.py,
    simulator.py, planner.py) works off these indexes, not raw DataFrames.
  - Be defensive about blanks/NaN. pandas turns empty CSV cells into NaN
    even for "string" columns, and turns blank-but-otherwise-integer columns
    into float64. We normalize all of that here, once, so no other file has
    to think about it.
  - Never silently treat a blank `amount` as 0 -- that's an explicit rule in
    problem_statement.md. We keep it as None and let evidence_resolver.py
    decide what to do (usually: look up the linked image).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

import pandas as pd

import config
from models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageEvidence,
    Message,
    PaymentOption,
    Request,
)


# ---------------------------------------------------------------------------
# Small parsing helpers -- centralize NaN/blank handling here
# ---------------------------------------------------------------------------

def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _opt_str(value) -> Optional[str]:
    return None if _is_blank(value) else str(value).strip()


def _opt_float(value) -> Optional[float]:
    return None if _is_blank(value) else float(value)


def _opt_int(value) -> Optional[int]:
    if _is_blank(value):
        return None
    return int(float(value))  # handles "7.0" style values pandas sometimes produces


def _req_float(value, *, field_name: str, row_id: str) -> float:
    if _is_blank(value):
        raise ValueError(f"Required float field '{field_name}' is blank for row {row_id}")
    return float(value)


def _req_str(value, *, field_name: str, row_id: str) -> str:
    if _is_blank(value):
        raise ValueError(f"Required string field '{field_name}' is blank for row {row_id}")
    return str(value).strip()


def _parse_date(value, *, field_name: str, row_id: str) -> date:
    if _is_blank(value):
        raise ValueError(f"Required date field '{field_name}' is blank for row {row_id}")
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value).strip(), config.DATE_FORMAT).date()


def _opt_date(value) -> Optional[date]:
    if _is_blank(value):
        return None
    return datetime.strptime(str(value).strip(), config.DATE_FORMAT).date()


def _pipe_list(value) -> list[str]:
    """Parse a pipe-delimited CSV cell (e.g. 'full_payment|installments') into a list.
    Blank -> empty list, not [''] -- easy source of bugs otherwise."""
    if _is_blank(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _parse_bool(value) -> bool:
    """allows_partial_payment may come through as True/False, 'True'/'False',
    'TRUE'/'FALSE', 1/0, or 'yes'/'no' depending on how pandas/Excel touched it.
    Handle all of them rather than assuming one representation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in ("true", "yes", "1", "y")


# ---------------------------------------------------------------------------
# The DataContext -- everything loaded + indexed
# ---------------------------------------------------------------------------

class DataContext:
    def __init__(self):
        self.requests: dict[str, Request] = {}
        self.profiles: dict[str, FinancialProfile] = {}
        self.events_by_id: dict[str, FinancialEvent] = {}
        self.events_by_user: dict[str, list[FinancialEvent]] = defaultdict(list)
        self.payment_options_by_request: dict[str, list[PaymentOption]] = defaultdict(list)
        self.messages_by_user: dict[str, list[Message]] = defaultdict(list)
        self.messages_by_request: dict[str, list[Message]] = defaultdict(list)
        self.messages_by_event: dict[str, list[Message]] = defaultdict(list)
        self.images_by_user: dict[str, list[ImageEvidence]] = defaultdict(list)
        self.images_by_request: dict[str, list[ImageEvidence]] = defaultdict(list)
        self.images_by_event: dict[str, list[ImageEvidence]] = defaultdict(list)
        # exchange rate lookups: (from_currency, to_currency, rate_date) -> rate
        self.exchange_rates: dict[tuple[str, str, date], float] = {}
        # ordered list too, kept for the fallback lookup strategy in currency.py
        self._exchange_rates_list: list[ExchangeRate] = []

    # -- convenience accessors -------------------------------------------------

    def profile_for(self, user_id: str) -> FinancialProfile:
        return self.profiles[user_id]

    def events_for_user(self, user_id: str) -> list[FinancialEvent]:
        return self.events_by_user.get(user_id, [])

    def payment_options_for_request(self, request_id: str) -> list[PaymentOption]:
        return self.payment_options_by_request.get(request_id, [])

    def evidence_for_request(self, user_id: str, request_id: str):
        """Union of messages/images tied to this user OR this specific request.
        (A message/image can be user-level with no request_id, e.g. a general
        payroll notice not tied to any single 'Buy or Wait?' question.)"""
        messages = list(self.messages_by_user.get(user_id, []))
        messages += [m for m in self.messages_by_request.get(request_id, []) if m not in messages]
        images = list(self.images_by_user.get(user_id, []))
        images += [i for i in self.images_by_request.get(request_id, []) if i not in images]
        return messages, images


# ---------------------------------------------------------------------------
# Loaders (one per CSV file)
# ---------------------------------------------------------------------------

def _load_requests(ctx: DataContext, path=config.REQUESTS_CSV) -> None:
    df = pd.read_csv(path, dtype=str)  # dtype=str first: we control all numeric parsing ourselves
    for _, row in df.iterrows():
        rid = _req_str(row["request_id"], field_name="request_id", row_id=row.get("request_id", "?"))
        req = Request(
            request_id=rid,
            user_id=_req_str(row["user_id"], field_name="user_id", row_id=rid),
            request_date=_parse_date(row["request_date"], field_name="request_date", row_id=rid),
            request_type=_req_str(row["request_type"], field_name="request_type", row_id=rid),
            requested_amount=_req_float(row["requested_amount"], field_name="requested_amount", row_id=rid),
            desired_completion_date=_parse_date(
                row["desired_completion_date"], field_name="desired_completion_date", row_id=rid
            ),
            allows_partial_payment=_parse_bool(row["allows_partial_payment"]),
            request_text=_opt_str(row.get("request_text")) or "",
        )
        if req.request_type not in config.REQUEST_TYPES:
            print(f"[data_loader] WARNING: unexpected request_type '{req.request_type}' on {rid}")
        ctx.requests[rid] = req


def _load_profiles(ctx: DataContext, path=config.FINANCIAL_PROFILES_CSV) -> None:
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        uid = _req_str(row["user_id"], field_name="user_id", row_id=row.get("user_id", "?"))
        profile = FinancialProfile(
            user_id=uid,
            home_currency=_req_str(row["home_currency"], field_name="home_currency", row_id=uid),
            current_available_balance=_req_float(
                row["current_available_balance"], field_name="current_available_balance", row_id=uid
            ),
            minimum_balance_to_keep=_req_float(
                row["minimum_balance_to_keep"], field_name="minimum_balance_to_keep", row_id=uid
            ),
            financial_priorities=_pipe_list(row.get("financial_priorities")),
            expense_categories_to_protect=_pipe_list(row.get("expense_categories_to_protect")),
            expense_categories_user_is_willing_to_reduce=_pipe_list(
                row.get("expense_categories_user_is_willing_to_reduce")
            ),
            expense_categories_user_is_willing_to_stop=_pipe_list(
                row.get("expense_categories_user_is_willing_to_stop")
            ),
            payment_methods_user_will_consider=_pipe_list(row.get("payment_methods_user_will_consider")),
            max_installment_months=_opt_int(row.get("max_installment_months")),
        )
        if profile.home_currency not in config.SUPPORTED_CURRENCIES:
            print(f"[data_loader] WARNING: unexpected home_currency '{profile.home_currency}' for {uid}")
        ctx.profiles[uid] = profile


def _load_events(ctx: DataContext, path=config.FINANCIAL_EVENTS_CSV) -> None:
    df = pd.read_csv(path, dtype=str)
    seen_statuses = set()
    for _, row in df.iterrows():
        eid = _req_str(row["event_id"], field_name="event_id", row_id=row.get("event_id", "?"))
        status = _req_str(row["status"], field_name="status", row_id=eid)
        seen_statuses.add(status)
        event = FinancialEvent(
            event_id=eid,
            user_id=_req_str(row["user_id"], field_name="user_id", row_id=eid),
            event_type=_req_str(row["event_type"], field_name="event_type", row_id=eid),
            description=_opt_str(row.get("description")) or "",
            category=_req_str(row["category"], field_name="category", row_id=eid),
            direction=_req_str(row["direction"], field_name="direction", row_id=eid),
            amount=_opt_float(row.get("amount")),  # NB: may be None -- do NOT default to 0.0
            currency=_req_str(row["currency"], field_name="currency", row_id=eid),
            event_date=_parse_date(row["event_date"], field_name="event_date", row_id=eid),
            settlement_date=_opt_date(row.get("settlement_date")),
            status=status,
            linked_event_id=_opt_str(row.get("linked_event_id")),
            flexibility=_req_str(row["flexibility"], field_name="flexibility", row_id=eid),
            minimum_allowed_amount=_opt_float(row.get("minimum_allowed_amount")),
        )
        ctx.events_by_id[eid] = event
        ctx.events_by_user[event.user_id].append(event)

        known_statuses = {
        config.EVENT_STATUS_SETTLED,
        config.EVENT_STATUS_PENDING,
        config.EVENT_STATUS_SCHEDULED,
        config.EVENT_STATUS_CANCELLED,
        config.EVENT_STATUS_FAILED,
        config.EVENT_STATUS_UNREALIZED,
    }
    unexpected = seen_statuses - known_statuses
    if unexpected:
        print(f"[data_loader] WARNING: financial_events.csv has status values not yet "
              f"handled in config.py: {sorted(unexpected)} -- these need explicit "
              f"inclusion/exclusion rules before the simulator can trust them.")


def _load_payment_options(ctx: DataContext, path=config.REQUEST_PAYMENT_OPTIONS_CSV) -> None:
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        oid = _req_str(row["payment_option_id"], field_name="payment_option_id", row_id=row.get("payment_option_id", "?"))
        opt = PaymentOption(
            payment_option_id=oid,
            request_id=_req_str(row["request_id"], field_name="request_id", row_id=oid),
            payment_method=_req_str(row["payment_method"], field_name="payment_method", row_id=oid),
            payment_amount=_req_float(row["payment_amount"], field_name="payment_amount", row_id=oid),
            number_of_payments=int(_req_float(row["number_of_payments"], field_name="number_of_payments", row_id=oid)),
            first_payment_date=_parse_date(row["first_payment_date"], field_name="first_payment_date", row_id=oid),
            payment_frequency_days=_opt_int(row.get("payment_frequency_days")),
            financing_fee=_opt_float(row.get("financing_fee")) or 0.0,
            total_payable_amount=_req_float(
                row["total_payable_amount"], field_name="total_payable_amount", row_id=oid
            ),
        )
        ctx.payment_options_by_request[opt.request_id].append(opt)


def _load_messages(ctx: DataContext, path=config.MESSAGES_CSV) -> None:
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        mid = _req_str(row["message_id"], field_name="message_id", row_id=row.get("message_id", "?"))
        msg = Message(
            message_id=mid,
            user_id=_req_str(row["user_id"], field_name="user_id", row_id=mid),
            request_id=_opt_str(row.get("request_id")),
            related_event_id=_opt_str(row.get("related_event_id")),
            sent_at=_opt_str(row.get("sent_at")) or "",
            source_type=_req_str(row["source_type"], field_name="source_type", row_id=mid),
            message_text=_opt_str(row.get("message_text")) or "",
        )
        ctx.messages_by_user[msg.user_id].append(msg)
        if msg.request_id:
            ctx.messages_by_request[msg.request_id].append(msg)
        if msg.related_event_id:
            ctx.messages_by_event[msg.related_event_id].append(msg)


def _load_images(ctx: DataContext, path=config.IMAGES_CSV) -> None:
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        iid = _req_str(row["image_id"], field_name="image_id", row_id=row.get("image_id", "?"))
        img = ImageEvidence(
            image_id=iid,
            user_id=_req_str(row["user_id"], field_name="user_id", row_id=iid),
            request_id=_opt_str(row.get("request_id")),
            related_event_id=_opt_str(row.get("related_event_id")),
        )
        ctx.images_by_user[img.user_id].append(img)
        if img.request_id:
            ctx.images_by_request[img.request_id].append(img)
        if img.related_event_id:
            ctx.images_by_event[img.related_event_id].append(img)


def _load_exchange_rates(ctx: DataContext, path=config.EXCHANGE_RATES_CSV) -> None:
    df = pd.read_csv(path, dtype=str)
    for _, row in df.iterrows():
        rate = ExchangeRate(
            rate_date=_parse_date(row["rate_date"], field_name="rate_date", row_id="exchange_rates"),
            from_currency=_req_str(row["from_currency"], field_name="from_currency", row_id="exchange_rates"),
            to_currency=_req_str(row["to_currency"], field_name="to_currency", row_id="exchange_rates"),
            rate=_req_float(row["rate"], field_name="rate", row_id="exchange_rates"),
        )
        key = (rate.from_currency, rate.to_currency, rate.rate_date)
        ctx.exchange_rates[key] = rate.rate
        ctx._exchange_rates_list.append(rate)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_all() -> DataContext:
    ctx = DataContext()
    _load_profiles(ctx)
    _load_events(ctx)
    _load_payment_options(ctx)
    _load_messages(ctx)
    _load_images(ctx)
    _load_exchange_rates(ctx)
    _load_requests(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Debug helper -- run this file directly to sanity-check assumptions against
# your REAL dataset before building the simulator on top of guesses.
#   python3 code/data_loader.py
# ---------------------------------------------------------------------------

def _debug_summary(ctx: DataContext) -> None:
    print(f"requests:        {len(ctx.requests)}")
    print(f"profiles:        {len(ctx.profiles)}")
    print(f"events:          {len(ctx.events_by_id)}")
    print(f"payment options: {sum(len(v) for v in ctx.payment_options_by_request.values())}")
    print(f"messages:        {sum(len(v) for v in ctx.messages_by_user.values())}")
    print(f"images:          {sum(len(v) for v in ctx.images_by_user.values())}")
    print(f"exchange rates:  {len(ctx.exchange_rates)}")

    statuses = {e.status for e in ctx.events_by_id.values()}
    directions = {e.direction for e in ctx.events_by_id.values()}
    flexibilities = {e.flexibility for e in ctx.events_by_id.values()}
    currencies = {e.currency for e in ctx.events_by_id.values()}
    blank_amount_count = sum(1 for e in ctx.events_by_id.values() if e.amount is None)
    print(f"event statuses seen:      {sorted(statuses)}")
    print(f"event directions seen:    {sorted(directions)}")
    print(f"event flexibilities seen: {sorted(flexibilities)}")
    print(f"event currencies seen:    {sorted(currencies)}")
    print(f"events with blank amount (need image lookup): {blank_amount_count}")

    currency_pairs = {(f, t) for (f, t, _d) in ctx.exchange_rates}
    print(f"exchange rate currency pairs: {sorted(currency_pairs)}")


if __name__ == "__main__":
    context = load_all()
    _debug_summary(context)