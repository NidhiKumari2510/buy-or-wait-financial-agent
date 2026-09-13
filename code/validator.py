"""
code/validator.py

Deterministic post-hoc checks on a finished output row, run before writing
to output.csv. Never blocks the row from being written (every request_id
must get a row) -- logs violations loudly instead, so they surface during
development rather than silently shipping a malformed submission.
"""

from __future__ import annotations

import re

import config
from models import FinancialState
from planner import Candidate

_PLAN_ENTRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:-?\d+(\.\d+)?$")


def validate_row(row: dict, state: FinancialState, candidate: Candidate | None) -> list[str]:
    problems = []
    requested = state.request.requested_amount

    amt = row["amount_safe_to_pay"]
    if not (0 <= amt <= requested + 1e-6):
        problems.append(f"amount_safe_to_pay={amt} out of bounds [0, {requested}]")

    if row["affordability_status"] not in config.AFFORDABILITY_STATUSES:
        problems.append(f"invalid affordability_status: {row['affordability_status']}")

    if row["recommended_payment_method"] not in config.PAYMENT_METHODS:
        problems.append(f"invalid recommended_payment_method: {row['recommended_payment_method']}")

    if row["affordability_status"] == "affordable_now":
        if row["earliest_date_for_full_payment"] != state.request.request_date.isoformat():
            problems.append("affordable_now requires earliest_date_for_full_payment == request_date")

    plan = row["payment_plan"]
    if plan != config.NONE_TOKEN:
        entries = plan.split("|")
        for entry in entries:
            if not _PLAN_ENTRY_RE.match(entry):
                problems.append(f"malformed payment_plan entry: {entry!r}")
        dates = [e.split(":")[0] for e in entries]
        if dates != sorted(dates):
            problems.append("payment_plan entries not in chronological order")

        if row["recommended_payment_method"] == "partial_payment":
            if len(entries) != 2:
                problems.append("partial_payment must have exactly 2 payments")
            else:
                amounts = [float(e.split(":")[1]) for e in entries]
                total = round(sum(amounts), 2)
                if abs(total - requested) > 0.01:
                    problems.append(f"partial_payment total {total} != requested_amount {requested}")

        if row["recommended_payment_method"] == "installments":
            if candidate is None or candidate.payment_option_id is None:
                problems.append("installments recommended but no matching payment_option_id recorded")
            else:
                matching = [o for o in state.payment_options if o.payment_option_id == candidate.payment_option_id]
                if not matching:
                    problems.append(f"payment_option_id {candidate.payment_option_id} not found in supplied options")

    changes = row["spending_changes_needed"]
    if changes != config.NONE_TOKEN:
        change_entries = changes.split("|")
        if len(change_entries) > config.MAX_SPENDING_CHANGES:
            problems.append(f"more than {config.MAX_SPENDING_CHANGES} spending changes")
        stop_ids = set()
        reduce_ids = set()
        for entry in change_entries:
            if entry.startswith("stop:"):
                stop_ids.add(entry.split(":", 1)[1])
            elif entry.startswith("reduce_to:"):
                reduce_ids.add(entry.split(":")[1])
            else:
                problems.append(f"malformed spending_changes_needed entry: {entry!r}")
        overlap = stop_ids & reduce_ids
        if overlap:
            problems.append(f"same event both stopped and reduced: {overlap}")

    return problems