"""
code/config.py

Central constants for the "Buy or Wait?" solution.
Values here are VERIFIED against the actual dataset via data_loader.py's
debug summary (python3 code/data_loader.py) -- not guessed from docs.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root; main.py should resolve from __file__)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_IMAGES_DIR = DATASET_DIR / "media" / "images"

REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
FINANCIAL_EVENTS_CSV = DATASET_DIR / "financial_events.csv"
REQUEST_PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"

OUTPUT_CSV = REPO_ROOT / "output.csv"  # final predictions at repo root, NOT dataset/

# ---------------------------------------------------------------------------
# Forecast parameters
# ---------------------------------------------------------------------------

FORECAST_DAYS = 90  # the "90-Day Safety Check" window from problem_statement.md

# ---------------------------------------------------------------------------
# Output schema (exact column order required by problem_statement.md)
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# ---------------------------------------------------------------------------
# Allowed values (from problem_statement.md "Allowed values" section)
# ---------------------------------------------------------------------------

AFFORDABILITY_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}

REQUEST_TYPES = {
    "purchase",
    "travel",
    "education",
    "family_transfer",
    "debt_repayment",
    "investment",
    "housing",
    "emergency_expense",
    "other",
}

MAX_SPENDING_CHANGES = 3
NONE_TOKEN = "none"

# ---------------------------------------------------------------------------
# financial_events.csv semantic values -- CONFIRMED against real dataset via
# `python3 code/data_loader.py` debug summary. Do not trust prior doc-based
# guesses over this.
# ---------------------------------------------------------------------------

# Actual status values seen: cancelled, failed, pending, scheduled, settled, unrealized
EVENT_STATUS_SETTLED = "settled"       # already occurred & cleared -- counts fully
EVENT_STATUS_PENDING = "pending"       # not yet cleared -- pending CREDITS excluded,
                                        # pending DEBITS still reserved (spec: "reserve
                                        # pending transactions" + "ignore pending credits")
EVENT_STATUS_SCHEDULED = "scheduled"   # confirmed future event (e.g. next salary) --
                                        # counts as a real future obligation/income
EVENT_STATUS_CANCELLED = "cancelled"   # exclude entirely
EVENT_STATUS_FAILED = "failed"         # exclude entirely
EVENT_STATUS_UNREALIZED = "unrealized" # non-cash investment gain/loss -- exclude entirely

EVENT_STATUSES_ALWAYS_EXCLUDE = {
    EVENT_STATUS_CANCELLED,
    EVENT_STATUS_FAILED,
    EVENT_STATUS_UNREALIZED,
}

# Actual direction values seen: credit, debit, non_cash
EVENT_DIRECTION_DEBIT = "debit"
EVENT_DIRECTION_CREDIT = "credit"
EVENT_DIRECTION_NON_CASH = "non_cash"  # investment valuations etc -- always excluded
                                        # from liquid cash-flow regardless of status

# Actual flexibility values seen: fixed, reducible, reducible_or_stoppable, stoppable
EVENT_FLEXIBILITY_FIXED = "fixed"
EVENT_FLEXIBILITY_STOPPABLE = "stoppable"
EVENT_FLEXIBILITY_REDUCIBLE = "reducible"
EVENT_FLEXIBILITY_REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"  # planner picks
                                        # whichever spending-change type applies

# ---------------------------------------------------------------------------
# Currencies -- confirmed: EUR, IDR, INR, USD, ZAR
# Confirmed exchange-rate pairs in dataset (hub structure -- verify this stays
# true as you re-run against the full file, this was from the debug summary):
#   EUR<->USD, EUR->ZAR, USD->EUR, USD->IDR, USD->INR
# i.e. USD and EUR are the only "hub" currencies with direct rates; any other
# pair (e.g. ZAR->IDR) must be converted via a two-hop route through USD/EUR.
# ---------------------------------------------------------------------------

SUPPORTED_CURRENCIES = {"INR", "ZAR", "IDR", "USD", "EUR"}
HUB_CURRENCIES = {"USD", "EUR"}

DATE_FORMAT = "%Y-%m-%d"