"""
code/utils/currency.py

Currency conversion using dataset/exchange_rates.csv.

Confirmed from your actual data (data_loader.py debug output) that only a
handful of direct pairs exist: EUR->USD, EUR->ZAR, USD->EUR, USD->IDR,
USD->INR. Rates are NOT symmetric in the file (e.g. there's no ZAR->EUR row,
only EUR->ZAR), and some pairs (e.g. IDR->ZAR) require multiple hops.

Strategy, per hop:
  1. Exact (from, to, date) match.
  2. Nearest PRECEDING date for that exact (from, to) pair (logged as a
     warning so you can see if this path is ever actually used).
  3. Inverse of the reverse pair (1 / rate(to, from, date)), same date
     rules as above.
Multi-hop routing is solved generically with a BFS over the small currency
graph, so it doesn't matter how many hops a given pair needs -- no hardcoded
hub logic to get subtly wrong.
"""

from __future__ import annotations

from collections import deque
from datetime import date

import config
from data_loader import DataContext


class ExchangeRateNotFound(Exception):
    pass


def _direct_rate(ctx: DataContext, from_ccy: str, to_ccy: str, on_date: date):
    return ctx.exchange_rates.get((from_ccy, to_ccy, on_date))


def _nearest_preceding_rate(ctx: DataContext, from_ccy: str, to_ccy: str, on_date: date):
    candidates = [
        r for r in ctx._exchange_rates_list
        if r.from_currency == from_ccy and r.to_currency == to_ccy and r.rate_date <= on_date
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r.rate_date)
    if best.rate_date != on_date:
        print(f"[currency] WARNING: no exact rate for {from_ccy}->{to_ccy} on {on_date}; "
              f"using nearest preceding rate from {best.rate_date} instead.")
    return best.rate


def _single_hop_rate(ctx: DataContext, from_ccy: str, to_ccy: str, on_date: date):
    """One direct conversion step, trying exact match, then nearest-preceding,
    then the inverse of the reverse pair (both exact and nearest-preceding)."""
    rate = _direct_rate(ctx, from_ccy, to_ccy, on_date)
    if rate is not None:
        return rate
    rate = _nearest_preceding_rate(ctx, from_ccy, to_ccy, on_date)
    if rate is not None:
        return rate
    inv = _direct_rate(ctx, to_ccy, from_ccy, on_date)
    if inv is not None:
        return 1.0 / inv
    inv = _nearest_preceding_rate(ctx, to_ccy, from_ccy, on_date)
    if inv is not None:
        return 1.0 / inv
    return None


def get_rate(ctx: DataContext, from_ccy: str, to_ccy: str, on_date: date) -> float:
    """Return the multiplier such that amount_in_to_ccy = amount_in_from_ccy * rate.
    Finds a path through the currency graph if a direct/inverse pair isn't
    available (e.g. IDR -> ZAR needs to route through USD and/or EUR)."""
    if from_ccy == to_ccy:
        return 1.0

    visited = {from_ccy}
    queue = deque([(from_ccy, 1.0)])
    while queue:
        current, acc_rate = queue.popleft()
        for candidate in config.SUPPORTED_CURRENCIES:
            if candidate in visited:
                continue
            hop = _single_hop_rate(ctx, current, candidate, on_date)
            if hop is None:
                continue
            new_rate = acc_rate * hop
            if candidate == to_ccy:
                return new_rate
            visited.add(candidate)
            queue.append((candidate, new_rate))

    raise ExchangeRateNotFound(
        f"No conversion route found from {from_ccy} to {to_ccy} on/before {on_date}"
    )


def convert(ctx: DataContext, amount: float, from_ccy: str, to_ccy: str, on_date: date) -> float:
    return amount * get_rate(ctx, from_ccy, to_ccy, on_date)