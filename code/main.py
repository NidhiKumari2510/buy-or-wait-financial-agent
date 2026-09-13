"""
code/main.py

Entry point: python3 code/main.py [--limit N]

Runs the full pipeline over dataset/requests.csv, writes output.csv to the
repo root, and writes evaluation/usage_report.md summarizing LLM usage
(zero calls / zero cost is a valid, honest report if no API key is set).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import config` etc.

import config
import data_loader
from decision_engine import decide
from evidence_resolver import UsageTotals

# Haiku 4.5 pricing (USD per million tokens) -- update if Anthropic's pricing
# page shows different numbers at submission time; this is only used for the
# cost ESTIMATE in usage_report.md, never for anything decision-affecting.
INPUT_COST_PER_MTOK = 1.00
OUTPUT_COST_PER_MTOK = 5.00


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N requests (testing)")
    args = parser.parse_args()

    start = time.time()
    ctx = data_loader.load_all()
    usage = UsageTotals()

    request_ids = list(ctx.requests.keys())
    if args.limit:
        request_ids = request_ids[: args.limit]

    rows = []
    for i, request_id in enumerate(request_ids, 1):
        try:
            row = decide(ctx, request_id, usage)
        except Exception as exc:
            print(f"[main] ERROR on {request_id}: {exc!r} -- writing safe fallback row")
            req = ctx.requests[request_id]
            row = {
                "request_id": request_id,
                "amount_safe_to_pay": "0",
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": config.NONE_TOKEN,
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": config.NONE_TOKEN,
                "decision_explanation": f"Could not evaluate safely due to an internal error: {exc}",
            }
        rows.append(row)
        if i % 25 == 0 or i == len(request_ids):
            print(f"[main] processed {i}/{len(request_ids)}")

    config.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    elapsed = time.time() - start
    print(f"[main] wrote {len(rows)} rows to {config.OUTPUT_CSV} in {elapsed:.1f}s")

    _write_usage_report(usage)


def _write_usage_report(usage: UsageTotals) -> None:
    report_path = Path(__file__).resolve().parent.parent / "evaluation" / "usage_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = usage.input_tokens + usage.output_tokens
    avg_tokens = total_tokens / usage.calls if usage.calls else 0
    input_cost = usage.input_tokens / 1_000_000 * INPUT_COST_PER_MTOK
    output_cost = usage.output_tokens / 1_000_000 * OUTPUT_COST_PER_MTOK
    total_cost = input_cost + output_cost
    avg_cost = total_cost / usage.calls if usage.calls else 0

    content = f"""# Usage Report

## Model

- Provider: Anthropic
- Model: claude-haiku-4-5-20251001
- Used for: extracting amounts from linked images (blank financial_events),
  and structured amendments (confirm/amend/cancel/delay) from user messages.
  All arithmetic, currency conversion, 90-day simulation, plan generation,
  ranking, and validation are deterministic Python -- the model is never
  asked to decide affordability directly.

## Calls (final full-dataset run that produced output.csv)

- Total model calls: {usage.calls}
- Total input tokens: {usage.input_tokens}
- Total output tokens: {usage.output_tokens}
- Total tokens: {total_tokens}
- Average tokens per request: {avg_tokens:.1f}

## Estimated Cost

- Input cost: ${input_cost:.4f}
- Output cost: ${output_cost:.4f}
- Total estimated cost: ${total_cost:.4f}
- Average cost per request: ${avg_cost:.6f}

{"Note: no ANTHROPIC_API_KEY was configured for this run, so all LLM-assisted " \
 "steps (image amount resolution, message amendment extraction) were skipped. " \
 "The pipeline degrades gracefully in this case: unresolved blank-amount events " \
 "are excluded from the 90-day forecast rather than guessed, and messages are " \
 "left unapplied, per the rule against inventing unsupported financial information." \
 if usage.calls == 0 else ""}
"""
    report_path.write_text(content, encoding="utf-8")
    print(f"[main] wrote usage report to {report_path}")


if __name__ == "__main__":
    main()