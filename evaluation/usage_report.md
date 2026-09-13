# Usage Report

## Model

- Provider: Anthropic
- Model: claude-haiku-4-5-20251001
- Used for: extracting amounts from linked images (blank financial_events),
  and structured amendments (confirm/amend/cancel/delay) from user messages.
  All arithmetic, currency conversion, 90-day simulation, plan generation,
  ranking, and validation are deterministic Python -- the model is never
  asked to decide affordability directly.

## Calls (final full-dataset run that produced output.csv)

- Total model calls: 0
- Total input tokens: 0
- Total output tokens: 0
- Total tokens: 0
- Average tokens per request: 0.0

## Estimated Cost

- Input cost: $0.0000
- Output cost: $0.0000
- Total estimated cost: $0.0000
- Average cost per request: $0.000000

Note: no ANTHROPIC_API_KEY was configured for this run, so all LLM-assisted steps (image amount resolution, message amendment extraction) were skipped. The pipeline degrades gracefully in this case: unresolved blank-amount events are excluded from the 90-day forecast rather than guessed, and messages are left unapplied, per the rule against inventing unsupported financial information.
