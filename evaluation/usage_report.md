# Usage Report

## Model

- Provider: Google Gemini
- Model: gemini-3.6-flash
- Used for: extracting amounts from linked images (blank financial_events),
  and structured amendments (confirm/amend/cancel/delay) from user messages.
  All arithmetic, currency conversion, 90-day simulation, plan generation,
  ranking, and validation are deterministic Python -- the model is never
  asked to decide affordability directly.

## Calls (final full-dataset run that produced output.csv)

- Total model calls: 9
- Total input tokens: 8383
- Total output tokens: 447
- Total tokens: 8830
- Average tokens per request: 981.1

## Estimated Cost

- Input cost: $0.0025
- Output cost: $0.0011
- Total estimated cost: $0.0036
- Average cost per request: $0.000404


