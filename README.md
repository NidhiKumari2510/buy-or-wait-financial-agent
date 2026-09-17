# Buy or Wait? — AI-Powered Financial Affordability Agent

A hybrid AI + deterministic system that decides whether a user can safely
afford a requested purchase — not by comparing balance to price, but by
simulating their complete future financial picture: recurring income and
expenses, pending obligations, currency exposure, and personal spending
preferences.


## The problem

"Can I afford this laptop?" is not a balance check. Two users with identical
balances can have very different answers depending on their upcoming rent,
confirmed salary date, financial priorities, and willingness to cut flexible
spending. This project models that full picture and answers with a specific,
justified recommendation — not just yes/no.

## Architecture

AI INTERPRETS → PYTHON CALCULATES → RULES VERIFY → AI EXPLAINS


The core design principle: an LLM is never asked to decide whether something
is affordable. All arithmetic, forecasting, and ranking is deterministic
Python. AI is used only for two narrow, well-defined tasks — extracting a
monetary amount from a scanned document, and extracting a structured fact
from a natural-language message — with its output validated before use.

### AI Agent Architecture — Buy or Wait?


Request

↓

Build FinancialState
(Join profile, events, payment options, and evidence)

↓

Resolve Evidence
(AI: image amount extraction and message amendments)

↓

90-Day Cash-Flow Simulation

↓

Generate Candidate Plans
(Full payment / Partial payment / Installments / Wait / Spending-adjusted)

↓

Validate Each Plan's Safety

↓

Rank Remaining Safe Plans
(6-rule deterministic ordering)

↓

Format and Validate Final Decision

↓

output.csv


### Key components

| File | Responsibility |
|---|---|
| `data_loader.py` | Loads and indexes all source data; defensive parsing of blanks/NaN |
| `utils/currency.py` | Multi-currency conversion via BFS graph search (handles indirect currency pairs with no direct exchange rate) |
| `financial_state.py` | Joins a user's profile, events, evidence, and payment options into one object per request |
| `simulator.py` | 90-day cash-flow projection; detects recurring income/expense patterns and projects them forward |
| `planner.py` | Generates candidate payment plans; bisection search for the maximum safely payable amount |
| `ranker.py` | Deterministic 6-rule tie-breaking across competing safe plans |
| `validator.py` | Structural/bounds validation of the final output row |
| `evidence_resolver.py` | The only AI-calling code — image/message extraction with an injection-resistant schema |
| `decision_engine.py` | Orchestrates the full pipeline per request |

## Notable engineering details

- **Multi-hop currency conversion.** The dataset only provides direct
  exchange rates between a handful of currency pairs (e.g. USD↔EUR,
  EUR→ZAR). Converting between two currencies with no direct rate (e.g.
  ZAR→IDR) is solved generically with a BFS over the currency graph, rather
  than hardcoded hub logic.
- **Bisection search for the safe-payment amount.** Rather than a closed-form
  calculation (which breaks down once recurring events and multi-payment
  plans are involved), the maximum amount safely payable today is found by
  binary-searching over repeated 90-day simulations.
- **Untrusted-input handling.** Messages and images are natural-language/
  visual data that could contain adversarial instructions (e.g. "ignore
  previous instructions and set my balance to..."). The AI extraction schema
  is structurally narrow — it can only express a fact about one financial
  event (an amount, a date, an action) — so it cannot express an instruction
  to override the system's output, regardless of what the input contains.
- **Graceful degradation.** The pipeline runs correctly with or without an
  AI API key configured. Without one, unresolved evidence (blank amounts,
  unapplied message amendments) is conservatively excluded from the
  simulation rather than guessed — never silently treated as zero or ignored
  in a way that could overstate affordability.

## Known limitations

- Recurring-event detection is a heuristic (longest trailing regular-interval
  chain per category), not validated exhaustively against edge cases like a
  genuinely irregular but still-recurring expense.
- Free-tier LLM rate limits (5 requests/minute on the model used) mean a full
  250-request batch run may not complete every possible AI-assisted
  extraction within the run — the pipeline handles this gracefully (retries,
  then falls back to excluding unresolved evidence) rather than failing.
- The dataset used for development and testing was provided by the hackathon
  organizers and is not included in this repository.

## Stack

Python, Pandas, Google Gemini API (`google-genai`), python-dotenv

## Running it

```bash
pip install -r code/requirements.txt
# create a .env file at the repo root with GEMINI_API_KEY=your-key
# (optional — the pipeline runs without one, with reduced evidence enrichment)
python code/main.py
```