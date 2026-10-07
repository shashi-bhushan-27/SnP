# The novel idea: history-calibrated, analog-explained stress scenarios

## In one sentence

When the NLP engine detects a high-impact event, PRISM does not apply a made-up shock. It applies **what
markets actually did after past events of the same kind**, at a chosen severity, and it shows the most similar
past events as the explanation ("this looks like SVB 2023 and Washington Mutual 2008").

## The problem it fixes

Stress scenarios are usually typed in by hand, like the brief's own example ("a 10% drop in equities, a 2%
increase in interest rates"). We tested hand-written shocks against 18 years of history and found two problems:

1. **The rates leg has the wrong sign.** In a crisis, money runs to government bonds and yields *fall*.
   In our library of 81 market-moving events (2008-2025), the 10-year Treasury yield fell after **100% of
   bankruptcies**, **81% of geopolitical shocks** and **73% of credit events**. The hand-written matrix assumes
   it rises by 50-200 bp. For a bank holding Treasuries that is not a detail: in the Lehman week our bond book
   *gained* $0.31M; the matrix books a $2.26M bond loss.
2. **The size is far beyond reality.** For our $60M synthetic book the worst real outcome among all 81 events
   (March 2020, oil war plus Covid, S&P 500 -16.5%) costs **-$2.72M**. Every hand-written scenario costs more
   (-$3.05M to -$5.97M), including "regulatory action".

## How it works

```
headline + event type (from the NLP engine)
        |
        +--> pool: past events of the same type (>= 5 of them, else all events)          backend/config/analog_library.json
        |        each with its real market reaction: S&P 500, 10-year yield, Baa spread      (Yahoo Finance + FRED)
        |
        +--> price every past reaction on TODAY's book
        |        stress   = the reaction at the chosen percentile (default 10th = "1-in-10 bad outcome")
        |                   -> a real historical move, so equities, rates and credit stay consistent
        |        expected = the pool's average reaction
        |
        +--> explanation: the 5 past events most similar in meaning (sentence embeddings)
                 with what markets did and what each would cost today
```

* Library: 81 events, curated with dates and neutral, outcome-free titles (`backend/config/analog_events.yaml`).
  Reactions are computed, never typed: `scripts/build_analog_library.py` pulls S&P 500 closes (Yahoo Finance),
  the 10-year yield (FRED `DGS10`) and the Baa-Treasury spread (FRED `BAA10Y`), and measures the move from the
  previous close to the S&P 500's lowest close within 5 sessions, with every factor read on that same day.
* Code: `backend/prism/modules/stress/analogs.py` (library + retrieval), `calibrated.py` (scenario builder);
  the trigger (`trigger.py`) uses it automatically; `SCENARIO_SOURCE=matrix` switches back.
* API: `POST /api/stress-test` (`source`, `quantile`, `headline`), `GET /api/analogs`, `GET /api/history/summary`.
* Dashboard: basis event, expected vs stress P&L, every past event priced on today's book, nearest analogs,
  side-by-side with the hand-written matrix, severity selector (1-in-5 / 10 / 20).

## Evidence (`evaluation/results/analog_backtest.md`)

Each event is held out and predicted from the others. Neighbours within 10 days are purged (they share the same
market window); a stricter time-respecting run uses only events that happened before the held-out one.

**Severity calibration** (trough horizon; coverage = share of events whose real loss was no worse than the
scenario's; excess = how much harsher than reality the scenario was, on average):

| Scenario rule | Leave-one-out coverage | Excess | Time-respecting coverage | Excess |
|---|---|---|---|---|
| History, 1-in-5 (q20, same type) | 77% | $0.26M | 77% | $0.23M |
| History, 1-in-10 (q10, same type) | 85% | $0.44M | 89% | $0.43M |
| Hand-written matrix | 95% | $4.72M | 94% | $4.69M |

The matrix buys roughly 6-10 more points of coverage with about **ten times** the over-reserving, and it still
misses about 1 event in 20 because its rates direction is wrong.

**Point prediction** (what will the reaction be?), trough horizon, leave-one-out:

| Method | P&L error | 10y direction correct |
|---|---|---|
| Hand-written matrix | $4.75M | 31% |
| Average of same event type | $0.38M | 63% |
| Analog retrieval (sentence embeddings, top 5) | $0.43M | 51% |
| Analog-weighted history | $0.38M | 65% |

## What did not work (and why we say so)

Retrieving the nearest past events by meaning does **not** predict the next reaction better than the average of
the same event type, at this sample size. The reason is instructive: embeddings match *topic*, not *direction*.
"US inflation runs hotter than expected" (yields up) and "US inflation cools" (yields down) sit next to each
other in embedding space. Filtering by FinBERT sentiment did not separate them either, because FinBERT scores
"inflation hits a 40-year high" as **positive** (+0.90). So analogs are used for what they are good at:
explanation and context. The severity comes from the event type's history.

## Limits

* 81 events is a small sample; the library is hand-curated and biased toward famous events.
* Results are for this synthetic book (Treasuries and a pay-fixed swap give it a flight-to-quality hedge);
  a pure equity book would look different. The method carries over; the numbers do not.
* History cannot contain truly new crises. Hypothetical scenarios still have a place, but their rates leg
  should follow the evidence, and their severity should be compared with what history says.
* Three factors (equities, 10-year yield, Baa spread) are a simplification of a real bank risk model.
