# NLP Risk Engine

The engine turns one cleaned `Document` into zero or more `RiskSignal`s. It is a pipeline of small, swappable
components, not one opaque model call, so every field can be explained and evaluated on its own.

```
Document.text
   |-- event classification  (per document)         -> event_type, confidence, top-3
   |-- entity linking        (per document)         -> [company | market]
   |-- sentiment             (per document x entity) -> score in [-1, 1], confidence
   '-- impact                (per signal)           -> 1..10 with per-feature points
                                                       -> confidence = geometric mean
```

## 1. Entity linking (`nlp/entities`)

Closed-world dictionary over ~40 S&P 100 names (`config/companies.yaml`): canonical names, aliases (Google -> Alphabet),
cashtags (`$TSLA`) and exchange tags (`(NASDAQ: MSFT)`). Matching is case-sensitive on word boundaries.
Names that are also everyday words (Apple, Meta, Visa, Intel, Oracle, Amazon) count only when the text has finance context
(shares, earnings, CEO, ...), which removes "Apple pie" and "Metadata". Up to three entities per document, ordered by first mention.
No entity but a recognisable event -> a `market`-scope signal; neither -> dropped.

## 2. Sentiment (`nlp/sentiment`)

- **FinBERT** (`ProsusAI/finbert`): `score = P(positive) - P(negative)` in [-1, 1], `confidence = max class probability`.
- **Lexicon baseline**: a small finance word list with negation handling, same score definition. It exists so the system runs
  with zero downloads and so FinBERT has a baseline to beat; it is deliberately weak.
- **Entity-aware input**: for each entity the model sees only the sentences that mention it (so "Apple surges. Intel plunges."
  yields opposite scores), falling back to the whole text.

## 3. Event classification (`nlp/events`)

Fixed taxonomy of 14 labels (`core/taxonomy.py`): Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch,
Regulatory, Earnings, Supply Chain, Leadership Change, Cybersecurity, Legal, Bankruptcy, Market Shock, Other.

| Backend | How | Cost | Notes |
|---|---|---|---|
| `rules` | regex cues, strong = 1.0, weak = 0.4, normalised to a distribution | instant, no deps | transparent; says "no evidence" (`matched=False`) instead of guessing |
| `embedding` | nearest prototype sentence in `bge-small-en-v1.5` space, softmax over classes | 33M params | editable without retraining (`event_prototypes.yaml`) |
| `hybrid` | rules fired: blend 0.6 rules + 0.4 embedding; no rule fired: embedding alone; no embedding model: rules | both | default for real runs |

Not implemented but supported by the protocol: NLI zero-shot (`deberta-v3-base-zeroshot-v2.0`) and an LLM with structured output.
They are comparison points for the evaluation rather than defaults (see `docs/research.md`).

## 4. Impact score (`nlp/impact`)

```
impact = 1 + 9 * ( 0.35 * severity + 0.25 * |sentiment| + 0.20 * exposure
                 + 0.10 * source_reliability + 0.10 * corroboration )         every feature in [0, 1]
```

| Feature | Definition |
|---|---|
| severity | prior per event type / 10 (Bankruptcy 10, Geopolitical 9, Market Shock 9, Credit/Regulatory/Macro 8, ..., Other 2) |
| |sentiment| | magnitude only; direction is deliberately ignored |
| exposure | company 1.0, market 0.8, sector 0.6 |
| source_reliability | per domain (Reuters/Bloomberg 1.0, ... social 0.4, unknown 0.5) |
| corroboration | `(distinct outlets - 1) / 4`, capped at 1 |

Every signal stores the per-feature contribution **in points**, so the score decomposes: the CLI example below is
`1 + 2.52 + 1.50 + 1.80 + 0.45 + 0.00 = 7.27`. All numbers live in `config/impact.yaml`.

**Status: priors, not calibrated.** The plan is to regress `|CAR|` (event-study abnormal return) on the five features, and to
compare against 200-300 human-scored headlines (`evaluation/README.md`). Until then, describe impact as a *severity prior*.

## 5. Confidence

`confidence = (sentiment_conf * event_conf * entity_conf)^(1/3)`: a geometric mean, so one shaky component drags the whole
signal down instead of being averaged away. The stress trigger requires `confidence >= 0.5`. Recent work on zero-shot financial
NLP found that being honest about when the model is unreliable was more useful than its point predictions
([arXiv 2606.12210](https://arxiv.org/abs/2606.12210)), which is why confidence is a first-class field.

## 6. Worked example (real output of `python -m prism.cli analyze`, lexicon + rules backends)

Input: *"Tesla shares fall after regulators open an investigation into its driver-assistance system"*

| Field | Value | Why |
|---|---|---|
| entity | Tesla (TSLA), scope company | dictionary match |
| event_type | Regulatory (p = 1.0) | cues: `regulators`, `investigation` |
| sentiment_score | -0.667 | two negative words (`fall`, `investigation`) |
| impact_score | 7.27 | 1 + severity 2.52 + sentiment 1.50 + exposure 1.80 + source 0.45 + corroboration 0.00 |
| confidence | 0.858 | geometric mean of 0.70 (sentiment), 0.95 (event), 0.95 (entity) |

With FinBERT in place of the lexicon the sentiment and confidence would change; the rest of the chain is identical.

## 7. Known limitations

- The lexicon baseline mislabels anything without a listed word as neutral; swap in FinBERT for real use.
- Single-label events; multi-event stories keep only the top class (top-3 is kept in the explanation).
- The entity dictionary covers ~40 companies; unknown companies produce market-scope signals.
- GDELT supplies headlines only, so GDELT-sourced signals have less context than NewsAPI-sourced ones.
- Impact is a prior and direction-agnostic; it is not a price forecast.
- No temporal evaluation yet; when calibrating on market data, tune and report on different time periods.
