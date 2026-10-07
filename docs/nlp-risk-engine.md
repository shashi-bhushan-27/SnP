# NLP Risk Engine

The engine turns one cleaned `Document` into zero or more `RiskSignal`s. It is a pipeline of small, swappable
components, not one opaque model call, so every field can be explained and evaluated on its own.

```
Document (title + text)
   |-- relevance             commentary / listicle / advice headline?  -> Other
   |-- event classification  headline first, body as discounted fallback -> event_type, confidence, top-3
   |-- entity linking        headline first, body mentions capped        -> [company | market]
   |-- salience gate         drop if subject and event are both body-only (or market-level and body-only)
   |-- sentiment             per document x entity                       -> score in [-1, 1], confidence
   '-- impact                per signal                                  -> 1..10 with per-feature points
                                                                            -> confidence = geometric mean
```

## 0. Relevance and salience (added after the first live-data run)

On 50 live NewsAPI articles the first version produced 36 signals and one false stress trigger. The causes and
the rules that fixed them (same 50 articles afterwards: 15 signals, 0 false triggers):

| Problem | Rule |
|---|---|
| Event cues buried in the body ("The spice of the matter" whose body mentions inflation -> Macroeconomic 7.6) | Classify the **headline**; only if it has no evidence, classify the body at 0.7x confidence (`method ...+body`). A market-level signal with body-only evidence is dropped. |
| Subject buried in the body (a grocery chain closing stores -> Walmart, which the body mentions) | Link entities in the **headline**; body-only mentions are capped at 0.6 confidence; if both the entity and the event are body-only, drop. |
| Listicles, advice, questions ("3 Canadian AI Stocks...", "How to Protect Your Portfolio...", "...Time to Sell or Load Up?") | Commentary patterns on the headline (`config/relevance.yaml`) -> event Other; without a tracked entity the item is dropped. |
| A single unknown site could trigger a portfolio stress test | Scenario triggers require confirmation: >= 2 distinct outlets **or** source reliability >= 0.9 (`scenarios.yaml`). |

Each signal records where its evidence came from: `explanation.evidence = {"event": "title"|"body"|"commentary", "entity": "title"|"body"|"none"}`.

## 1. Entity linking (`nlp/entities`)

Two backends behind the same protocol: `dictionary` (default) and `spacy` (`ENTITY_BACKEND=spacy`), which runs the
dictionary first and adds spaCy ORG entities it does not know as ticker-less company mentions (confidence 0.6),
ignoring regulators, central banks, exchanges and news outlets (`non_company_orgs` in `companies.yaml`).


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

- FinBERT reads macro direction literally ("inflation hits a 40-year high, beating forecasts" -> +0.90). A rule layer
  (`config/macro_direction.yaml`, `MacroDirectionSentiment`) now sets the direction for hot/cooling inflation, hawkish
  central banks, jumping yields and weakening growth: on 81 historical events the sentiment sign agrees with the
  next-day S&P 500 move 82% of the time vs 77% for raw FinBERT (in-sample: the rules were written with these
  headlines in view). It covers only the patterns listed; anything else is still read literally.
- The lexicon baseline mislabels anything without a listed word as neutral; it is the baseline, FinBERT is the default.
- Single-label events; multi-event stories keep only the top class (top-3 is kept in the explanation).
- The entity dictionary covers ~40 companies; unknown companies produce market-scope signals.
- GDELT supplies headlines only, so GDELT-sourced signals have less context than NewsAPI-sourced ones.
- Impact is a prior and direction-agnostic; it is not a price forecast.
- No temporal evaluation yet; when calibrating on market data, tune and report on different time periods.
