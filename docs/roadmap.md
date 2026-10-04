# Roadmap to 11 October

Deliverables (from the brief): public repository, live demo of at most 5 minutes, at most 7 slides.
Order of work follows the dependency chain: engine -> evaluation -> module -> dashboard -> polish. A dashboard on
top of unvalidated logic is worth less than a plain one on a defensible engine.

## Done

- [x] Research on sources, models and rate limits (`docs/research.md`)
- [x] Modular scaffold: contracts, protocols, composition root, boundary test
- [x] ETL: GDELT / NewsAPI / replay adapters, normalize, dedupe, idempotent load, run reports, scheduler
- [x] Risk Engine baseline: dictionary linker, lexicon + FinBERT adapter, rules + embedding + hybrid events, explainable impact
- [x] Module B baseline: sensitivity-based stress engine, scenario matrix, auto-trigger, API
- [x] Tests (unit, pipeline, API, architecture) and CI

## Plan

| Day | Focus | Outcome |
|---|---|---|
| **Oct 5** | Run the real models | FinBERT + `bge-small` run on sample text on the dev machine; fetch HF tweet data; verify GDELT live from a clean network and re-record the fixture; get a NewsAPI key |
| **Oct 6** | Engine accuracy | switch defaults to `finbert` + `hybrid`; tune `event_rules.yaml` / `event_prototypes.yaml` on error cases; start the human-labelled set (200+ headlines) |
| **Oct 7** | Evaluation harness | sentiment / event / impact metrics with baselines; event-study calibration of impact weights with yfinance; confidence calibration check |
| **Oct 8** | Module B depth | derive exposures from the transaction data (document it as seeding a synthetic book); review shock numbers; asset-level contribution views |
| **Oct 9** | Dashboard + live mode | one page: KPIs, risk feed, stress panel, charts; enable GDELT + NewsAPI polling with replay as fallback |
| **Oct 10** | Freeze and rehearse | no new features; write the 7 slides from measured results; rehearse the 5-minute demo twice; record a fallback video |
| **Oct 11** | Buffer | final test run, README check from a clean clone, tag the release, submit |

## Decisions needed from the team

1. Module B only (recommended), or also Module A?
2. Dashboard: Streamlit (fast) or React (polished)? See `frontend/README.md`.
3. Scenario shock numbers in `config/scenarios.yaml`, especially the credit-spread leg (the brief said "+5%" without a unit).
4. Hosted demo or local? A free NewsAPI key may not be used on a hosted service; GDELT may block shared cloud IPs.

## Risks

| Risk | Mitigation |
|---|---|
| GDELT blocks the demo network (429 seen already) | replay mode is the primary demo path; live sources are a bonus; `SourceUnavailable` never stops a run |
| NewsAPI is 24 h delayed and capped at 100 requests/day | present it as a delayed source; budget guard; never the star of the demo |
| Model download on the demo machine | download and cache FinBERT/bge before the demo; the lexicon + rules fallback still runs |
| Numbers on slides without a source | only report outputs of `evaluation/`, with baselines and sample sizes |
| Scope creep | Module A, semantic de-duplication, React and cloud deployment are explicitly optional |

## Demo script (5 minutes)

1. 0:00 Dashboard idle, one sentence on the problem.
2. 0:30 Start replay: articles stream in, signals appear with sentiment, event, impact and confidence.
3. 1:15 Open one signal: show the impact breakdown and the explanation.
4. 2:15 A high-impact negative event arrives: the stress test triggers automatically.
5. 3:30 Portfolio before / after, loss by asset class, the swap hedge offsetting part of the loss.
6. 4:15 Architecture and measured evaluation numbers; one closing sentence.
