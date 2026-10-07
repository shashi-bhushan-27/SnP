# PRISM: AI Financial Risk Intelligence and Portfolio Stress Engine

Turns unstructured financial text (news, social posts) into structured risk signals, and turns high-impact
signals into **history-calibrated** stress tests of a synthetic wholesale-banking portfolio.

```
NewsAPI / GDELT / replay -> ETL -> NLP Risk Engine -> risk signals -> stress test (Module B) -> dashboard
                                   entity, event type,   (API + JSONL)   shocks from what markets
                                   sentiment, impact,                    actually did after similar
                                   confidence                            past events
```

## What is new here

Stress scenarios are normally typed in by hand ("equities -10%, rates +2%"). We checked that against 81 real
market-moving events (2008-2025): the 10-year Treasury yield **fell** after 100% of bankruptcies and 81% of
geopolitical shocks (flight to safety), and for our book every hand-written scenario is harsher than the worst
real outcome in 18 years (March 2020, -$2.72M). PRISM instead shocks the book with **what markets actually did
after past events of the same kind** (S&P 500, 10-year yield, Baa spread from Yahoo Finance and FRED), at a
chosen severity, and shows the most similar past events as the explanation.

Back-test (purged leave-one-out and time-respecting, 81 events): the 1-in-10 historical rule covers 85-89% of
real outcomes with ~$0.4M average excess; the hand-written matrix covers 95% with ~$4.7M excess.
Details, including what did *not* work: [docs/history-calibrated-stress.md](docs/history-calibrated-stress.md).

## Measured results

| What | Result | Source |
|---|---|---|
| Event classification (1,099 labelled finance tweets) | macro-F1 **0.81** hybrid vs 0.78 embeddings vs 0.62 rules | `evaluation/results/latest.md` |
| Sentiment (2,388 labelled finance tweets) | macro-F1 **0.66** FinBERT vs 0.60 lexicon baseline | `evaluation/results/latest.md` |
| Stress scenarios (81 events, 2008-2025) | 85-89% coverage at 1/10 the over-reserving of the hand-written matrix | `evaluation/results/analog_backtest.md` |
| Live NewsAPI (50 articles) | false stress triggers 1 -> 0 after the relevance rules | `docs/research.md` |
| Live GDELT (119 articles) | 130 -> 64 signals after the relevance fixes; no junk entities | `docs/research.md` |

## Quick start

Requires Python 3.11+. The default configuration needs no model downloads and no API keys.

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest                                          # 222 tests
python -m prism.cli analyze "Fed signals more rate hikes as inflation stays high"
python -m prism.cli run --sources replay --limit 100      # one ETL run over the sample data
python -m uvicorn prism.api.app:create_app --factory --reload
```

Then open <http://127.0.0.1:8000/docs>. Try `POST /api/stress-test` with
`{"event_type": "Bankruptcy", "headline": "Regional lender collapses after a deposit run"}`, or
`GET /api/analogs?text=bank collapses after a deposit run`.

Dashboard (second terminal, API running):

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Real models (FinBERT ~440 MB, bge-small ~130 MB, spaCy ~13 MB; downloaded on first use):

```bash
pip install -r requirements-ml.txt
python -m spacy download en_core_web_sm
# .env (copy .env.example): SENTIMENT_BACKEND=finbert EVENT_BACKEND=hybrid ENTITY_BACKEND=spacy ANALOG_ENCODER=embedding
python ../scripts/fetch_replay_data.py && python ../evaluation/run_eval.py     # labelled data + metrics
python ../evaluation/backtest_analogs.py                                       # scenario back-test
```

Live sources: `ENABLED_SOURCES=replay,gdelt_gkg,newsapi`, `NEWSAPI_KEY=...`, `SCHEDULER_ENABLED=true`.
`gdelt_gkg` reads GDELT's static 15-minute files (the GDELT search API rate-limits hard); the free NewsAPI plan
is 24 h delayed and capped at 100 requests/day (`docs/research.md`).

## Repository layout

```
backend/
  prism/
    core/       contracts, protocols, event taxonomy                  (imports nothing else)
    etl/        extract/ (newsapi, gdelt, gdelt_gkg, replay)  transform/ (normalize, language, dedupe)
                load/ (jsonl sink)  pipeline.py  scheduler.py
    nlp/        engine.py  entities/  sentiment/  events/  impact/  embeddings.py      <- the Risk Engine
    modules/    stress/  portfolio, scenarios, engine, trigger, analogs, calibrated    <- Module B
    storage/    SQLAlchemy models + SqlStore (SQLite default, PostgreSQL via DATABASE_URL)
    api/        FastAPI app + routers          bootstrap.py (composition root)   cli.py   config.py
  config/       companies, event rules/prototypes, relevance, impact, scenarios, portfolio, sources,
                analog_events.yaml (curated) -> analog_library.json (computed market reactions)
  tests/
frontend/       Streamlit dashboard (app.py, prism_client.py, prism_charts.py)
evaluation/     run_eval.py, backtest_analogs.py, label maps, results/
scripts/        fetch_replay_data.py, build_analog_library.py, smoke_models.py
data/           replay/ (sample + fetched datasets), raw/, processed/, eval/   (mostly git-ignored)
docs/           architecture, NLP engine, history-calibrated stress, research, roadmap, pitch
```

Dependency rule, enforced by `backend/tests/test_architecture.py`: `core` imports nothing; `etl`, `nlp`, `storage`
and `modules` import only `core`; only the API, `bootstrap`, `cli` and `config` see everything. To add a
source, model or downstream module, implement one protocol and register it in `bootstrap.py`.

## Status

| Area | State |
|---|---|
| ETL: replay, NewsAPI (verified live), GDELT GKG files (verified live), GDELT DOC API (429 from our network) | built, tested |
| Risk Engine: FinBERT, hybrid events with calibrated abstention, dictionary + spaCy entities, impact, confidence | built, evaluated on labelled data, tuned on live data |
| Module B: history-calibrated scenarios, analogs, confirmed auto-trigger, hand-written matrix for comparison | built, back-tested |
| Streamlit dashboard | built, tested headless, checked in a browser |
| PostgreSQL, Docker | written, **never run** |
| Known gaps | macro-direction rules cover a fixed list of patterns; 81-event library; 3 risk factors; impact score not calibrated against returns |

## Data and licensing notes

- `data/replay/sample_news.jsonl` is **synthetic** (flagged in every record); it is not real news.
- Labelled tweets come from MIT-licensed Hugging Face datasets and are not committed.
- `analog_library.json` holds derived daily market moves from Yahoo Finance and FRED for 81 dates.
- The free NewsAPI plan is for development only and must not back a hosted demo.

## Documentation

[History-calibrated stress (the novel idea)](docs/history-calibrated-stress.md) | [Architecture](docs/architecture.md) |
[NLP Risk Engine](docs/nlp-risk-engine.md) | [Research notes](docs/research.md) | [Roadmap](docs/roadmap.md) |
[Pitch and demo script](docs/pitch.md) | [Evaluation](evaluation/README.md) | [Dashboard](frontend/README.md)
