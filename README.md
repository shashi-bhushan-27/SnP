# PRISM: AI Financial Risk Intelligence and Portfolio Stress Engine

Turns unstructured financial text (news, social posts) into structured risk signals, then uses those
signals to stress-test a synthetic wholesale-banking portfolio.

```
GDELT / NewsAPI / replay  ->  ETL  ->  NLP Risk Engine  ->  risk signals  ->  stress test  ->  dashboard
                            (extract, transform, load)   sentiment, event,      (Module B)
                                                         impact, confidence
```

Every signal looks like this and is served over a REST API (and written to JSONL):

```json
{ "entity": "Tesla", "ticker": "TSLA", "event_type": "Regulatory",
  "sentiment_score": -0.67, "impact_score": 7.27, "confidence": 0.86, "source": "replay", "timestamp": "..." }
```

## Quick start

Requires Python 3.11+. The default configuration needs no model downloads and no API keys.

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest                                          # run the tests
python -m prism.cli analyze "Fed signals more rate hikes as inflation stays high"
python -m prism.cli run --sources replay --limit 100      # one ETL run over the sample data
python -m uvicorn prism.api.app:create_app --factory --reload
```

Then open <http://127.0.0.1:8000/docs>. After an ETL run, try `GET /api/risk-signals?min_impact=7`,
`GET /api/stress-runs` and `POST /api/stress-test` with `{"event_type": "Geopolitical", "impact_score": 9}`.

Dashboard (second terminal, API running):

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Real models (FinBERT ~440 MB, optional bge-small ~130 MB and spaCy, downloaded on first use):

```bash
pip install -r requirements-ml.txt
# in .env (copy .env.example):  SENTIMENT_BACKEND=finbert   EVENT_BACKEND=hybrid   ENTITY_BACKEND=spacy
python ../scripts/smoke_models.py      # lexicon vs FinBERT on labelled JSONL
```

Live sources: set `ENABLED_SOURCES=replay,gdelt,newsapi`, add `NEWSAPI_KEY`, and `SCHEDULER_ENABLED=true`.
Read `docs/research.md` first: GDELT rate-limits aggressively and the free NewsAPI plan is delayed 24 h and capped at 100 requests/day.

## Repository layout

```
backend/
  prism/
    core/       contracts, protocols, event taxonomy          (imports nothing else)
    etl/        extract/ (gdelt, newsapi, replay)  transform/ (normalize, language, dedupe)
                load/ (jsonl sink)  pipeline.py  scheduler.py
    nlp/        engine.py  entities/  sentiment/  events/  impact/       <- the Risk Engine
    modules/    stress/    portfolio, scenarios, stress engine, trigger   <- Module B
    storage/    SQLAlchemy models + SqlStore (SQLite default, PostgreSQL via DATABASE_URL)
    api/        FastAPI app + routers          bootstrap.py (composition root)   cli.py   config.py
  config/       YAML: companies, event rules/prototypes, impact weights, scenarios, portfolio, sources
  tests/
data/           replay/ (sample + fetched datasets)  raw/  processed/  eval/   (mostly git-ignored)
evaluation/     evaluation plan and label maps          scripts/   data download
frontend/       dashboard (not built yet)               docs/      architecture, research, engine, roadmap
```

Dependency rule, enforced by `backend/tests/test_architecture.py`: `core` imports nothing; `etl`, `nlp`, `storage`
and `modules` import only `core`; only the API, `bootstrap`, `cli` and `config` see everything.
To add a source, model or downstream module, implement one protocol and register it in `bootstrap.py`
(`docs/architecture.md`).

## Status

| Area | State |
|---|---|
| ETL (extract, transform, idempotent load, run reports, scheduler) | built, tested |
| NewsAPI adapter | built, tested, **verified live** (24 h delay confirmed; finance-domain filter on) |
| GDELT DOC adapter | built, tested against mocked HTTP; **live format not verified** (HTTP 429 from the dev network) |
| Risk Engine: entities, relevance/salience rules, events, impact, confidence | built, tested; rules tuned on 50 live articles |
| FinBERT sentiment | **verified** (~30 ms/headline CPU); known weakness on macro direction |
| Embedding event classifier, spaCy NER | adapters unit-tested with fakes; models not downloaded yet |
| Module B stress test (engine, scenarios, confirmed auto-trigger, API) | built, tested; shock numbers are illustrative |
| Streamlit dashboard | built, tested headless, checked in a browser |
| PostgreSQL, Docker | written, **never run** |
| Evaluation harness, impact calibration, transaction-seeded portfolio | not started (`docs/roadmap.md`) |

## Data and licensing notes

- `data/replay/sample_news.jsonl` is **synthetic** (flagged in every record); it is not real news.
- `scripts/fetch_replay_data.py` fetches MIT-licensed Hugging Face tweet datasets; they are not committed.
- Financial PhraseBank is CC BY-NC-SA 3.0 and is never committed.
- The free NewsAPI plan is for development only and must not back a hosted demo.

## Documentation

[Architecture](docs/architecture.md) | [NLP Risk Engine](docs/nlp-risk-engine.md) | [Research](docs/research.md) |
[Roadmap](docs/roadmap.md) | [Evaluation plan](evaluation/README.md) | [Frontend notes](frontend/README.md)
