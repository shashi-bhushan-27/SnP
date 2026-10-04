# Research notes

Findings that shaped the design, with sources. Research was done 4-5 Oct 2026. "Verified" means checked
against the live page, API or a real call; anything I could not check is listed at the bottom.

## 1. Data sources

| Finding | Source | What it changed |
|---|---|---|
| GDELT DOC 2.0 `mode=artlist&format=json` returns article metadata (title, URL, date, domain, language, country) and **no article text**. `maxrecords` is capped at 250, `timespan` has a 15-minute minimum, the search window is the rolling last 3 months, and queries support `sourcelang:`, `theme:`, `tone` filters. | [GDELT DOC 2.0 announcement](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) | NLP for GDELT runs on **headlines**. Incremental fetches use `timespan` (computed from the cursor, minimum 15 min). |
| GDELT accepts "one request every 5 seconds" per IP, but the limiter is stricter in practice: a request 5.2 s after the previous one can still get a 429 and the gate then stays closed for about a minute. Exponential backoff (60 s, 120 s, ...) is the community workaround. | [cyanheads/gdelt-mcp-server#44](https://github.com/cyanheads/gdelt-mcp-server/issues/44), [gdelt-py](https://rbozydar.github.io/py-gdelt/user-guide/rest-apis/) | `GdeltSource` spaces requests (6 s), backs off 60 s / 120 s on 429 and then raises `SourceUnavailable`; the pipeline skips the source and keeps going. |
| **Observed:** the first call from the dev machine returned HTTP 200 with `{}`, and every later call (including one after a 150 s wait) returned 429. | live calls, 4-5 Oct | The adapter treats `{}` as "no results". The live response format is **not yet verified** (see section 6). |
| GDELT's index refreshes about every 15 minutes. | brief + GDELT blog | Default GDELT poll interval is 5 min; polling faster adds nothing and risks the limiter. |
| NewsAPI free Developer plan: **100 requests/day, articles delayed 24 h, one month of history, for development only ("cannot be used in a staging or production environment"), CORS for localhost only**. Paid tiers start at $449/month for real-time. | [newsapi.org/pricing](https://newsapi.org/pricing) | The original plan's "poll every 30-60 s" would exhaust the daily quota in under two hours. `QuotaGuard` caps usage (default 80/day), poll interval is 20 min, and the key stays server-side. NewsAPI is therefore a *delayed* second source; **replay** is the real-time demo. Do not deploy a hosted demo with a free key. |
| `zeroshot/twitter-financial-news-sentiment`: 11,931 tweets (9,938 train / 2,486 valid), labels 0 Bearish, 1 Bullish, 2 Neutral, columns `text,label`, MIT license. | [dataset card](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Second source (social) and sentiment test set, with no Kaggle login needed. |
| `zeroshot/twitter-financial-news-topic`: 21,107 tweets (16,990 / 4,118), 20 topic labels (Fed, Earnings, M&A, Legal/Regulation, Macro, Politics, Personnel Change, ...), MIT license. | [dataset card](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic) | Test set for event classification through a label map (`evaluation/label_maps`). |
| Financial PhraseBank: 4,846 sentences at >=50% annotator agreement (2,264 at 100%), **CC BY-NC-SA 3.0** (non-commercial). | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | Fine for a hackathon; never committed to the repo. This is the data FinBERT was fine-tuned on, so evaluating FinBERT on it flatters FinBERT. |

## 2. Models

| Finding | Source | What it changed |
|---|---|---|
| FinBERT (`ProsusAI/finbert`): BERT further trained on a financial corpus and fine-tuned on Financial PhraseBank; classes positive / negative / neutral; loads with `pipeline("text-classification", model="ProsusAI/finbert")`. | [model card](https://huggingface.co/ProsusAI/finbert) | `score = P(positive) - P(negative)`, `confidence = max probability`. Lighter fallback if latency matters: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` (82M params, Apache-2.0, verified via the HF API). |
| In a 2026 benchmark of zero-shot text classification (22 datasets, 38 checkpoints), rerankers were best (Qwen3-Reranker-8B, macro-F1 0.72), embedding models gave the best accuracy/latency trade-off, 4-12B instruction-tuned LLMs reached up to 0.67, and **NLI cross-encoders plateau as they grow**. | [BTZSC, arXiv 2603.11991](https://arxiv.org/abs/2603.11991) | Event classification defaults to **rules + sentence-embedding nearest-prototype**, not an NLI pipeline. NLI and LLM backends stay optional comparison points for the evaluation. |
| Candidate sizes and licenses (HF API): `BAAI/bge-small-en-v1.5` 33M MIT; `sentence-transformers/all-MiniLM-L6-v2` 23M Apache-2.0; `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` 184M MIT; `facebook/bart-large-mnli` 407M MIT; `dslim/bert-base-NER` 108M MIT. | [Hugging Face API](https://huggingface.co/docs/hub/api) | Default embedding model is `bge-small-en-v1.5` (small enough for CPU and a laptop demo). |
| A zero-shot NLI pipeline for news-to-market prediction underperformed baselines, particularly on negative moves; the explainability and uncertainty layer was the useful part. | [arXiv 2606.12210](https://arxiv.org/abs/2606.12210) | Every signal carries a **confidence** and an **explanation**; impact is documented as a severity prior, not a price forecast; the stress trigger requires a minimum confidence. |
| Event-study methodology: estimate normal returns with a market model on a clean window, abnormal return = actual - expected, cumulate to a CAR. | e.g. [MacKinlay (1997) overview](https://www.eventstudytools.com/expected-return-models) | Plan for calibrating `impact.yaml` (see `docs/nlp-risk-engine.md`, `evaluation/README.md`). |

## 3. Differences from the original plan, and why

| Original plan | Now | Reason |
|---|---|---|
| Poll every 30-60 s | Per-source intervals: GDELT 5 min, NewsAPI 20 min, replay 5 s | Rate limits and refresh rates above. |
| NewsAPI as a live source | NewsAPI as a delayed source with a daily budget | 24 h delay and dev-only terms. |
| spaCy NER + ticker dictionary | Dictionary linker (names, aliases, cashtags, finance-context rule for ambiguous names) | Closed universe of ~40 names: higher precision, no model download. NER can slot in behind the same protocol. |
| Impact formula with a "novelty" term | "Corroboration" (distinct outlets reporting the story) instead | Novelty needs a reference corpus we do not have in replay; corroboration comes free from de-duplication. |
| Credit shock "+5%" in the shock table | `credit_spread_bps` (150 bps for Geopolitical, ...) | "+5%" has no unit; 500 bps would dwarf the equity leg. **Your call: change the numbers in `scenarios.yaml` if you meant something else.** |
| Flat percentage shocks on asset values | Per-asset sensitivities (beta, duration, convexity, spread duration, DV01) | Bonds, loans, equities and swaps respond differently; the swap hedge visibly offsets losses. |
| PostgreSQL | SQLite by default, PostgreSQL through `DATABASE_URL` | Zero setup for the team; same SQLAlchemy models. |
| React dashboard | Open decision (Streamlit recommended first), see `frontend/README.md` | 7-day deadline. |

## 4. Resources from the brief

- **GDELT, NewsAPI, yfinance**: used as described above (yfinance is used in the evaluation/calibration step).
- **Kaggle datasets** (financial news sentiment, stock tweets, Financial Transactions): require a Kaggle login and were
  not touched; the Hugging Face tweet datasets cover the same replay/evaluation need.
- **Salad Money Open Banking / Kaggle Financial Transactions**: these are *consumer* transactions, not loans, bonds or
  derivatives. The default Module B portfolio is therefore synthetic and hand-specified; deriving exposures from
  transaction categories is planned (`docs/roadmap.md`) and should be described honestly as seeding a synthetic book.

## 5. Weak spots to be upfront about

- The lexicon sentiment baseline is crude (and `inflation`, `tariffs` count as negative words by design).
- Event classification is single-label; a story can be Regulatory and Legal at once.
- Impact weights are priors until calibrated; "impact" is direction-agnostic (record earnings can score 7+), which is why the
  KPI "high risk" additionally requires negative sentiment.
- Near-duplicate detection is lexical (word 3-grams); paraphrases from different outlets are not merged.
- Replay data is not live; the demo must say so.

## 6. Learned from live data (5 Oct 2026)

| Observation | Evidence | Change made |
|---|---|---|
| NewsAPI's free plan really is delayed: the newest article was **24.0 h** old. | live request through `NewsApiSource` | Documented; replay stays the real-time demo path. |
| A broad query searched over full content returns noise (NBA contracts, box office, politics). | 29 results, mostly off-topic | `searchIn=title,description` by default. |
| A finance-domain filter (CNBC, Business Insider, Fortune, Bloomberg, ...) gives on-topic results, 1-3 days old. | 50 results, 4 publishers | `domains` list is the default in `sources.yaml`; outlets added to the reliability table. |
| On 50 live articles the first engine produced 36 signals and **one false stress trigger** ("The spice of the matter" -> Macroeconomic 7.6 -> -$5.4M), plus listicles ("3 Canadian AI Stocks...") and a misattribution (a grocery chain closing stores -> Walmart, mentioned only in the body). | pipeline run, NewsAPI + FinBERT | Headline-first evidence, entity salience, commentary filter (`config/relevance.yaml`), and trigger confirmation (>= 2 outlets or a trusted source). Re-run on the same 50 articles: **15 signals, 0 false triggers**. |
| FinBERT runs on CPU at **~30 ms/headline** (24 s first load); weights are `pytorch_model.bin` (~440 MB, no safetensors). | `scripts/smoke_models.py` | `SENTIMENT_BACKEND=finbert` verified. |
| FinBERT misreads macro direction: *"CPI rises 0.5%, above forecasts"* -> **+0.86**; *"Fed signals further rate hikes"* -> neutral. | smoke test | Open: macro headlines need a polarity rule on top of FinBERT (rising inflation / higher rates = negative for markets). |
| The lexicon baseline scores 92% on the synthetic sample. | smoke test | **Not meaningful**: the word list and the sample headlines were written together. Real evaluation needs the labelled tweet sets. |
| GDELT DOC API still answers 429 from this network a day later; GDELT's raw 15-minute file feed (`data.gdeltproject.org/gdeltv2/lastupdate.txt`) is reachable. The GKG file (~2.5 MB zipped per 15 min) carries titles, organisations, tone and themes. | live requests | Option: a GKG-file adapter as the GDELT path (not built; downloads files every 15 min). |
| figshare "Effects of Twitter sentiment on stock price returns" (CC BY 4.0, 377 KB): daily sentiment *counts* per DJIA-30 stock, 2013-06 to 2014-09, no tweet text. | figshare API | Usable for an impact/returns sanity check, not as NLP input. |
| The Kaggle "financial news with ticker-level sentiment" page needs a login. | fetch attempt | Needs a manual download by the team. |

## 7. Not verified yet

- The **live GDELT DOC response format** (fixture hand-written from the docs; 429 from this network).
- **bge-small embeddings and spaCy NER on real text** (adapters unit-tested with fakes; models not downloaded yet).
- FinBERT license (not stated on the model card as fetched).
- PostgreSQL and Docker paths (written, never run).
- That the HF CSV files have exactly the column layout the dataset cards describe (converter unit-tested on that layout; nothing downloaded).
