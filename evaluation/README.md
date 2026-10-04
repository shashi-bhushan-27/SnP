# Evaluation

The deck is only allowed to show numbers that come from here. Nothing in this folder has been run yet
(the harness is the next engine task, see `docs/roadmap.md`); this file is the plan and the data contract.

## What is evaluated

| Signal | Ground truth | Metrics | Baseline to beat |
|---|---|---|---|
| Sentiment score | `zeroshot/twitter-financial-news-sentiment` (valid split, 2,486 tweets, Bearish/Bullish/Neutral) and Financial PhraseBank (4,846 sentences, CC BY-NC-SA 3.0, **not** redistributed in this repo) | accuracy, per-class P/R/F1, macro-F1; the continuous score is binned at +/-0.2 | lexicon backend vs FinBERT |
| Event type | `zeroshot/twitter-financial-news-topic` (valid split, 4,118 tweets) through `label_maps/hf_topic_to_event.json`, restricted to `exact` (and separately `exact+approx`) labels | macro-F1, confusion matrix, per-class recall | rules vs embedding vs hybrid |
| Impact score | 200-300 headlines scored 1-10 by 2-3 team members (blind to the model); plus realised market reaction (below) | MAE, RMSE, Spearman rho vs the human mean; inter-annotator agreement reported alongside | constant (median) predictor |
| Confidence | the three above | calibration: accuracy per confidence bucket; does low confidence flag the errors? | none |

### Impact against market reaction (event study)

For signals on tracked tickers, fit a market model on a clean estimation window (e.g. 120 trading days ending
10 days before the event), compute the cumulative abnormal return over [0, +1] days, and report the
Spearman correlation between `impact_score` and `|CAR|`. This is also how the weights in
`backend/config/impact.yaml` can be calibrated (ridge regression of `|CAR|` on the five features).
Prices: `yfinance`. Use only data available at the signal timestamp: no look-ahead.

## Rules that keep the numbers honest

- Split by **time** (or at least by source) when calibrating on market data; tune on one split, report on another.
- Report sample sizes. 26 synthetic rows in `data/replay/sample_news.jsonl` are a smoke test, not a benchmark.
- Show the lexicon baseline next to FinBERT; a number without a baseline means nothing.
- Hand-labelled data and any dataset with a non-commercial licence stay out of git (`data/eval/` is git-ignored).
- The brief's example stress results are illustrative; stress outputs are scenarios, not forecasts, and are not evaluated for accuracy.

## Files

- `label_maps/hf_topic_to_event.json`: draft mapping, review it before quoting event-F1.
- `data/eval/` (git-ignored): labelled sets live here. Expected JSONL fields: `text`, and any of `label_sentiment` (negative|neutral|positive), `label_event` (an event type), `label_impact` (1-10).
- `scripts/fetch_replay_data.py` downloads the two tweet datasets.
