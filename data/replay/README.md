# Replay data

Files here are streamed through the normal ETL pipeline by the `replay` source (`ReplaySource`),
which makes demos deterministic when live APIs are slow, delayed or rate-limited.

## `sample_news.jsonl` (committed)

26 **synthetic** headlines written for tests and demos. They are not real news: every record carries
`"synthetic": true`, and severe events (bankruptcy, breach, downgrade, lawsuit) use fictional companies.
Real company names appear only in routine positive items.

The file is designed to exercise every pipeline path, and `backend/tests/test_pipeline.py` relies on it:

The order tells a 5-minute demo story when streamed in batches of 3: calm company news (1-5), noise the engine
filters (6-8), rising macro tension (9-13), escalation that fires the first stress test (14-15), then credit,
bankruptcy and market-shock events (19-21), and the aftermath.

| Records | Purpose |
|---|---|
| 13 vs 12, 15 vs 14 | near-duplicates from different outlets (merged, raise corroboration) |
| 25 vs 1 | exact duplicate from a second outlet |
| 8 | Spanish record (rejected by the language filter) |
| 6, 7 | no tracked entity and no recognisable event (dropped as irrelevant) |
| 26 | social-style post with two cashtags (two signals) |
| 14, 19, 20, 21 | high-impact negative events that should fire stress scenarios |

`label_sentiment` / `label_event` are hand labels kept for a smoke evaluation; with 26 rows they are
**not** a benchmark.

## Real replay data (not committed, fetched on demand)

`scripts/fetch_replay_data.py` converts the MIT-licensed Hugging Face datasets
`zeroshot/twitter-financial-news-sentiment` and `zeroshot/twitter-financial-news-topic`
(~21k labelled finance tweets) into JSONL files in this folder. They are git-ignored.
