# Evaluation results

Generated 2026-10-07T19:41:41+00:00 by `evaluation/run_eval.py`.

## Sentiment - 2388 labelled finance tweets

Gold labels: {'negative': 347, 'positive': 475, 'neutral': 1566}. Score bands: <= -0.2 negative, >= +0.2 positive.

| Model | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 | ms/text |
|---|---|---|---|---|---|---|
| lexicon | 71.0% | 0.596 | 0.480 | 0.803 | 0.505 | 0.0 |
| finbert | 66.2% | 0.624 | 0.568 | 0.729 | 0.575 | 43.4 |

FinBERT plain argmax (no banding): accuracy 71.7%, macro-F1 0.663.

## Event classification - 1099 topic-labelled tweets (exact label mappings)

Gold distribution: {'Earnings': 242, 'Macroeconomic': 629, 'Merger/Acquisition': 116, 'Leadership Change': 112}

| Classifier | Accuracy | Macro-F1 | Answered (not Other) | Accuracy when answered | ms/text |
|---|---|---|---|---|---|
| rules | 48.0% | 0.621 | 51.4% | 93.3% | 0.3 |
| embedding | 62.1% | 0.775 | 98.0% | 63.3% | 22.8 |
| hybrid | 68.3% | 0.811 | 98.2% | 69.6% | 19.6 |

## Event classification - 2829 topic-labelled tweets (exact + approximate mappings)

Gold distribution: {'Product Launch': 852, 'Other': 336, 'Earnings': 339, 'Macroeconomic': 629, 'Regulatory': 119, 'Merger/Acquisition': 116, 'Leadership Change': 112, 'Geopolitical': 249, 'Credit Event': 77}

| Classifier | Accuracy | Macro-F1 | Answered (not Other) | Accuracy when answered | ms/text |
|---|---|---|---|---|---|
| rules | 33.7% | 0.398 | 30.9% | 75.2% | 0.3 |
| embedding | 46.9% | 0.475 | 93.1% | 48.6% | 16.0 |
| hybrid | 49.1% | 0.485 | 93.8% | 50.7% | 16.2 |

Caveats: tweets are shorter and noisier than news headlines; the topic labels map onto PRISM's taxonomy only partially (see `label_maps/hf_topic_to_event.json`); FinBERT was fine-tuned on Financial PhraseBank, not on these tweets.
