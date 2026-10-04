#!/usr/bin/env python
"""Smoke-test the sentiment backends on a labelled JSONL file (field `label_sentiment`).

    python scripts/smoke_models.py
    python scripts/smoke_models.py --file data/replay/tweets_sentiment_valid.jsonl --limit 500

Reports accuracy (score binned at +/-0.2), latency and the rows each model gets wrong.
On the 25-row synthetic sample this is a smoke test, NOT an evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from prism.nlp.sentiment import FinBertSentiment, LexiconSentiment  # noqa: E402

LABELS = ("negative", "neutral", "positive")


def to_label(score: float, band: float = 0.2) -> str:
    return "positive" if score >= band else "negative" if score <= -band else "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=REPO_ROOT / "data" / "replay" / "sample_news.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--finbert-model", default="ProsusAI/finbert")
    parser.add_argument("--show-errors", type=int, default=10)
    args = parser.parse_args()

    with args.file.open(encoding="utf-8") as fh:
        rows = [r for r in (json.loads(line) for line in fh if line.strip()) if r.get("label_sentiment") in LABELS]
    rows = rows[: args.limit]
    texts, gold = [r["text"] for r in rows], [r["label_sentiment"] for r in rows]
    print(f"{len(rows)} labelled rows from {args.file.name}  (gold: {dict(Counter(gold))})\n")

    for model in (LexiconSentiment(), FinBertSentiment(args.finbert_model)):
        t = perf_counter()
        model.predict(texts[:1])  # loads / downloads the model on first use
        load_s = perf_counter() - t
        t = perf_counter()
        preds = model.predict(texts)
        run_s = perf_counter() - t

        predicted = [to_label(p.score) for p in preds]
        correct = sum(p == g for p, g in zip(predicted, gold))
        per_class = {
            label: f"{sum(p == g == label for p, g in zip(predicted, gold))}/{gold.count(label)}" for label in LABELS
        }
        print(f"== {model.name}: accuracy {correct}/{len(rows)} = {correct / len(rows):.1%}   per class {per_class}")
        print(f"   load {load_s:.1f}s, {1000 * run_s / len(rows):.1f} ms/text, mean confidence {sum(p.confidence for p in preds) / len(preds):.2f}")
        errors = [(g, p, s.score, txt) for g, p, s, txt in zip(gold, predicted, preds, texts) if g != p]
        for g, p, score, txt in errors[: args.show_errors]:
            print(f"   gold={g:<8} pred={p:<8} score={score:+.2f}  {txt[:90]}")
        print()


if __name__ == "__main__":
    main()
