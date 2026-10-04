#!/usr/bin/env python
"""Download MIT-licensed finance-tweet datasets from Hugging Face and convert them to replay JSONL.

    python scripts/fetch_replay_data.py                 # validation splits (~3.5k + ~2.5k tweets, ~1.5 MB)
    python scripts/fetch_replay_data.py --split train   # training splits instead
    python scripts/fetch_replay_data.py --split both

Sources (license: MIT, English finance tweets collected via the Twitter API):
    zeroshot/twitter-financial-news-sentiment   labels Bearish / Bullish / Neutral
    zeroshot/twitter-financial-news-topic       20 topic labels (see evaluation/label_maps)

Output (git-ignored) goes to data/replay/ and is streamed by the `replay` source:
    tweets_sentiment_<split>.jsonl, tweets_topic_<split>.jsonl
Point the pipeline at one with REPLAY_FILE=data/replay/tweets_topic_valid.jsonl
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Iterable, Iterator

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/zeroshot/{repo}/resolve/main/{file}"

SENTIMENT_LABELS = {0: "negative", 1: "positive", 2: "neutral"}  # Bearish, Bullish, Neutral
TOPIC_LABELS = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend",
    "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro",
    "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]
DATASETS = {
    "sentiment": ("twitter-financial-news-sentiment", {"train": "sent_train.csv", "valid": "sent_valid.csv"}),
    "topic": ("twitter-financial-news-topic", {"train": "topic_train.csv", "valid": "topic_valid.csv"}),
}


def convert_sentiment(rows: Iterable[dict[str, str]]) -> Iterator[dict]:
    for row in rows:
        text = (row.get("text") or "").strip()
        if text:
            yield {"publisher": "twitter", "text": text, "label_sentiment": SENTIMENT_LABELS[int(row["label"])]}


def convert_topic(rows: Iterable[dict[str, str]]) -> Iterator[dict]:
    for row in rows:
        text = (row.get("text") or "").strip()
        if text:
            yield {"publisher": "twitter", "text": text, "label_topic": TOPIC_LABELS[int(row["label"])]}


CONVERTERS = {"sentiment": convert_sentiment, "topic": convert_topic}


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["valid", "train", "both"], default="valid")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "replay")
    args = parser.parse_args()
    splits = ["train", "valid"] if args.split == "both" else [args.split]

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for kind, (repo, files) in DATASETS.items():
            for split in splits:
                url = URL.format(repo=repo, file=files[split])
                response = client.get(url)
                response.raise_for_status()
                rows = csv.DictReader(io.StringIO(response.text))
                target = args.out / f"tweets_{kind}_{split}.jsonl"
                print(f"{target.relative_to(REPO_ROOT)}: {write_jsonl(target, CONVERTERS[kind](rows))} records  <- {url}")


if __name__ == "__main__":
    main()
