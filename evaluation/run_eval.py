#!/usr/bin/env python
"""Evaluate Risk Engine components on labelled data and write a report.

    python evaluation/run_eval.py                      # sentiment + events, all backends
    python evaluation/run_eval.py --limit 500          # quicker
    python evaluation/run_eval.py --skip-models        # lexicon + rules only (no downloads)

Data (fetch with scripts/fetch_replay_data.py):
    data/replay/tweets_sentiment_valid.jsonl   label_sentiment in {negative, neutral, positive}
    data/replay/tweets_topic_valid.jsonl       label_topic (20 topics) -> PRISM events via the label map

Writes evaluation/results/latest.json and evaluation/results/latest.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

from prism.config import Settings  # noqa: E402
from prism.core.taxonomy import EventType  # noqa: E402
from prism.etl.transform import clean_text  # noqa: E402
from prism.nlp.events import EmbeddingEventClassifier, HybridEventClassifier, RuleEventClassifier  # noqa: E402
from prism.nlp.sentiment import FinBertSentiment, LexiconSentiment  # noqa: E402

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
BAND = 0.2


def load(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[:limit] if limit else rows


def band(score: float) -> str:
    return "positive" if score >= BAND else "negative" if score <= -BAND else "neutral"


# ------------------------------------------------------------------------------------ sentiment
def eval_sentiment(rows: list[dict], models: list) -> dict:
    texts = [clean_text(r["text"]) for r in rows]
    gold = [r["label_sentiment"] for r in rows]
    out = {"n": len(rows), "gold_distribution": dict(Counter(gold)), "models": {}}
    for model in models:
        model.predict(texts[:2])  # warm-up / model load
        t = perf_counter()
        preds = model.predict(texts)
        ms = 1000 * (perf_counter() - t) / len(texts)
        banded = [band(p.score) for p in preds]
        entry = {
            "accuracy": accuracy_score(gold, banded),
            "macro_f1": f1_score(gold, banded, labels=SENTIMENT_LABELS, average="macro", zero_division=0),
            "per_class": classification_report(gold, banded, labels=SENTIMENT_LABELS, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(gold, banded, labels=SENTIMENT_LABELS).tolist(),
            "ms_per_text": ms,
            "mean_confidence": sum(p.confidence for p in preds) / len(preds),
        }
        if model.name == "finbert":  # also the plain argmax, to separate the model from the +/-0.2 banding
            argmax = [max(p.probabilities, key=p.probabilities.get) for p in preds]
            entry["argmax_accuracy"] = accuracy_score(gold, argmax)
            entry["argmax_macro_f1"] = f1_score(gold, argmax, labels=SENTIMENT_LABELS, average="macro", zero_division=0)
        out["models"][model.name] = entry
    return out


# ------------------------------------------------------------------------------------ events
def eval_events(rows: list[dict], label_map: dict, classifiers: list, qualities: tuple[str, ...]) -> dict:
    mapped = []
    for r in rows:
        spec = label_map[r["label_topic"]]
        if spec["quality"] in qualities and spec["maps_to"]:
            mapped.append((clean_text(r["text"]), spec["maps_to"], r["label_topic"]))
    texts = [m[0] for m in mapped]
    out = {"n": len(mapped), "qualities": list(qualities), "gold_distribution": dict(Counter(m[1][0] for m in mapped)), "models": {}}
    for clf in classifiers:
        t = perf_counter()
        preds = clf.classify(texts)
        ms = 1000 * (perf_counter() - t) / max(1, len(texts))
        pred_labels = [p.event_type.value for p in preds]
        # a prediction is correct if it is any of the acceptable PRISM types for that topic
        gold = [p if p in acceptable else acceptable[0] for p, (_, acceptable, _) in zip(pred_labels, mapped)]
        answered = [p != EventType.OTHER.value for p in pred_labels]
        n_answered = sum(answered)
        correct = [p == g for p, g in zip(pred_labels, gold)]
        classes = sorted(set(gold))
        out["models"][clf.name] = {
            "accuracy": sum(correct) / len(correct),
            "macro_f1": f1_score(gold, pred_labels, labels=classes, average="macro", zero_division=0),
            "answered_rate": n_answered / len(pred_labels),
            "accuracy_when_answered": (sum(c for c, a in zip(correct, answered) if a) / n_answered) if n_answered else 0.0,
            "per_class_recall": {
                c: sum(p == g == c for p, g in zip(pred_labels, gold)) / max(1, gold.count(c)) for c in classes
            },
            "ms_per_text": ms,
        }
    return out


# ------------------------------------------------------------------------------------ report
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(report: dict) -> str:
    lines = [f"# Evaluation results\n\nGenerated {report['generated_at']} by `evaluation/run_eval.py`.\n"]
    s = report.get("sentiment")
    if s:
        lines += [
            f"## Sentiment - {s['n']} labelled finance tweets\n",
            f"Gold labels: {s['gold_distribution']}. Score bands: <= -{BAND} negative, >= +{BAND} positive.\n",
            "| Model | Accuracy | Macro-F1 | Neg F1 | Neu F1 | Pos F1 | ms/text |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, m in s["models"].items():
            pc = m["per_class"]
            lines.append(
                f"| {name} | {pct(m['accuracy'])} | {m['macro_f1']:.3f} | {pc['negative']['f1-score']:.3f} | "
                f"{pc['neutral']['f1-score']:.3f} | {pc['positive']['f1-score']:.3f} | {m['ms_per_text']:.1f} |"
            )
        fb = s["models"].get("finbert")
        if fb:
            lines.append(f"\nFinBERT plain argmax (no banding): accuracy {pct(fb['argmax_accuracy'])}, macro-F1 {fb['argmax_macro_f1']:.3f}.")
    for key, title in (("events_exact", "exact label mappings"), ("events_all", "exact + approximate mappings")):
        e = report.get(key)
        if not e:
            continue
        lines += [
            f"\n## Event classification - {e['n']} topic-labelled tweets ({title})\n",
            f"Gold distribution: {e['gold_distribution']}\n",
            "| Classifier | Accuracy | Macro-F1 | Answered (not Other) | Accuracy when answered | ms/text |",
            "|---|---|---|---|---|---|",
        ]
        for name, m in e["models"].items():
            lines.append(
                f"| {name} | {pct(m['accuracy'])} | {m['macro_f1']:.3f} | {pct(m['answered_rate'])} | "
                f"{pct(m['accuracy_when_answered'])} | {m['ms_per_text']:.1f} |"
            )
    lines.append(
        "\nCaveats: tweets are shorter and noisier than news headlines; the topic labels map onto PRISM's taxonomy only "
        "partially (see `label_maps/hf_topic_to_event.json`); FinBERT was fine-tuned on Financial PhraseBank, not on these tweets."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-models", action="store_true", help="lexicon and rules only")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evaluation" / "results")
    args = parser.parse_args()

    settings = Settings(_env_file=None)
    cfg = settings.config_dir
    replay = REPO_ROOT / "data" / "replay"
    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    sentiment_models = [LexiconSentiment()] + ([] if args.skip_models else [FinBertSentiment(settings.finbert_model)])
    print("sentiment ...", flush=True)
    report["sentiment"] = eval_sentiment(load(replay / "tweets_sentiment_valid.jsonl", args.limit), sentiment_models)

    rules = RuleEventClassifier.from_yaml(cfg / "event_rules.yaml")
    classifiers = [rules]
    if not args.skip_models:
        embedding = EmbeddingEventClassifier.from_yaml(cfg / "event_prototypes.yaml", model_name=settings.embedding_model)
        classifiers += [embedding, HybridEventClassifier(rules, embedding)]
    label_map = json.loads((REPO_ROOT / "evaluation" / "label_maps" / "hf_topic_to_event.json").read_text(encoding="utf-8"))["labels"]
    topic_rows = load(replay / "tweets_topic_valid.jsonl", args.limit)
    print("events ...", flush=True)
    report["events_exact"] = eval_events(topic_rows, label_map, classifiers, ("exact",))
    report["events_all"] = eval_events(topic_rows, label_map, classifiers, ("exact", "approx"))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / "latest.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
