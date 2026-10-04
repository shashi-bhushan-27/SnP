"""The data scripts are plain files outside the package; load them by path. No network is used."""

from __future__ import annotations

import csv
import importlib.util
import io
import json

from prism.config import REPO_ROOT


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(csv_text: str):
    return csv.DictReader(io.StringIO(csv_text))


def test_sentiment_labels_are_converted_to_the_shared_vocabulary() -> None:
    fetch = load_script("fetch_replay_data")
    out = list(fetch.convert_sentiment(rows('text,label\n"$TSLA falls, sells off",0\nNvidia beats,1\nFed meets today,2\n')))
    assert [r["label_sentiment"] for r in out] == ["negative", "positive", "neutral"]
    assert out[0]["text"] == "$TSLA falls, sells off" and out[0]["publisher"] == "twitter"


def test_topic_label_ids_map_to_names_and_blank_rows_are_skipped() -> None:
    fetch = load_script("fetch_replay_data")
    out = list(fetch.convert_topic(rows("text,label\nQ3 results out,5\n,3\nMerger agreed,13\nRates on hold,1\n")))
    assert [r["label_topic"] for r in out] == ["Earnings", "M&A | Investments", "Fed | Central Banks"]
    assert len(fetch.TOPIC_LABELS) == 20


def test_write_jsonl_round_trips_unicode(tmp_path) -> None:
    fetch = load_script("fetch_replay_data")
    path = tmp_path / "nested" / "x.jsonl"
    assert fetch.write_jsonl(path, [{"text": "café ↑ 5%"}, {"text": "b"}]) == 2
    assert [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()] == ["café ↑ 5%", "b"]
