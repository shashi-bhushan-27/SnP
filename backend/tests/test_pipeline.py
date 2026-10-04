from __future__ import annotations

import json

from prism.core.contracts import FetchResult, RawDocument, RiskSignal
from prism.core.errors import SourceUnavailable
from prism.etl.extract import ReplaySource
from prism.etl.load import JsonlSignalSink
from prism.etl.pipeline import EtlPipeline
from prism.modules.stress import Portfolio, ScenarioBook, StressEngine, StressTrigger


class Boom:
    """A source that fails in a configurable way."""

    def __init__(self, name: str, error: Exception) -> None:
        self.name, self.error = name, error

    def fetch(self, cursor, limit):
        raise self.error


class Recorder:
    def __init__(self) -> None:
        self.batches: list[list[RiskSignal]] = []

    def on_signals(self, signals) -> None:
        self.batches.append(list(signals))


class Exploding:
    def on_signals(self, signals) -> None:
        raise RuntimeError("downstream bug")


def build(store, engine, sample_replay, *, extra_sources=(), consumers=()):
    sources = {"replay": ReplaySource(sample_replay)}
    sources.update({s.name: s for s in extra_sources})
    return EtlPipeline(sources=sources, store=store, analyzer=engine, consumers=list(consumers))


def line_count(path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


# ---------------------------------------------------------------- end to end
def test_full_run_accounts_for_every_record(store, engine, sample_replay) -> None:
    report = build(store, engine, sample_replay).run_once(["replay"], limit=100)
    stats = report.sources["replay"]

    assert stats.fetched == line_count(sample_replay)
    assert stats.rejected == 1  # the Spanish record
    assert report.normalized == stats.fetched - stats.rejected
    assert report.duplicates == 3  # 2 near-duplicates + 1 exact duplicate
    assert report.documents_loaded == report.normalized - report.duplicates
    assert report.skipped_irrelevant == 2  # no entity, no event
    assert report.signals_loaded >= report.documents_loaded - report.skipped_irrelevant
    assert set(report.stage_seconds) == {"extract", "transform", "analyze", "load", "notify"}
    assert not report.errors


def test_signals_are_queryable_after_a_run(store, engine, sample_replay) -> None:
    build(store, engine, sample_replay).run_once(limit=100)
    tsla = store.query_signals(ticker="TSLA")
    assert {s.ticker for s in tsla} == {"TSLA"} and tsla
    assert store.stats()["signals"] == len(store.query_signals(limit=500))
    assert store.list_runs()[0]["sources"]["replay"]["fetched"] > 0


def test_cross_source_duplicates_raise_corroboration(store, engine, sample_replay) -> None:
    build(store, engine, sample_replay).run_once(limit=100)
    apple = store.query_signals(ticker="AAPL")
    assert len(apple) == 1  # the same story from two outlets is one signal
    assert apple[0].explanation["corroboration"] == 2


def test_run_is_incremental_and_idempotent(store, engine, sample_replay) -> None:
    pipeline = build(store, engine, sample_replay)
    first = pipeline.run_once(limit=100)
    second = pipeline.run_once(limit=100)  # cursor is at the end
    assert second.sources["replay"].fetched == 0 and second.signals_loaded == 0

    store.set_state("cursor:replay", "0")  # force a full re-read
    third = pipeline.run_once(limit=100)
    assert third.documents_loaded == 0 and third.signals_loaded == 0
    assert third.duplicates == third.normalized
    assert store.stats()["signals"] == first.signals_loaded


def test_merged_near_duplicates_are_remembered_across_runs(store, engine, sample_replay) -> None:
    """GDELT-style overlapping windows re-deliver the same items; merged duplicates must not come back."""
    pipeline = build(store, engine, sample_replay)
    pipeline.run_once(limit=100)
    docs_after_first = len(store.list_documents(limit=500))
    store.set_state("cursor:replay", "0")
    pipeline.run_once(limit=100)
    assert len(store.list_documents(limit=500)) == docs_after_first


def test_rejected_records_are_kept_with_their_reason(store, engine, sample_replay) -> None:
    build(store, engine, sample_replay).run_once(limit=100)
    (reject,) = store.list_rejects()
    assert reject["reason"] == "non_english" and reject["source"] == "replay"


def test_cursor_advances_batch_by_batch(store, engine, sample_replay) -> None:
    pipeline = build(store, engine, sample_replay)
    pipeline.run_once(limit=5)
    assert store.get_state("cursor:replay") == "5"
    pipeline.run_once(limit=5)
    assert store.get_state("cursor:replay") == "10"


# ---------------------------------------------------------------- failure isolation
def test_a_failing_source_does_not_stop_the_others(store, engine, sample_replay) -> None:
    pipeline = build(
        store,
        engine,
        sample_replay,
        extra_sources=[Boom("gdelt", SourceUnavailable("rate limited")), Boom("newsapi", RuntimeError("kaput"))],
    )
    report = pipeline.run_once(limit=100)
    assert report.signals_loaded > 0
    assert report.sources["gdelt"].error == "rate limited"
    assert "kaput" in report.sources["newsapi"].error
    assert len(report.errors) == 2
    assert store.get_state("cursor:gdelt") is None  # cursors of failed sources do not move


def test_unknown_source_is_reported_not_raised(store, engine, sample_replay) -> None:
    report = build(store, engine, sample_replay).run_once(["nope"])
    assert report.sources["nope"].error == "unknown source"


def test_a_poison_document_is_rejected_without_losing_the_batch(store, engine, sample_replay) -> None:
    class PoisonEngine:
        def analyze(self, docs):
            if len(docs) > 1 or "Tesla" in docs[0].text or "$TSLA" in docs[0].text:
                raise ValueError("model crashed")
            return engine.analyze(docs)

    pipeline = EtlPipeline(sources={"replay": ReplaySource(sample_replay)}, store=store, analyzer=PoisonEngine())
    report = pipeline.run_once(limit=100)
    assert report.signals_loaded > 0  # healthy documents still made it through
    assert store.query_signals(ticker="TSLA") == []
    reasons = [r["reason"] for r in store.list_rejects()]
    assert sum(r.startswith("analysis_error") for r in reasons) == 2  # the two poisoned documents, recorded


# ---------------------------------------------------------------- consumers
def test_consumers_receive_only_newly_loaded_signals(store, engine, sample_replay) -> None:
    recorder = Recorder()
    pipeline = build(store, engine, sample_replay, consumers=[recorder])
    report = pipeline.run_once(limit=100)
    assert len(recorder.batches[0]) == report.signals_loaded
    pipeline.run_once(limit=100)
    assert recorder.batches[1] == []


def test_a_crashing_consumer_is_logged_and_does_not_break_the_run(store, engine, sample_replay) -> None:
    recorder = Recorder()
    report = build(store, engine, sample_replay, consumers=[Exploding(), recorder]).run_once(limit=100)
    assert any("Exploding" in e for e in report.errors)
    assert recorder.batches and recorder.batches[0]  # later consumers still run


def test_jsonl_sink_writes_one_line_per_signal(store, engine, sample_replay, tmp_path) -> None:
    sink_path = tmp_path / "out" / "signals.jsonl"
    report = build(store, engine, sample_replay, consumers=[JsonlSignalSink(sink_path)]).run_once(limit=100)
    lines = sink_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == report.signals_loaded
    assert {"sentiment_score", "event_type", "impact_score", "confidence"} <= set(json.loads(lines[0]))


def test_stress_module_plugs_in_as_a_consumer(store, engine, sample_replay, config_dir) -> None:
    trigger = StressTrigger(
        Portfolio.from_yaml(config_dir / "portfolio.yaml"),
        ScenarioBook.from_yaml(config_dir / "scenarios.yaml"),
        StressEngine(),
        on_result=lambda r: store.save_stress_run(r.model_dump(mode="json")),
    )
    build(store, engine, sample_replay, consumers=[trigger]).run_once(limit=100)
    runs = store.list_stress_runs()
    events = {r["event_type"] for r in runs}
    assert {"Geopolitical", "Market Shock"} <= events
    assert all(r["pnl"] < 0 and r["value_after"] < r["value_before"] for r in runs)
    geo = next(r for r in runs if r["event_type"] == "Geopolitical")
    assert geo["trigger_signal_id"] and geo["impact_score"] > 7


def test_pipeline_accepts_any_source_implementing_the_protocol(store, engine) -> None:
    class Inline:
        name = "inline"

        def fetch(self, cursor, limit):
            return FetchResult(
                documents=[RawDocument(source="inline", external_id="1", body="Tesla shares rise as quarterly deliveries beat analyst estimates")],
                next_cursor="1",
            )

    report = EtlPipeline(sources={"inline": Inline()}, store=store, analyzer=engine).run_once()
    assert report.signals_loaded == 1
