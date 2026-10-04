"""The ETL orchestrator.

    extract    each source, incrementally from its cursor; raw payloads are saved first (bronze)
    transform  normalize -> language filter -> de-duplicate (cross-source) -> NLP analyze
    load       documents (silver) + signals (gold), idempotent upserts; cursors advance only now
    notify     hand newly loaded signals to downstream consumers

Failure model: a failing source is skipped, a poison document is rejected, a failing consumer is
logged - none of them stop the run. Loads are idempotent, so a crash mid-run is simply re-run.
"""

from __future__ import annotations

import logging
import uuid
from time import perf_counter
from typing import Mapping, Sequence

from prism.core.contracts import (
    Document,
    EtlRunReport,
    FetchResult,
    RejectedRecord,
    RiskSignal,
    SourceStats,
    utcnow,
)
from prism.core.errors import SourceUnavailable
from prism.core.ids import stable_id
from prism.core.interfaces import Analyzer, PipelineStore, SignalConsumer, Source
from prism.etl.transform.dedupe import Deduper
from prism.etl.transform.normalize import normalize_batch

log = logging.getLogger(__name__)


class EtlPipeline:
    def __init__(
        self,
        *,
        sources: Mapping[str, Source],
        store: PipelineStore,
        analyzer: Analyzer,
        consumers: Sequence[SignalConsumer] = (),
        fetch_limit: int = 50,
        near_duplicate_threshold: float = 0.75,
    ) -> None:
        self.sources = dict(sources)
        self.store = store
        self.analyzer = analyzer
        self.consumers = list(consumers)
        self.fetch_limit = fetch_limit
        self.near_duplicate_threshold = near_duplicate_threshold

    def run_once(self, source_names: Sequence[str] | None = None, limit: int | None = None) -> EtlRunReport:
        report = EtlRunReport(run_id=stable_id(uuid.uuid4().hex), started_at=utcnow())
        started = perf_counter()
        names = list(source_names) if source_names else list(self.sources)

        # ---------------------------------------------------------------- extract
        t = perf_counter()
        fetched: dict[str, FetchResult] = {}
        for name in names:
            stats = report.sources.setdefault(name, SourceStats())
            source = self.sources.get(name)
            if source is None:
                stats.error = "unknown source"
                report.errors.append(f"{name}: unknown source")
                continue
            try:
                result = source.fetch(cursor=self.store.get_state(f"cursor:{name}"), limit=limit or self.fetch_limit)
            except SourceUnavailable as exc:
                log.warning("source %s unavailable: %s", name, exc)
                stats.error = str(exc)
                report.errors.append(f"{name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one bad source must not stop the run
                log.exception("source %s crashed", name)
                stats.error = f"unexpected error: {exc!r}"
                report.errors.append(f"{name}: {exc!r}")
                continue
            stats.fetched = len(result.documents)
            if result.documents:
                self.store.save_raw(result.documents)
            fetched[name] = result
        report.stage_seconds["extract"] = perf_counter() - t

        # ---------------------------------------------------------------- transform
        t = perf_counter()
        docs: list[Document] = []
        rejects: list[RejectedRecord] = []
        for name, result in fetched.items():
            batch_docs, batch_rejects = normalize_batch(result.documents)
            docs.extend(batch_docs)
            rejects.extend(batch_rejects)
            report.sources[name].rejected = len(batch_rejects)
        report.normalized = len(docs)

        deduped = Deduper(self.store.known_hashes(), self.near_duplicate_threshold).run(docs)
        docs = deduped.documents
        report.duplicates = deduped.duplicates
        report.stage_seconds["transform"] = perf_counter() - t

        # ---------------------------------------------------------------- analyze
        t = perf_counter()
        signals = self._analyze(docs, rejects)
        report.skipped_irrelevant = len(docs) - len({s.doc_id for s in signals})
        report.stage_seconds["analyze"] = perf_counter() - t

        # ---------------------------------------------------------------- load
        t = perf_counter()
        report.documents_loaded = self.store.save_documents(docs)
        new_signals = self.store.save_signals(signals)
        report.signals_loaded = len(new_signals)
        if rejects:
            self.store.log_rejects(report.run_id, rejects)
        for name, result in fetched.items():
            if result.next_cursor is not None:
                self.store.set_state(f"cursor:{name}", result.next_cursor)
        report.stage_seconds["load"] = perf_counter() - t

        # ---------------------------------------------------------------- notify
        t = perf_counter()
        for consumer in self.consumers:
            try:
                consumer.on_signals(new_signals)
            except Exception as exc:  # noqa: BLE001
                log.exception("consumer %s failed", type(consumer).__name__)
                report.errors.append(f"consumer {type(consumer).__name__}: {exc!r}")
        report.stage_seconds["notify"] = perf_counter() - t

        report.finished_at = utcnow()
        report.duration_seconds = perf_counter() - started
        self.store.log_run(report)
        return report

    def _analyze(self, docs: list[Document], rejects: list[RejectedRecord]) -> list[RiskSignal]:
        if not docs:
            return []
        try:
            return self.analyzer.analyze(docs)
        except Exception:  # noqa: BLE001
            log.exception("batch analysis failed; retrying document by document")
        signals: list[RiskSignal] = []
        for doc in docs:
            try:
                signals.extend(self.analyzer.analyze([doc]))
            except Exception as exc:  # noqa: BLE001
                rejects.append(RejectedRecord(source=doc.source, reason=f"analysis_error: {exc!r}", snippet=doc.title[:120]))
        return signals
