"""Extension points. Every pluggable part of the system is one of these protocols.

To add a data source, sentiment model, event classifier, ... implement the protocol and
register it in `prism/bootstrap.py`; nothing else changes.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .contracts import (
    Document,
    EntityMention,
    EtlRunReport,
    EventResult,
    FetchResult,
    ImpactInput,
    ImpactResult,
    RawDocument,
    RejectedRecord,
    RiskSignal,
    SentimentResult,
)


# ------------------------------------------------------------------ ETL
@runtime_checkable
class Source(Protocol):
    """Extract: pull new raw documents. `cursor` is opaque, owned by the source (watermark/offset)."""

    name: str

    def fetch(self, cursor: str | None, limit: int) -> FetchResult: ...


class KeyValueState(Protocol):
    def get_state(self, key: str) -> str | None: ...

    def set_state(self, key: str, value: str) -> None: ...


class PipelineStore(KeyValueState, Protocol):
    """Load: where the ETL writes. Implemented by prism.storage.SqlStore."""

    def save_raw(self, docs: Sequence[RawDocument]) -> None: ...

    def known_hashes(self, limit: int = 5000) -> set[str]: ...

    def save_documents(self, docs: Sequence[Document]) -> int: ...

    def save_signals(self, signals: Sequence[RiskSignal]) -> list[RiskSignal]:
        """Idempotent upsert. Returns only the signals that were new."""
        ...

    def log_rejects(self, run_id: str, rejects: Sequence[RejectedRecord]) -> None: ...

    def log_run(self, report: EtlRunReport) -> None: ...


# ------------------------------------------------------------------ NLP risk engine
@runtime_checkable
class EntityLinker(Protocol):
    def link(self, text: str) -> list[EntityMention]: ...


@runtime_checkable
class SentimentModel(Protocol):
    name: str

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]: ...


@runtime_checkable
class EventClassifier(Protocol):
    name: str

    def classify(self, texts: Sequence[str]) -> list[EventResult]: ...


@runtime_checkable
class ImpactModel(Protocol):
    name: str

    def score(self, inp: ImpactInput) -> ImpactResult: ...


@runtime_checkable
class Analyzer(Protocol):
    """The Risk Engine as the ETL sees it: documents in, structured signals out."""

    def analyze(self, docs: Sequence[Document]) -> list[RiskSignal]: ...


# ------------------------------------------------------------------ downstream
@runtime_checkable
class SignalConsumer(Protocol):
    """Anything that reacts to freshly loaded signals (stress tester, rebalancer, file sink...)."""

    def on_signals(self, signals: Sequence[RiskSignal]) -> None: ...
