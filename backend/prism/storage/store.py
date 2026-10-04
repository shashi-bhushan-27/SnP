from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from prism.core.contracts import (
    Document,
    EtlRunReport,
    RawDocument,
    RejectedRecord,
    RiskSignal,
    utcnow,
)
from prism.core.ids import stable_id
from prism.core.taxonomy import EventType
from prism.storage.models import (
    Base,
    DocumentRow,
    EtlRejectRow,
    EtlRunRow,
    KvStateRow,
    RawDocumentRow,
    RiskSignalRow,
    StressRunRow,
)

_CHUNK = 500


def _naive(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


def _chunks(items: Sequence[Any], size: int = _CHUNK):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class SqlStore:
    """Implements PipelineStore plus the read side used by the API."""

    def __init__(self, url: str) -> None:
        kwargs: dict[str, Any] = {}
        is_sqlite = url.startswith("sqlite")
        if is_sqlite:
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool
            else:
                Path(url.split("///", 1)[-1]).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(url, **kwargs)
        if is_sqlite:

            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()

        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def _session(self) -> Session:
        return self._sessions()

    # ------------------------------------------------------------------ key/value state
    def get_state(self, key: str) -> str | None:
        with self._session() as s:
            row = s.get(KvStateRow, key)
            return row.value if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._session() as s, s.begin():
            row = s.get(KvStateRow, key)
            if row:
                row.value = value
            else:
                s.add(KvStateRow(key=key, value=value))

    # ------------------------------------------------------------------ bronze
    def save_raw(self, docs: Sequence[RawDocument]) -> None:
        rows: dict[str, RawDocumentRow] = {}
        for d in docs:
            key = d.external_id or d.url or json.dumps(d.payload, sort_keys=True, default=str)
            rid = stable_id(d.source, key)
            rows[rid] = RawDocumentRow(
                id=rid,
                source=d.source,
                external_id=(d.external_id or d.url or "")[:512] or None,
                fetched_at=_naive(d.fetched_at),
                payload=json.loads(json.dumps(d.payload, default=str)),
            )
        with self._session() as s, s.begin():
            for rid, row in rows.items():
                if s.get(RawDocumentRow, rid) is None:
                    s.add(row)

    # ------------------------------------------------------------------ silver
    def known_hashes(self, limit: int = 5000) -> set[str]:
        with self._session() as s:
            stmt = (
                select(DocumentRow.content_hash, DocumentRow.merged_hashes)
                .order_by(DocumentRow.created_at.desc())
                .limit(limit)
            )
            hashes: set[str] = set()
            for content_hash, merged in s.execute(stmt):
                hashes.add(content_hash)
                hashes.update(merged or [])
            return hashes

    def save_documents(self, docs: Sequence[Document]) -> int:
        unique = {d.id: d for d in docs}
        inserted = 0
        with self._session() as s, s.begin():
            for chunk in _chunks(list(unique)):
                existing = set(s.scalars(select(DocumentRow.id).where(DocumentRow.id.in_(chunk))))
                for doc_id in chunk:
                    if doc_id in existing:
                        continue
                    d = unique[doc_id]
                    s.add(
                        DocumentRow(
                            id=d.id,
                            source=d.source,
                            source_domain=d.source_domain,
                            url=d.url,
                            title=d.title,
                            text=d.text,
                            published_at=_naive(d.published_at),
                            language=d.language,
                            content_hash=d.content_hash,
                            corroboration=d.corroboration,
                            merged_hashes=list(d.merged_hashes),
                            created_at=_naive(utcnow()),
                        )
                    )
                    inserted += 1
        return inserted

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[Document]:
        with self._session() as s:
            rows = s.scalars(select(DocumentRow).order_by(DocumentRow.published_at.desc()).limit(limit).offset(offset))
            return [
                Document(
                    id=r.id,
                    source=r.source,
                    source_domain=r.source_domain,
                    url=r.url,
                    title=r.title,
                    text=r.text,
                    published_at=_aware(r.published_at),
                    language=r.language,
                    content_hash=r.content_hash,
                    corroboration=r.corroboration,
                    merged_hashes=list(r.merged_hashes or []),
                )
                for r in rows
            ]

    # ------------------------------------------------------------------ gold
    def save_signals(self, signals: Sequence[RiskSignal]) -> list[RiskSignal]:
        unique = {sig.id: sig for sig in signals}
        new: list[RiskSignal] = []
        with self._session() as s, s.begin():
            for chunk in _chunks(list(unique)):
                existing = set(s.scalars(select(RiskSignalRow.id).where(RiskSignalRow.id.in_(chunk))))
                for sig_id in chunk:
                    if sig_id in existing:
                        continue
                    sig = unique[sig_id]
                    s.add(
                        RiskSignalRow(
                            id=sig.id,
                            doc_id=sig.doc_id,
                            source=sig.source,
                            timestamp=_naive(sig.timestamp),
                            created_at=_naive(sig.created_at),
                            entity=sig.entity,
                            ticker=sig.ticker,
                            sector=sig.sector,
                            scope=sig.scope,
                            sentiment_score=sig.sentiment_score,
                            event_type=sig.event_type.value,
                            impact_score=sig.impact_score,
                            confidence=sig.confidence,
                            headline=sig.headline,
                            url=sig.url,
                            explanation=json.loads(json.dumps(sig.explanation, default=str)),
                        )
                    )
                    new.append(sig)
        return new

    @staticmethod
    def _to_signal(r: RiskSignalRow) -> RiskSignal:
        return RiskSignal(
            id=r.id,
            doc_id=r.doc_id,
            source=r.source,
            timestamp=_aware(r.timestamp),
            created_at=_aware(r.created_at),
            entity=r.entity,
            ticker=r.ticker,
            sector=r.sector,
            scope=r.scope,
            sentiment_score=r.sentiment_score,
            event_type=EventType(r.event_type),
            impact_score=r.impact_score,
            confidence=r.confidence,
            headline=r.headline,
            url=r.url,
            explanation=r.explanation or {},
        )

    def get_signal(self, signal_id: str) -> RiskSignal | None:
        with self._session() as s:
            row = s.get(RiskSignalRow, signal_id)
            return self._to_signal(row) if row else None

    def query_signals(
        self,
        *,
        ticker: str | None = None,
        event_type: EventType | None = None,
        source: str | None = None,
        min_impact: float | None = None,
        min_confidence: float | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RiskSignal]:
        stmt = select(RiskSignalRow)
        if ticker:
            stmt = stmt.where(RiskSignalRow.ticker == ticker.upper())
        if event_type:
            stmt = stmt.where(RiskSignalRow.event_type == event_type.value)
        if source:
            stmt = stmt.where(RiskSignalRow.source == source)
        if min_impact is not None:
            stmt = stmt.where(RiskSignalRow.impact_score >= min_impact)
        if min_confidence is not None:
            stmt = stmt.where(RiskSignalRow.confidence >= min_confidence)
        if since is not None:
            stmt = stmt.where(RiskSignalRow.timestamp >= _naive(since))
        stmt = stmt.order_by(RiskSignalRow.timestamp.desc(), RiskSignalRow.impact_score.desc()).limit(limit).offset(offset)
        with self._session() as s:
            return [self._to_signal(r) for r in s.scalars(stmt)]

    # ------------------------------------------------------------------ run bookkeeping
    def log_rejects(self, run_id: str, rejects: Sequence[RejectedRecord]) -> None:
        now = _naive(utcnow())
        with self._session() as s, s.begin():
            s.add_all(
                EtlRejectRow(run_id=run_id, source=r.source, reason=r.reason[:256], snippet=r.snippet, created_at=now)
                for r in rejects
            )

    def list_rejects(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.scalars(select(EtlRejectRow).order_by(EtlRejectRow.id.desc()).limit(limit))
            return [
                {"run_id": r.run_id, "source": r.source, "reason": r.reason, "snippet": r.snippet, "created_at": _aware(r.created_at)}
                for r in rows
            ]

    def log_run(self, report: EtlRunReport) -> None:
        with self._session() as s, s.begin():
            s.merge(
                EtlRunRow(
                    id=report.run_id,
                    started_at=_naive(report.started_at),
                    finished_at=_naive(report.finished_at) if report.finished_at else None,
                    report=report.model_dump(mode="json"),
                )
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._session() as s:
            return [r.report for r in s.scalars(select(EtlRunRow).order_by(EtlRunRow.started_at.desc()).limit(limit))]

    # ------------------------------------------------------------------ stress runs (plain dicts)
    def save_stress_run(self, result: Mapping[str, Any]) -> None:
        created = datetime.fromisoformat(str(result["created_at"]).replace("Z", "+00:00"))
        with self._session() as s, s.begin():
            s.merge(
                StressRunRow(
                    id=result["id"],
                    created_at=_naive(created),
                    scenario=result["scenario"],
                    event_type=str(result["event_type"]),
                    trigger_signal_id=result.get("trigger_signal_id"),
                    impact_score=result.get("impact_score"),
                    value_before=result["value_before"],
                    value_after=result["value_after"],
                    pnl=result["pnl"],
                    payload=dict(result),
                )
            )

    def list_stress_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._session() as s:
            rows = s.scalars(select(StressRunRow).order_by(StressRunRow.created_at.desc()).limit(limit))
            return [r.payload for r in rows]

    # ------------------------------------------------------------------ dashboard KPIs
    def stats(self, high_risk_threshold: float = 7.0, negative_sentiment: float = -0.2) -> dict[str, Any]:
        """KPIs. "High impact" ignores direction; "high risk" also requires negative-leaning sentiment
        (the same gate the stress trigger uses), so record earnings do not inflate the risk count."""
        with self._session() as s:
            documents = s.scalar(select(func.count()).select_from(DocumentRow)) or 0
            signals = s.scalar(select(func.count()).select_from(RiskSignalRow)) or 0
            high_impact_stmt = select(func.count()).select_from(RiskSignalRow).where(
                RiskSignalRow.impact_score >= high_risk_threshold
            )
            high_impact = s.scalar(high_impact_stmt) or 0
            high_risk = s.scalar(high_impact_stmt.where(RiskSignalRow.sentiment_score <= negative_sentiment)) or 0
            avg_impact = s.scalar(select(func.avg(RiskSignalRow.impact_score)))
            by_event = dict(
                s.execute(select(RiskSignalRow.event_type, func.count()).group_by(RiskSignalRow.event_type)).all()
            )
            by_source = dict(s.execute(select(RiskSignalRow.source, func.count()).group_by(RiskSignalRow.source)).all())
            last_run = s.scalar(select(EtlRunRow.report).order_by(EtlRunRow.started_at.desc()).limit(1))
        return {
            "documents": documents,
            "signals": signals,
            "high_impact_signals": high_impact,
            "high_risk_signals": high_risk,
            "high_risk_threshold": high_risk_threshold,
            "avg_impact": round(avg_impact, 2) if avg_impact is not None else None,
            "signals_by_event": by_event,
            "signals_by_source": by_source,
            "last_run": last_run,
        }
