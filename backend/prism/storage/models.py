"""Tables, laid out as medallion layers: raw_documents (bronze) -> documents (silver) -> risk_signals (gold),
plus run bookkeeping (etl_runs, etl_rejects, kv_state) and stress_runs. All datetimes are naive UTC."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RawDocumentRow(Base):
    __tablename__ = "raw_documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    external_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)
    payload: Mapped[dict] = mapped_column(JSON)


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    source_domain: Mapped[str | None] = mapped_column(String(256), nullable=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    language: Mapped[str] = mapped_column(String(8))
    content_hash: Mapped[str] = mapped_column(String(40), index=True)
    corroboration: Mapped[int] = mapped_column(Integer, default=1)
    merged_hashes: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class RiskSignalRow(Base):
    __tablename__ = "risk_signals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    entity: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    sector: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scope: Mapped[str] = mapped_column(String(16))
    sentiment_score: Mapped[float] = mapped_column(Float)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    impact_score: Mapped[float] = mapped_column(Float, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    corroboration: Mapped[int] = mapped_column(Integer, default=1)
    source_reliability: Mapped[float] = mapped_column(Float, default=0.5)
    headline: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    explanation: Mapped[dict] = mapped_column(JSON)


class EtlRunRow(Base):
    __tablename__ = "etl_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    report: Mapped[dict] = mapped_column(JSON)


class EtlRejectRow(Base):
    __tablename__ = "etl_rejects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(256))
    snippet: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)


class KvStateRow(Base):
    __tablename__ = "kv_state"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class StressRunRow(Base):
    __tablename__ = "stress_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    scenario: Mapped[str] = mapped_column(String(128))
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    trigger_signal_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    impact_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_before: Mapped[float] = mapped_column(Float)
    value_after: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict] = mapped_column(JSON)
