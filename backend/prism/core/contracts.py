"""The single data vocabulary of the system.

Data moves through three shapes (a medallion layout):
    RawDocument  (bronze)  exactly what a source returned
    Document     (silver)  cleaned, de-duplicated, ready for NLP
    RiskSignal   (gold)    structured output consumed by downstream modules
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .taxonomy import EventType

Scope = Literal["company", "sector", "market"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- extract
class RawDocument(BaseModel):
    source: str
    external_id: str | None = None
    url: str | None = None
    title: str | None = None
    body: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    language: str | None = None
    publisher: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class FetchResult(BaseModel):
    documents: list[RawDocument] = Field(default_factory=list)
    next_cursor: str | None = None


# --------------------------------------------------------------------------- transform
class Document(BaseModel):
    id: str
    source: str
    source_domain: str | None = None
    url: str | None = None
    title: str
    text: str
    published_at: datetime
    language: str = "en"
    content_hash: str
    corroboration: int = Field(default=1, ge=1)
    merged_hashes: list[str] = Field(default_factory=list)  # near-duplicates folded into this document


class EntityMention(BaseModel):
    name: str
    ticker: str | None = None
    sector: str | None = None
    scope: Scope = "company"
    matched_text: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SentimentResult(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)  # P(positive) - P(negative)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    adjustment: str | None = None  # set when a rule layer changed the model's direction (e.g. "macro-bearish")


class EventResult(BaseModel):
    event_type: EventType
    confidence: float = Field(ge=0.0, le=1.0)
    scores: dict[str, float] = Field(default_factory=dict)  # EventType.value -> probability
    method: str = ""
    matched: bool = True  # False when the method had no evidence (e.g. no rule fired)


class ImpactInput(BaseModel):
    event_type: EventType
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    scope: Scope = "company"
    tracked: bool = True  # company in our universe (has a ticker); untracked companies weigh less
    source: str = ""
    source_domain: str | None = None
    corroboration: int = Field(default=1, ge=1)


class ImpactResult(BaseModel):
    score: float = Field(ge=1.0, le=10.0)
    features: dict[str, float] = Field(default_factory=dict)  # each in [0, 1]
    components: dict[str, float] = Field(default_factory=dict)  # points contributed on top of 1.0


# --------------------------------------------------------------------------- load / serve
class RiskSignal(BaseModel):
    id: str
    doc_id: str
    source: str
    timestamp: datetime  # when the news was published
    created_at: datetime = Field(default_factory=utcnow)
    entity: str | None = None
    ticker: str | None = None
    sector: str | None = None
    scope: Scope = "company"
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    event_type: EventType
    impact_score: float = Field(ge=1.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)
    corroboration: int = Field(default=1, ge=1)  # distinct outlets reporting the story
    source_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    headline: str
    url: str | None = None
    explanation: dict[str, Any] = Field(default_factory=dict)


class RejectedRecord(BaseModel):
    source: str
    reason: str
    snippet: str = ""


class SourceStats(BaseModel):
    fetched: int = 0
    rejected: int = 0
    error: str | None = None


class EtlRunReport(BaseModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    sources: dict[str, SourceStats] = Field(default_factory=dict)
    normalized: int = 0
    duplicates: int = 0
    documents_loaded: int = 0
    signals_loaded: int = 0
    skipped_irrelevant: int = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
