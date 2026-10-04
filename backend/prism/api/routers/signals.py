from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from prism.api.deps import get_container
from prism.bootstrap import Container
from prism.core.contracts import Document, RiskSignal
from prism.core.taxonomy import EventType

router = APIRouter(prefix="/api", tags=["risk signals"])


@router.get("/risk-signals", response_model=list[RiskSignal])
def list_signals(
    symbol: str | None = Query(None, description="Ticker, e.g. TSLA"),
    event_type: EventType | None = None,
    source: str | None = None,
    min_impact: float | None = Query(None, ge=1, le=10),
    min_confidence: float | None = Query(None, ge=0, le=1),
    since: datetime | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    c: Container = Depends(get_container),
) -> list[RiskSignal]:
    return c.store.query_signals(
        ticker=symbol,
        event_type=event_type,
        source=source,
        min_impact=min_impact,
        min_confidence=min_confidence,
        since=since,
        limit=limit,
        offset=offset,
    )


@router.get("/risk-signals/latest", response_model=RiskSignal)
def latest_signal(symbol: str | None = None, c: Container = Depends(get_container)) -> RiskSignal:
    found = c.store.query_signals(ticker=symbol, limit=1)
    if not found:
        raise HTTPException(404, "no signals yet" + (f" for {symbol}" if symbol else ""))
    return found[0]


@router.get("/risk-signals/{signal_id}", response_model=RiskSignal)
def get_signal(signal_id: str, c: Container = Depends(get_container)) -> RiskSignal:
    signal = c.store.get_signal(signal_id)
    if signal is None:
        raise HTTPException(404, "signal not found")
    return signal


@router.get("/news", response_model=list[Document])
def list_news(
    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), c: Container = Depends(get_container)
) -> list[Document]:
    return c.store.list_documents(limit=limit, offset=offset)


@router.get("/stats")
def stats(c: Container = Depends(get_container)) -> dict:
    return c.store.stats()
