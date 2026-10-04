from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from prism.api.deps import get_container
from prism.bootstrap import Container
from prism.core.contracts import RawDocument, RejectedRecord, RiskSignal
from prism.etl.transform import normalize_batch

router = APIRouter(prefix="/api", tags=["risk engine"])


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=50)
    source: str = "adhoc"
    persist: bool = False


class AnalyzeResponse(BaseModel):
    signals: list[RiskSignal]
    rejected: list[RejectedRecord]


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, c: Container = Depends(get_container)) -> AnalyzeResponse:
    """Run the Risk Engine on arbitrary text (the engine as a standalone service)."""
    docs, rejected = normalize_batch(RawDocument(source=req.source, body=text) for text in req.texts)
    signals = c.engine.analyze(docs)
    if req.persist:
        c.store.save_documents(docs)
        c.store.save_signals(signals)
    return AnalyzeResponse(signals=signals, rejected=rejected)
