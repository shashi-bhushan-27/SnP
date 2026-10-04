from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from prism.api.deps import get_container
from prism.bootstrap import Container
from prism.core.contracts import EtlRunReport

router = APIRouter(prefix="/api/pipeline", tags=["etl"])


class RunRequest(BaseModel):
    sources: list[str] | None = Field(None, description="Subset of enabled sources; default all")
    limit: int | None = Field(None, ge=1, le=250, description="Max documents per source")


@router.post("/run", response_model=EtlRunReport)
def run_pipeline(req: RunRequest | None = None, c: Container = Depends(get_container)) -> EtlRunReport:
    """Trigger one ETL run now (blocking). Live polling is handled by the scheduler."""
    req = req or RunRequest()
    return c.pipeline.run_once(req.sources, req.limit)


@router.get("/runs")
def list_runs(limit: int = 20, c: Container = Depends(get_container)) -> list[dict]:
    return c.store.list_runs(limit)


@router.get("/rejects")
def list_rejects(limit: int = 50, c: Container = Depends(get_container)) -> list[dict]:
    """Records the pipeline refused (too short, non-English, analysis errors) and why."""
    return c.store.list_rejects(limit)
