from __future__ import annotations

from fastapi import APIRouter, Depends

from prism import __version__
from prism.api.deps import get_container
from prism.bootstrap import Container

router = APIRouter(tags=["health"])


@router.get("/health")
def health(c: Container = Depends(get_container)) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "sentiment_model": c.engine.sentiment.name,
        "event_classifier": c.engine.events.name,
        "impact_model": c.engine.impact.name,
        "sources": list(c.pipeline.sources),
        "scheduler_running": c.scheduler is not None,
    }
