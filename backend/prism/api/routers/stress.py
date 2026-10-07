from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from prism.api.deps import get_container
from prism.bootstrap import Container
from prism.core.taxonomy import EventType
from prism.modules.stress import HistoricalOutcome, Portfolio, Scenario, Shock, StressResult

router = APIRouter(prefix="/api", tags=["stress test"])


class StressRequest(BaseModel):
    event_type: EventType
    impact_score: float | None = Field(None, ge=1, le=10)
    headline: str = Field("", description="Used to find the most similar past events (history source)")
    source: Literal["history", "matrix"] | None = Field(None, description="Default: history when the library is loaded")
    quantile: float | None = Field(None, gt=0, lt=1, description="History source: 0.10 = 1-in-10 bad outcome")
    shock: Shock | None = Field(None, description="Override with a custom shock (manual what-if)")
    persist: bool = True


@router.get("/portfolio", response_model=Portfolio)
def get_portfolio(c: Container = Depends(get_container)) -> Portfolio:
    return c.portfolio


@router.get("/scenarios", response_model=list[Scenario])
def list_scenarios(c: Container = Depends(get_container)) -> list[Scenario]:
    return c.scenarios.all()


@router.post("/stress-test", response_model=StressResult)
def run_stress_test(req: StressRequest, c: Container = Depends(get_container)) -> StressResult:
    """Run a stress test on demand, regardless of trigger thresholds (demo / what-if)."""
    scenario = c.scenarios.for_event(req.event_type)
    if scenario is None:
        raise HTTPException(404, f"no scenario defined for event type {req.event_type.value!r}")
    source = req.source or ("history" if c.history else "matrix")
    if req.shock is not None:
        result = c.stress_engine.run(c.portfolio, scenario, shock=req.shock, impact_score=req.impact_score)
    elif source == "history":
        if c.history is None:
            raise HTTPException(409, "history-calibrated scenarios are not enabled (SCENARIO_SOURCE / analog library)")
        history = c.history.build(req.headline or scenario.description or scenario.name, req.event_type, quantile=req.quantile)
        result = c.stress_engine.run(c.portfolio, scenario, impact_score=req.impact_score, history=history)
    else:
        result = c.stress_engine.run(c.portfolio, scenario, impact_score=req.impact_score)
    if req.persist:
        c.store.save_stress_run(result.model_dump(mode="json"))
    return result


@router.get("/stress-runs", response_model=list[StressResult])
def list_stress_runs(limit: int = 20, c: Container = Depends(get_container)) -> list[dict]:
    return c.store.list_stress_runs(limit)


@router.get("/analogs", response_model=list[HistoricalOutcome])
def find_analogs(
    text: str = Query(..., min_length=3),
    event_type: EventType | None = None,
    k: int = Query(5, ge=1, le=20),
    c: Container = Depends(get_container),
) -> list[HistoricalOutcome]:
    """The past events most similar to a headline, what markets did, and what each would cost today."""
    if c.analogs is None or c.history is None:
        raise HTTPException(409, "analog library not loaded")
    return [
        HistoricalOutcome(
            id=m.id, date=m.date, title=m.title, event_type=m.event_type, shock=m.shock,
            pnl=c.history.pnl(m.shock), similarity=m.similarity,
        )
        for m in c.analogs.nearest(text, event_type, k=k)
    ]


@router.get("/history/summary")
def history_summary(c: Container = Depends(get_container)) -> dict:
    """What markets actually did after each kind of event vs the hand-written matrix (evidence panel)."""
    if c.analogs is None:
        raise HTTPException(409, "analog library not loaded")
    h = c.analogs.horizon
    rows = []
    for event_type in EventType:
        group = [e.moves[h] for e in c.analogs.events if e.event_type == event_type]
        if not group:
            continue
        matrix = c.scenarios.for_event(event_type)
        rows.append(
            {
                "event_type": event_type.value,
                "events": len(group),
                "median_equity_pct": float(np.median([g.equity_pct for g in group])),
                "median_rate_bps": float(np.median([g.rate_bps for g in group])),
                "share_rate_down": float(np.mean([g.rate_bps < 0 for g in group])),
                "median_credit_spread_bps": float(np.median([g.credit_spread_bps for g in group])),
                "matrix_shock": matrix.shock.model_dump() if matrix else None,
            }
        )
    return {
        "events": len(c.analogs.events),
        "first": min(e.reaction_date for e in c.analogs.events).isoformat(),
        "last": max(e.reaction_date for e in c.analogs.events).isoformat(),
        "horizon": h,
        "scenario_source": "history" if c.history else "matrix",
        "quantile": c.history.quantile if c.history else None,
        "by_type": rows,
    }
