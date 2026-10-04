from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from prism.api.deps import get_container
from prism.bootstrap import Container
from prism.core.taxonomy import EventType
from prism.modules.stress import Portfolio, Scenario, Shock, StressResult

router = APIRouter(prefix="/api", tags=["stress test"])


class StressRequest(BaseModel):
    event_type: EventType
    impact_score: float | None = Field(None, ge=1, le=10)
    shock: Shock | None = Field(None, description="Override the scenario's default shock")
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
    result = c.stress_engine.run(c.portfolio, scenario, shock=req.shock, impact_score=req.impact_score)
    if req.persist:
        c.store.save_stress_run(result.model_dump(mode="json"))
    return result


@router.get("/stress-runs", response_model=list[StressResult])
def list_stress_runs(limit: int = 20, c: Container = Depends(get_container)) -> list[dict]:
    return c.store.list_stress_runs(limit)
