from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from pydantic import BaseModel

from prism.core.contracts import RiskSignal
from prism.core.taxonomy import EventType


class Shock(BaseModel):
    equity_pct: float = 0.0  # -0.10 = equity prices -10%
    rate_bps: float = 0.0  # +200 = rates +2.00 percentage points
    credit_spread_bps: float = 0.0  # +150 = spreads +1.50 percentage points


class Trigger(BaseModel):
    min_impact: float = 7.0
    min_confidence: float = 0.5
    max_sentiment: float = 0.0  # only negative-leaning news stresses the book


class Scenario(BaseModel):
    event_type: EventType
    name: str
    description: str = ""
    trigger: Trigger = Trigger()
    shock: Shock


class ScenarioBook:
    """Event type -> scenario (one per type). Decides whether a signal fires a stress test."""

    def __init__(self, scenarios: Iterable[Scenario]) -> None:
        self._by_event = {s.event_type: s for s in scenarios}

    @classmethod
    def from_yaml(cls, path: Path | str) -> ScenarioBook:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(Scenario.model_validate(item) for item in raw["scenarios"])

    def all(self) -> list[Scenario]:
        return list(self._by_event.values())

    def for_event(self, event_type: EventType) -> Scenario | None:
        return self._by_event.get(event_type)

    def match(self, signal: RiskSignal) -> Scenario | None:
        scenario = self.for_event(signal.event_type)
        if scenario is None:
            return None
        t = scenario.trigger
        fires = (
            signal.impact_score > t.min_impact
            and signal.confidence >= t.min_confidence
            and signal.sentiment_score <= t.max_sentiment
        )
        return scenario if fires else None
