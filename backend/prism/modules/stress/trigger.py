from __future__ import annotations

from typing import Callable, Sequence

from prism.core.contracts import RiskSignal
from prism.core.taxonomy import EventType
from prism.modules.stress.calibrated import HistoricalScenarioBuilder
from prism.modules.stress.engine import StressEngine, StressResult
from prism.modules.stress.portfolio import Portfolio
from prism.modules.stress.scenarios import Scenario, ScenarioBook


class StressTrigger:
    """SignalConsumer: when a loaded signal satisfies a scenario's trigger, run the stress test.

    Within one batch only the highest-impact signal per event type fires, so a burst of articles
    about the same event produces one stress run, not dozens. Results go to `on_result`
    (persistence is the caller's concern, keeping this module storage-free).

    With a `history` builder the shock is calibrated on past events of the same kind (the default);
    without one, the hand-written shock from scenarios.yaml is used. The trigger rules apply either way.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        scenarios: ScenarioBook,
        engine: StressEngine,
        on_result: Callable[[StressResult], None],
        history: HistoricalScenarioBuilder | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.scenarios = scenarios
        self.engine = engine
        self.on_result = on_result
        self.history = history

    def on_signals(self, signals: Sequence[RiskSignal]) -> None:
        strongest: dict[EventType, tuple[RiskSignal, Scenario]] = {}
        for signal in signals:
            scenario = self.scenarios.match(signal)
            if scenario is None:
                continue
            current = strongest.get(signal.event_type)
            if current is None or signal.impact_score > current[0].impact_score:
                strongest[signal.event_type] = (signal, scenario)

        for signal, scenario in strongest.values():
            history = self.history.build(signal.headline, signal.event_type) if self.history else None
            self.on_result(
                self.engine.run(
                    self.portfolio,
                    scenario,
                    impact_score=signal.impact_score,
                    trigger_signal_id=signal.id,
                    history=history,
                )
            )
