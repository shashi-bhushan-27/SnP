"""History-calibrated stress scenarios (the default scenario source for Module B).

For a triggering event of type T:
    pool      = past events of type T (>= min_pool of them), otherwise all past events
    stress    = the REAL historical reaction at the chosen coverage quantile of this book's P&L over the pool
                (default q = 0.10: "a 1-in-10 bad outcome for this kind of event") - a real event, so the equity,
                rate and credit moves stay mutually consistent
    expected  = the pool's average reaction (the central case)
    analogs   = the most similar past events by meaning, with what markets did and what each would cost today

Back-test (evaluation/results/analog_backtest.md, 81 events 2008-2025, purged leave-one-out and time-respecting):
the q10 rule covered 85-89% of real outcomes with ~$0.4M average excess severity on the $60M book, where the
hand-written matrix covered 95% only by being ~$4.7M too severe on average - and it assumes yields rise in
crises, while after 100% of bankruptcies and 81% of geopolitical shocks in the library the 10-year yield fell.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from prism.core.taxonomy import EventType
from prism.modules.stress.analogs import AnalogLibrary
from prism.modules.stress.engine import StressEngine
from prism.modules.stress.portfolio import Portfolio
from prism.modules.stress.scenarios import HistoricalOutcome, HistoricalScenario, Scenario, Shock


class HistoricalScenarioBuilder:
    def __init__(
        self,
        library: AnalogLibrary,
        portfolio: Portfolio,
        engine: StressEngine | None = None,
        *,
        quantile: float = 0.10,
        min_pool: int = 5,
        n_analogs: int = 5,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self.library = library
        self.portfolio = portfolio
        self.engine = engine or StressEngine()
        self.quantile = quantile
        self.min_pool = min_pool
        self.n_analogs = n_analogs
        self._valuation = Scenario(event_type=EventType.OTHER, name="valuation", shock=Shock())

    def pnl(self, shock: Shock) -> float:
        return self.engine.run(self.portfolio, self._valuation, shock=shock).pnl

    def build(
        self,
        headline: str,
        event_type: EventType | None,
        *,
        quantile: float | None = None,
        exclude: Iterable[str] = (),
        before: date | None = None,
    ) -> HistoricalScenario:
        q = self.quantile if quantile is None else quantile
        h = self.library.horizon
        pool, same_type = self.library.pool(event_type, min_pool=self.min_pool, exclude=exclude, before=before)
        if not pool:
            raise ValueError("no historical events available")

        outcomes = sorted(
            (
                HistoricalOutcome(
                    id=e.id, date=e.reaction_date, title=e.title, event_type=e.event_type,
                    shock=e.moves[h], pnl=self.pnl(e.moves[h]),
                )
                for e in pool
            ),
            key=lambda o: o.pnl,
        )
        basis = outcomes[int(q * (len(outcomes) - 1))]  # "lower" quantile: a real event, never interpolated
        n = len(pool)
        expected = Shock(
            equity_pct=sum(e.moves[h].equity_pct for e in pool) / n,
            rate_bps=sum(e.moves[h].rate_bps for e in pool) / n,
            credit_spread_bps=sum(e.moves[h].credit_spread_bps for e in pool) / n,
        )
        matches = self.library.nearest(headline, event_type, k=self.n_analogs, exclude=exclude, before=before)
        analogs = [
            HistoricalOutcome(
                id=m.id, date=m.date, title=m.title, event_type=m.event_type,
                shock=m.shock, pnl=self.pnl(m.shock), similarity=m.similarity,
            )
            for m in matches
        ]
        label = event_type.value if same_type and event_type else "all event types"
        return HistoricalScenario(
            name=f"History-calibrated: {label}, 1-in-{round(1 / q)} outcome",
            event_type=event_type,
            horizon=h,
            quantile=q,
            shock=basis.shock,
            basis=basis,
            pool_size=n,
            pool_same_type=same_type,
            expected_shock=expected,
            expected_pnl=self.pnl(expected),
            distribution=outcomes,
            analogs=analogs,
        )
