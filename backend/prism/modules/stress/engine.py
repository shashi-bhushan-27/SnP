"""Deterministic stress valuation. For every asset (V = market value):

    dV =  V * beta * equity_pct                                  equity prices
        + V * (-duration * dy + 0.5 * convexity * dy^2)          interest rates (dy = rate_bps / 1e4)
        - V * spread_duration * ds                               credit spreads (ds = spread_bps / 1e4)
        + dv01 * rate_bps                                        derivatives (hedges show up as gains)

A first-order (+convexity) sensitivity model: transparent and explainable, not a pricing engine.
Values are floored at zero (no negative market value from a single shock).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from prism.core.contracts import utcnow
from prism.core.ids import stable_id
from prism.core.taxonomy import EventType
from prism.modules.stress.portfolio import Asset, Portfolio
from prism.modules.stress.scenarios import Scenario, Shock


class AssetImpact(BaseModel):
    asset_id: str
    name: str
    asset_type: str
    value_before: float
    value_after: float
    pnl: float
    pnl_pct: float


class StressResult(BaseModel):
    id: str
    created_at: datetime
    scenario: str
    event_type: EventType
    trigger_signal_id: str | None = None
    impact_score: float | None = None
    shock: Shock
    value_before: float
    value_after: float
    pnl: float
    pnl_pct: float
    by_asset: list[AssetImpact]
    by_type: dict[str, float]


def asset_pnl(asset: Asset, shock: Shock) -> float:
    dy = shock.rate_bps / 1e4
    ds = shock.credit_spread_bps / 1e4
    v = asset.value
    pnl = v * asset.effective_beta * shock.equity_pct
    pnl += v * (-asset.duration * dy + 0.5 * asset.convexity * dy**2)
    pnl -= v * asset.spread_duration * ds
    pnl += asset.dv01 * shock.rate_bps
    return max(pnl, -v)


class StressEngine:
    def run(
        self,
        portfolio: Portfolio,
        scenario: Scenario,
        *,
        shock: Shock | None = None,
        impact_score: float | None = None,
        trigger_signal_id: str | None = None,
    ) -> StressResult:
        applied = shock or scenario.shock
        impacts: list[AssetImpact] = []
        by_type: dict[str, float] = {}
        for asset in portfolio.assets:
            pnl = asset_pnl(asset, applied)
            impacts.append(
                AssetImpact(
                    asset_id=asset.id,
                    name=asset.name,
                    asset_type=asset.asset_type,
                    value_before=asset.value,
                    value_after=asset.value + pnl,
                    pnl=pnl,
                    pnl_pct=pnl / asset.value if asset.value else 0.0,
                )
            )
            by_type[asset.asset_type] = by_type.get(asset.asset_type, 0.0) + pnl

        before = portfolio.total_value
        pnl_total = sum(i.pnl for i in impacts)
        created = utcnow()
        return StressResult(
            id=stable_id(scenario.name, trigger_signal_id or "manual", created.isoformat()),
            created_at=created,
            scenario=scenario.name,
            event_type=scenario.event_type,
            trigger_signal_id=trigger_signal_id,
            impact_score=impact_score,
            shock=applied,
            value_before=before,
            value_after=before + pnl_total,
            pnl=pnl_total,
            pnl_pct=pnl_total / before if before else 0.0,
            by_asset=impacts,
            by_type=by_type,
        )
