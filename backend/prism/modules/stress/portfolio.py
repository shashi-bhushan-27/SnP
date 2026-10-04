from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

AssetType = Literal["loan", "bond", "equity", "derivative"]


class Asset(BaseModel):
    """One position. All sensitivities are optional; unset means "not exposed to that factor"."""

    id: str
    name: str
    asset_type: AssetType
    sector: str | None = None
    value: float  # market value, USD
    beta: float | None = None  # equity beta (defaults to 1.0 for equities, 0 otherwise)
    duration: float = 0.0  # modified duration, years
    convexity: float = 0.0
    spread_duration: float = 0.0  # sensitivity to credit-spread widening
    dv01: float = 0.0  # USD per +1bp rate move (derivatives, signed)

    @property
    def effective_beta(self) -> float:
        if self.beta is not None:
            return self.beta
        return 1.0 if self.asset_type == "equity" else 0.0


class Portfolio(BaseModel):
    name: str
    currency: str = "USD"
    assets: list[Asset] = Field(min_length=1)

    @property
    def total_value(self) -> float:
        return sum(a.value for a in self.assets)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Portfolio:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
