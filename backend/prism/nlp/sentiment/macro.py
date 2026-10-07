"""Market-direction layer on top of any sentiment model.

FinBERT (and most finance sentiment models) reads macro headlines literally: "US inflation hits a 40-year high,
beating forecasts" scores +0.90 because "high" and "beating" sound positive, and "Fed signals further rate
hikes" scores neutral. For markets both are bad news. This wrapper applies a few transparent direction rules
(config/macro_direction.yaml) after the model: when a bearish (bullish) pattern matches, the score becomes
negative (positive), keeping the model's magnitude with a floor of `min_magnitude`. Texts that match both or
neither are left alone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import yaml

from prism.core.contracts import SentimentResult
from prism.core.interfaces import SentimentModel


class MacroDirectionSentiment:
    def __init__(
        self,
        base: SentimentModel,
        bearish: Sequence[str],
        bullish: Sequence[str],
        min_magnitude: float = 0.5,
    ) -> None:
        self.base = base
        self.name = f"{base.name}+macro"
        self._bearish = [re.compile(p, re.IGNORECASE) for p in bearish]
        self._bullish = [re.compile(p, re.IGNORECASE) for p in bullish]
        self.min_magnitude = min_magnitude

    @classmethod
    def from_yaml(cls, base: SentimentModel, path: Path | str, **kwargs) -> MacroDirectionSentiment:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(base, raw.get("bearish", []), raw.get("bullish", []), **kwargs)

    def direction(self, text: str) -> int:
        bear = any(p.search(text) for p in self._bearish)
        bull = any(p.search(text) for p in self._bullish)
        return -1 if bear and not bull else 1 if bull and not bear else 0

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        results = self.base.predict(texts)
        out = []
        for text, result in zip(texts, results):
            sign = self.direction(text)
            if sign == 0 or (result.score * sign) >= self.min_magnitude:
                out.append(result)
                continue
            magnitude = max(abs(result.score), self.min_magnitude)
            out.append(
                result.model_copy(
                    update={"score": round(sign * magnitude, 4), "adjustment": "macro-bearish" if sign < 0 else "macro-bullish"}
                )
            )
        return out
