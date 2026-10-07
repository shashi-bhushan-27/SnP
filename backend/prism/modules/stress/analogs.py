"""Historical-analog scenario engine.

Instead of a hand-written shock per event type, a headline is matched (by meaning) to past market-moving
events, and the portfolio is shocked with what markets actually did after them:

    headline --encode--> nearest past events (cosine similarity, small bonus for the same event type)
             --softmax weights over the top k--> weighted realised move of equities, 10y yield, Baa spread

Why: the realised history disagrees with intuition-built matrices in sign as well as size (after Lehman,
SVB, Brexit and most geopolitical shocks the 10-year yield FELL - flight to safety - while inflation
surprises pushed it UP). Matching on the content of the news distinguishes those cases; an event-type
label cannot. Every scenario carries its analogs, so the stress test explains itself ("looks like SVB 2023").

Fallbacks: if no past event is similar enough, use the average reaction to past events of the same type,
then the average over all events. `exclude` / `before` let the back-test hold events out honestly.

The encoder is injected (TextEncoder protocol), so this module stays independent of the NLP package.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from pydantic import BaseModel, Field

from prism.core.interfaces import TextEncoder
from prism.core.taxonomy import EventType
from prism.modules.stress.scenarios import Shock

Horizon = Literal["1d", "5d", "trough"]
Method = Literal["analog", "type_mean", "global_mean"]


class AnalogEvent(BaseModel):
    id: str
    date: date
    reaction_date: date
    title: str
    event_type: EventType
    moves: dict[str, Shock]  # horizon -> realised moves


class AnalogMatch(BaseModel):
    id: str
    date: date
    title: str
    event_type: EventType
    similarity: float
    weight: float
    shock: Shock


class AnalogScenario(BaseModel):
    shock: Shock
    horizon: str
    method: Method
    max_similarity: float = 0.0
    matches: list[AnalogMatch] = Field(default_factory=list)


def _mean(shocks: list[Shock]) -> Shock:
    return Shock(
        equity_pct=float(np.mean([s.equity_pct for s in shocks])),
        rate_bps=float(np.mean([s.rate_bps for s in shocks])),
        credit_spread_bps=float(np.mean([s.credit_spread_bps for s in shocks])),
    )


class AnalogLibrary:
    def __init__(
        self,
        events: Iterable[AnalogEvent],
        encoder: TextEncoder,
        *,
        horizon: Horizon = "trough",
        k: int = 5,
        temperature: float = 0.05,
        same_type_bonus: float = 0.05,
        min_similarity: float = 0.55,
    ) -> None:
        self.events = list(events)
        self.encoder = encoder
        self.horizon = horizon
        self.k = k
        self.temperature = temperature
        self.same_type_bonus = same_type_bonus
        self.min_similarity = min_similarity
        self._matrix: np.ndarray | None = None

    @classmethod
    def from_json(cls, path: Path | str, encoder: TextEncoder, **kwargs) -> AnalogLibrary:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        events = [
            AnalogEvent(
                id=e["id"],
                date=e["date"],
                reaction_date=e["reaction_date"],
                title=e["title"],
                event_type=EventType.parse(e["event_type"]),
                moves={h: Shock(**{k: v for k, v in m.items() if k != "days"}) for h, m in e["moves"].items()},
            )
            for e in raw["events"]
        ]
        return cls(events, encoder, **kwargs)

    # ------------------------------------------------------------------ retrieval
    def _embeddings(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.asarray(self.encoder.encode([e.title for e in self.events], normalize_embeddings=True))
        return self._matrix

    def similarities(self, text: str) -> np.ndarray:
        query = np.asarray(self.encoder.encode([text], normalize_embeddings=True))[0]
        return self._embeddings() @ query

    def _candidates(self, exclude: set[str], before: date | None) -> list[int]:
        return [
            i for i, e in enumerate(self.events) if e.id not in exclude and (before is None or e.reaction_date < before)
        ]

    def pool(
        self, event_type: EventType | None, *, min_pool: int = 5, exclude: Iterable[str] = (), before: date | None = None
    ) -> tuple[list[AnalogEvent], bool]:
        """Past events of the same type (if there are at least `min_pool`), else all past events.
        Returns (events, used_same_type)."""
        candidates = [self.events[i] for i in self._candidates(set(exclude), before)]
        same = [e for e in candidates if event_type is not None and e.event_type == event_type]
        return (same, True) if len(same) >= min_pool else (candidates, False)

    def type_mean(self, event_type: EventType | None, *, exclude: set[str] = frozenset(), before: date | None = None) -> Shock | None:
        shocks = [
            self.events[i].moves[self.horizon]
            for i in self._candidates(set(exclude), before)
            if event_type is not None and self.events[i].event_type == event_type
        ]
        return _mean(shocks) if shocks else None

    def global_mean(self, *, exclude: set[str] = frozenset(), before: date | None = None) -> Shock | None:
        shocks = [self.events[i].moves[self.horizon] for i in self._candidates(set(exclude), before)]
        return _mean(shocks) if shocks else None

    def nearest(
        self,
        text: str,
        event_type: EventType | None = None,
        *,
        k: int | None = None,
        exclude: Iterable[str] = (),
        before: date | None = None,
    ) -> list[AnalogMatch]:
        """The k most similar past events (similarity + same-type bonus), softmax-weighted; always returns
        up to k matches, whatever their similarity - use for explanation."""
        candidates = self._candidates(set(exclude), before)
        if not candidates:
            return []
        sims = self.similarities(text)[candidates]
        bonus = np.array([self.same_type_bonus if self.events[i].event_type == event_type else 0.0 for i in candidates])
        ranked = np.argsort(-(sims + bonus))[: k or self.k]
        scores = (sims[ranked] + bonus[ranked]) / self.temperature
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        matches = []
        for idx, weight in zip(ranked, weights):
            event = self.events[candidates[idx]]
            matches.append(
                AnalogMatch(
                    id=event.id,
                    date=event.reaction_date,
                    title=event.title,
                    event_type=event.event_type,
                    similarity=round(float(sims[idx]), 4),
                    weight=round(float(weight), 4),
                    shock=event.moves[self.horizon],
                )
            )
        return matches

    def scenario(
        self,
        text: str,
        event_type: EventType | None = None,
        *,
        exclude: Iterable[str] = (),
        before: date | None = None,
    ) -> AnalogScenario:
        """Similarity-weighted realised reaction of the nearest past events; falls back to the same-type
        average, then the overall average, when nothing in history is similar enough."""
        exclude = set(exclude)
        matches = self.nearest(text, event_type, exclude=exclude, before=before)
        if not matches:
            raise ValueError("analog library has no eligible events")
        max_similarity = max(m.similarity for m in matches)
        if max_similarity < self.min_similarity:
            fallback = self.type_mean(event_type, exclude=exclude, before=before)
            method: Method = "type_mean"
            if fallback is None:
                fallback, method = self.global_mean(exclude=exclude, before=before), "global_mean"
            return AnalogScenario(shock=fallback, horizon=self.horizon, method=method, max_similarity=max_similarity)
        shock = Shock(
            equity_pct=sum(m.weight * m.shock.equity_pct for m in matches),
            rate_bps=sum(m.weight * m.shock.rate_bps for m in matches),
            credit_spread_bps=sum(m.weight * m.shock.credit_spread_bps for m in matches),
        )
        return AnalogScenario(shock=shock, horizon=self.horizon, method="analog", max_similarity=max_similarity, matches=matches)
