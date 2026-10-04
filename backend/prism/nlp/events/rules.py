"""Keyword/regex event classifier: transparent, instant, high precision on explicit cues.

Each event type has strong cues (1.0 each) and weak cues (0.4 each). Scores are normalised into a
probability-like distribution; confidence grows with the amount of evidence and with how clearly
the top class beats the rest. No rule firing -> OTHER with `matched=False`, which lets the hybrid
classifier hand the decision to the embedding model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from prism.core.contracts import EventResult
from prism.core.taxonomy import EventType

STRONG, WEAK = 1.0, 0.4


class RuleEventClassifier:
    name = "rules"

    def __init__(self, rules: Mapping[EventType, Mapping[str, Sequence[str]]], min_score: float = 0.8) -> None:
        # min_score 0.8 = one strong cue or two weak ones; a lone weak word ("plunge" in a box-office
        # story) is not evidence of an event
        self.min_score = min_score
        # insertion order is the tie-break priority
        self._rules = [
            (
                event,
                [re.compile(p, re.IGNORECASE) for p in spec.get("strong", [])],
                [re.compile(p, re.IGNORECASE) for p in spec.get("weak", [])],
            )
            for event, spec in rules.items()
        ]

    @classmethod
    def from_yaml(cls, path: Path | str) -> RuleEventClassifier:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls({EventType.parse(key): spec for key, spec in raw.items()})

    def classify(self, texts: Sequence[str]) -> list[EventResult]:
        return [self._classify_one(text) for text in texts]

    def _classify_one(self, text: str) -> EventResult:
        scores: dict[EventType, float] = {}
        for event, strong, weak in self._rules:
            score = STRONG * sum(1 for p in strong if p.search(text)) + WEAK * sum(1 for p in weak if p.search(text))
            if score > 0:
                scores[event] = score

        if not scores or max(scores.values()) < self.min_score:
            return EventResult(
                event_type=EventType.OTHER,
                confidence=0.3,
                scores={EventType.OTHER.value: 1.0},
                method=self.name,
                matched=False,
            )

        top_event = max(scores, key=scores.__getitem__)  # first max wins -> priority order
        top, total = scores[top_event], sum(scores.values())
        confidence = min(0.95, 0.45 + 0.25 * top) * (0.6 + 0.4 * top / total)
        return EventResult(
            event_type=top_event,
            confidence=round(confidence, 4),
            scores={event.value: score / total for event, score in scores.items()},
            method=self.name,
        )
