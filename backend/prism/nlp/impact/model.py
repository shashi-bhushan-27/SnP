"""Explainable impact score on a 1-10 scale.

    impact = 1 + 9 * sum(weight_i * feature_i),   every feature in [0, 1]

Features: event severity prior, |sentiment|, exposure (company/sector/market), source reliability,
corroboration (distinct outlets reporting it). Every signal carries the per-feature contribution in
points, so "why 8.7?" has an answer. Weights are priors in config/impact.yaml; the plan is to
calibrate them against realised price reaction (event study) once data is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from prism.core.contracts import ImpactInput, ImpactResult
from prism.core.taxonomy import EventType

FEATURES = ("severity", "sentiment_magnitude", "exposure", "source_reliability", "corroboration")


class WeightedImpactModel:
    name = "weighted-v1"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.weights = {name: float(config["weights"][name]) for name in FEATURES}
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"impact weights must sum to 1.0, got {total:.3f}")
        self.severity = {EventType.parse(k): float(v) for k, v in config["severity"].items()}
        self.exposure = {k: float(v) for k, v in config["exposure"].items()}
        reliability = config["source_reliability"]
        self._default_reliability = float(reliability["default"])
        self._social_reliability = float(reliability["social"])
        self._social_domains = {d.lower() for d in reliability.get("social_domains", [])}
        self._domain_reliability = {d.lower(): float(v) for d, v in reliability.get("domains", {}).items()}
        self._saturation = max(2, int(config["corroboration"]["saturation_sources"]))

    @classmethod
    def from_yaml(cls, path: Path | str) -> WeightedImpactModel:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def reliability(self, domain: str | None) -> float:
        if not domain:
            return self._default_reliability
        domain = domain.lower()
        if domain in self._social_domains:
            return self._social_reliability
        for known, value in self._domain_reliability.items():
            if domain == known or domain.endswith("." + known):
                return value
        return self._default_reliability

    def score(self, inp: ImpactInput) -> ImpactResult:
        features = {
            "severity": self.severity.get(inp.event_type, 2.0) / 10.0,
            "sentiment_magnitude": abs(inp.sentiment_score),
            "exposure": self.exposure.get(inp.scope, 0.5),
            "source_reliability": self.reliability(inp.source_domain),
            "corroboration": min(1.0, (inp.corroboration - 1) / (self._saturation - 1)),
        }
        raw = sum(self.weights[name] * value for name, value in features.items())
        return ImpactResult(
            score=round(1.0 + 9.0 * min(1.0, max(0.0, raw)), 2),
            features={k: round(v, 4) for k, v in features.items()},
            components={k: round(9.0 * self.weights[k] * v, 4) for k, v in features.items()},
        )
