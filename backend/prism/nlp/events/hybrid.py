"""Rules first, embeddings for what the rules cannot see.

  * rules fired      -> blend rule and embedding distributions (rule_weight vs 1 - rule_weight)
  * no rule fired    -> trust the embedding model alone, but only if it is confident enough; otherwise
                        abstain (Other, matched=False) so the engine can drop irrelevant text.
                        Calibrated on 4,117 labelled finance tweets: without abstention the embedding model
                        labels 86% of non-event tweets (analyst notes, stock commentary) as events; a minimum
                        confidence of 0.3 halves that (41%) while macro-F1 on real events stays ~0.51
  * no embedding model configured -> plain rules (graceful degradation)
"""

from __future__ import annotations

from typing import Sequence

from prism.core.contracts import EventResult
from prism.core.interfaces import EventClassifier
from prism.core.taxonomy import EventType


class HybridEventClassifier:
    name = "hybrid"

    def __init__(
        self,
        rules: EventClassifier,
        embedding: EventClassifier | None = None,
        rule_weight: float = 0.6,
        embedding_min_confidence: float = 0.3,
    ) -> None:
        self.rules = rules
        self.embedding = embedding
        self.rule_weight = rule_weight
        self.embedding_min_confidence = embedding_min_confidence

    def classify(self, texts: Sequence[str]) -> list[EventResult]:
        rule_results = self.rules.classify(texts)
        if self.embedding is None:
            return rule_results
        embedding_results = self.embedding.classify(texts)

        out: list[EventResult] = []
        for rule, emb in zip(rule_results, embedding_results):
            if not rule.matched:
                if emb.confidence >= self.embedding_min_confidence:
                    out.append(emb.model_copy(update={"method": "hybrid:embedding"}))
                else:
                    out.append(rule.model_copy(update={"method": "hybrid:abstain"}))
                continue
            w = self.rule_weight
            keys = set(rule.scores) | set(emb.scores)
            blended = {k: w * rule.scores.get(k, 0.0) + (1 - w) * emb.scores.get(k, 0.0) for k in keys}
            top = max(blended, key=blended.__getitem__)
            out.append(
                EventResult(
                    event_type=EventType(top),
                    confidence=min(1.0, blended[top]),
                    scores=blended,
                    method="hybrid:rules+embedding",
                )
            )
        return out
