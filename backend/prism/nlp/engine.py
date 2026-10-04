"""RiskEngine: composes the components into one `Analyzer` (documents -> RiskSignals).

    event classification (per document)
    entity linking       (per document; none found -> market-scope signal, or dropped if irrelevant)
    sentiment            (per document x entity, on the sentences that mention the entity)
    impact + confidence  (per signal)

Confidence is the geometric mean of the weakest-link components (sentiment, event, entity), so one
shaky component pulls the whole signal down instead of being averaged away.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from prism.core.contracts import Document, EntityMention, EventResult, ImpactInput, RiskSignal, SentimentResult
from prism.core.ids import stable_id
from prism.core.interfaces import EntityLinker, EventClassifier, ImpactModel, SentimentModel
from prism.core.taxonomy import EventType

MARKET_MENTION = EntityMention(name="MARKET", scope="market", confidence=0.8)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_FOCUS_CHARS = 1000


def focus_text(doc: Document, mention: EntityMention) -> str:
    """Entity-aware sentiment input: only sentences that mention the entity (else the whole text)."""
    if mention.scope != "company" or not mention.matched_text:
        return doc.text[:_MAX_FOCUS_CHARS]
    sentences = [s for s in _SENTENCE_RE.split(doc.text) if mention.matched_text in s]
    return (" ".join(sentences) or doc.text)[:_MAX_FOCUS_CHARS]


def _geometric_mean(*values: float) -> float:
    clipped = [min(1.0, max(0.0, v)) for v in values]
    return math.prod(clipped) ** (1.0 / len(clipped))


class RiskEngine:
    def __init__(
        self,
        *,
        entity_linker: EntityLinker,
        sentiment: SentimentModel,
        events: EventClassifier,
        impact: ImpactModel,
        max_entities_per_doc: int = 3,
        drop_irrelevant: bool = True,
    ) -> None:
        self.entity_linker = entity_linker
        self.sentiment = sentiment
        self.events = events
        self.impact = impact
        self.max_entities_per_doc = max_entities_per_doc
        self.drop_irrelevant = drop_irrelevant

    def analyze(self, docs: Sequence[Document]) -> list[RiskSignal]:
        if not docs:
            return []

        event_results = self.events.classify([d.text for d in docs])
        targets: list[tuple[Document, EventResult, EntityMention]] = []
        for doc, event in zip(docs, event_results):
            mentions = self.entity_linker.link(doc.text)[: self.max_entities_per_doc]
            if not mentions:
                if event.event_type == EventType.OTHER and self.drop_irrelevant:
                    continue  # no tracked entity and no recognisable event: not a risk signal
                mentions = [MARKET_MENTION]
            targets.extend((doc, event, mention) for mention in mentions)
        if not targets:
            return []

        sentiments = self.sentiment.predict([focus_text(doc, mention) for doc, _, mention in targets])
        return [
            self._build(doc, event, mention, sentiment)
            for (doc, event, mention), sentiment in zip(targets, sentiments)
        ]

    def _build(
        self, doc: Document, event: EventResult, mention: EntityMention, sentiment: SentimentResult
    ) -> RiskSignal:
        impact = self.impact.score(
            ImpactInput(
                event_type=event.event_type,
                sentiment_score=sentiment.score,
                scope=mention.scope,
                source=doc.source,
                source_domain=doc.source_domain,
                corroboration=doc.corroboration,
            )
        )
        top_events = sorted(event.scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return RiskSignal(
            id=stable_id(doc.id, mention.ticker or mention.name),
            doc_id=doc.id,
            source=doc.source,
            timestamp=doc.published_at,
            entity=mention.name,
            ticker=mention.ticker,
            sector=mention.sector,
            scope=mention.scope,
            sentiment_score=sentiment.score,
            event_type=event.event_type,
            impact_score=impact.score,
            confidence=round(_geometric_mean(sentiment.confidence, event.confidence, mention.confidence), 4),
            headline=doc.title,
            url=doc.url,
            explanation={
                "impact_points": impact.components,
                "impact_features": impact.features,
                "event_top3": [{"event": name, "p": round(p, 4)} for name, p in top_events],
                "event_method": event.method,
                "sentiment_probabilities": {k: round(v, 4) for k, v in sentiment.probabilities.items()},
                "entity_match": mention.matched_text,
                "corroboration": doc.corroboration,
                "models": {"sentiment": self.sentiment.name, "event": self.events.name, "impact": self.impact.name},
            },
        )
