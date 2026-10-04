"""RiskEngine: composes the components into one `Analyzer` (documents -> RiskSignals).

    relevance            commentary / listicle / advice headlines are not events
    event classification headline first; body-only evidence counts at reduced confidence
    entity linking       headline first; body-only mentions count at reduced confidence
    sentiment            per document x entity, on the sentences that mention the entity
    impact + confidence  per signal

Salience rules (learned on live NewsAPI data, Oct 2026): a story whose subject AND event appear only in
the body, or a market-level story whose event appears only in the body, is dropped - those were the
false positives ("The spice of the matter" -> Macroeconomic 7.6; a grocery chain closing stores ->
Walmart, which the body merely mentioned).

Confidence is the geometric mean of the components (sentiment, event, entity), so one shaky
component pulls the whole signal down instead of being averaged away.
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
COMMENTARY = EventResult(
    event_type=EventType.OTHER, confidence=0.8, scores={EventType.OTHER.value: 1.0}, method="commentary", matched=True
)
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
        commentary_patterns: Sequence[str] = (),
        max_entities_per_doc: int = 3,
        drop_irrelevant: bool = True,
        body_evidence_discount: float = 0.7,
        body_entity_confidence: float = 0.6,
    ) -> None:
        self.entity_linker = entity_linker
        self.sentiment = sentiment
        self.events = events
        self.impact = impact
        self.max_entities_per_doc = max_entities_per_doc
        self.drop_irrelevant = drop_irrelevant
        self.body_evidence_discount = body_evidence_discount
        self.body_entity_confidence = body_entity_confidence
        self._commentary = [re.compile(p, re.IGNORECASE) for p in commentary_patterns]

    # ------------------------------------------------------------------ evidence gathering
    def is_commentary(self, title: str) -> bool:
        return any(p.search(title) for p in self._commentary)

    def _classify_events(self, docs: Sequence[Document]) -> list[tuple[EventResult, str]]:
        """(event, where the evidence came from: 'title' | 'body' | 'commentary')."""
        results: list[tuple[EventResult, str]] = [(COMMENTARY, "commentary")] * len(docs)
        pending = [i for i, d in enumerate(docs) if not self.is_commentary(d.title)]
        need_body: list[int] = []
        for i, event in zip(pending, self.events.classify([docs[i].title for i in pending])):
            results[i] = (event, "title")
            if not event.matched and docs[i].text != docs[i].title:
                need_body.append(i)
        if need_body:
            for i, event in zip(need_body, self.events.classify([docs[i].text for i in need_body])):
                if event.matched:
                    discounted = event.model_copy(
                        update={
                            "confidence": round(event.confidence * self.body_evidence_discount, 4),
                            "method": f"{event.method}+body",
                        }
                    )
                    results[i] = (discounted, "body")
        return results

    def _link_entities(self, doc: Document) -> tuple[list[EntityMention], str]:
        mentions = self.entity_linker.link(doc.title)
        if mentions or doc.text == doc.title:
            return mentions, "title"
        body = self.entity_linker.link(doc.text)
        capped = [m.model_copy(update={"confidence": min(m.confidence, self.body_entity_confidence)}) for m in body]
        return capped, "body"

    # ------------------------------------------------------------------ analysis
    def analyze(self, docs: Sequence[Document]) -> list[RiskSignal]:
        if not docs:
            return []

        targets: list[tuple[Document, EventResult, EntityMention, dict]] = []
        for doc, (event, event_source) in zip(docs, self._classify_events(docs)):
            mentions, entity_source = self._link_entities(doc)
            mentions = mentions[: self.max_entities_per_doc]
            weak_event = event.event_type == EventType.OTHER or event_source == "body"
            if self.drop_irrelevant:
                if not mentions and weak_event:
                    continue  # no tracked entity and no event reported in the headline
                if mentions and entity_source == "body" and weak_event:
                    continue  # neither the subject nor the event is in the headline
            if not mentions:
                mentions, entity_source = [MARKET_MENTION], "none"
            evidence = {"event": event_source, "entity": entity_source}
            targets.extend((doc, event, mention, evidence) for mention in mentions)
        if not targets:
            return []

        sentiments = self.sentiment.predict([focus_text(doc, mention) for doc, _, mention, _ in targets])
        return [
            self._build(doc, event, mention, sentiment, evidence)
            for (doc, event, mention, evidence), sentiment in zip(targets, sentiments)
        ]

    def _build(
        self, doc: Document, event: EventResult, mention: EntityMention, sentiment: SentimentResult, evidence: dict
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
            corroboration=doc.corroboration,
            source_reliability=impact.features.get("source_reliability", 0.5),
            headline=doc.title,
            url=doc.url,
            explanation={
                "impact_points": impact.components,
                "impact_features": impact.features,
                "event_top3": [{"event": name, "p": round(p, 4)} for name, p in top_events],
                "event_method": event.method,
                "evidence": evidence,
                "sentiment_probabilities": {k: round(v, 4) for k, v in sentiment.probabilities.items()},
                "entity_match": mention.matched_text,
                "corroboration": doc.corroboration,
                "models": {"sentiment": self.sentiment.name, "event": self.events.name, "impact": self.impact.name},
            },
        )
