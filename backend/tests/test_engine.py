from __future__ import annotations

from conftest import make_docs
from prism.core.taxonomy import EventType


def test_company_headline_becomes_one_entity_signal(engine) -> None:
    (signal,) = engine.analyze(make_docs("Tesla shares rise as quarterly deliveries beat analyst estimates"))
    assert (signal.ticker, signal.entity, signal.scope) == ("TSLA", "Tesla", "company")
    assert signal.event_type is EventType.EARNINGS
    assert signal.sentiment_score > 0
    assert 1.0 <= signal.impact_score <= 10.0
    assert 0.0 <= signal.confidence <= 1.0


def test_signal_explains_itself(engine) -> None:
    (signal,) = engine.analyze(make_docs("Tesla shares rise as quarterly deliveries beat analyst estimates"))
    explanation = signal.explanation
    assert set(explanation["impact_points"]) == {"severity", "sentiment_magnitude", "exposure", "source_reliability", "corroboration"}
    assert explanation["event_top3"][0]["event"] == "Earnings"
    assert explanation["models"] == {"sentiment": "lexicon", "event": "rules", "impact": "weighted-v1"}
    assert explanation["entity_match"] == "Tesla"


def test_text_without_entity_but_with_an_event_is_market_scope(engine) -> None:
    (signal,) = engine.analyze(make_docs("Escalating tensions between two major economies trigger new sanctions and tariffs"))
    assert signal.scope == "market" and signal.entity == "MARKET" and signal.ticker is None
    assert signal.event_type is EventType.GEOPOLITICAL
    assert signal.sentiment_score < 0


def test_irrelevant_text_is_dropped(engine) -> None:
    assert engine.analyze(make_docs("Lumen Labs raises 50 million in a Series B round")) == []


def test_each_entity_gets_its_own_sentiment_from_its_own_sentences(engine) -> None:
    signals = engine.analyze(make_docs("Apple shares surge after record earnings. Intel shares plunge after weak guidance."))
    by_ticker = {s.ticker: s for s in signals}
    assert set(by_ticker) == {"AAPL", "INTC"}
    assert by_ticker["AAPL"].sentiment_score > 0 > by_ticker["INTC"].sentiment_score
    assert by_ticker["AAPL"].id != by_ticker["INTC"].id


def test_signal_ids_are_deterministic(engine) -> None:
    text = "Tesla shares rise as quarterly deliveries beat analyst estimates"
    first = engine.analyze(make_docs(text))[0]
    second = engine.analyze(make_docs(text))[0]
    assert first.id == second.id


def test_more_corroboration_means_higher_impact(engine) -> None:
    (doc,) = make_docs("Stocks plunge in broad sell-off as volatility index spikes to highest level this year")
    single = engine.analyze([doc])[0]
    corroborated = engine.analyze([doc.model_copy(update={"corroboration": 4})])[0]
    assert corroborated.impact_score > single.impact_score


def test_empty_batch(engine) -> None:
    assert engine.analyze([]) == []
