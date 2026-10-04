from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------- salience rules (from live NewsAPI data)
def article(title: str, body: str):
    from prism.core.contracts import RawDocument
    from prism.etl.transform import to_document

    return to_document(RawDocument(source="newsapi", title=title, body=body, url="https://example.com/x", language="en"))


@pytest.mark.parametrize(
    "title",
    [
        "3 Canadian AI Stocks With Revenue Growth Up To 176%",
        "How to Protect Your Portfolio as the Fed Raises Interest Rates",
        "Cameco Stock Is Down 24% Over the Past Three Months. Time to Sell or Load Up?",
        "Top 3 Japanese Hidden Gem Stocks To Watch In October",
        "4 Dividends That Have Not Been Raised in Years",
    ],
)
def test_commentary_headlines_are_not_events(engine, title) -> None:
    assert engine.is_commentary(title)
    assert engine.analyze([article(title, "Inflation and rate hikes weigh on markets as recession fears grow.")]) == []


def test_commentary_with_a_tracked_company_is_kept_but_marked_other(engine) -> None:
    (signal,) = engine.analyze([article("Should You Buy Tesla Stock Before Earnings?", "Analysts are divided.")])
    assert signal.ticker == "TSLA" and signal.event_type is EventType.OTHER
    assert signal.explanation["event_method"] == "commentary"


def test_event_only_in_the_body_does_not_make_a_market_signal(engine) -> None:
    # live false positive: a food article whose body mentioned inflation became Macroeconomic 7.6
    doc = article("The spice of the matter", "Prices have soared as inflation and interest rates squeeze cooks.")
    assert engine.analyze([doc]) == []


def test_entity_and_event_only_in_the_body_are_dropped(engine) -> None:
    # live false positive: a grocery chain closing stores was attributed to Walmart (mentioned in the body)
    doc = article(
        "Discount grocery chain makes big changes after 42 store closures",
        "The chain, which competes with Walmart, said the closures follow a probe by regulators.",
    )
    assert engine.analyze([doc]) == []


def test_headline_company_with_body_event_is_kept_at_lower_confidence(engine) -> None:
    title_only = article("Tesla faces regulatory investigation over autopilot", "")
    body_event = article("Tesla in focus this week", "Regulators opened an investigation into the autopilot system.")
    (strong,) = engine.analyze([title_only])
    (weak,) = engine.analyze([body_event])
    assert strong.event_type is weak.event_type is EventType.REGULATORY
    assert weak.explanation["evidence"] == {"event": "body", "entity": "title"}
    assert weak.explanation["event_method"].endswith("+body")
    assert weak.confidence < strong.confidence


def test_company_found_only_in_the_body_gets_capped_confidence(engine) -> None:
    (signal,) = engine.analyze(
        [article("Regulators open antitrust investigation into cloud pricing", "Microsoft said it would cooperate.")]
    )
    assert signal.ticker == "MSFT" and signal.explanation["evidence"] == {"event": "title", "entity": "body"}
    assert signal.confidence <= 0.75


def test_signal_carries_corroboration_and_source_reliability(engine) -> None:
    (doc,) = make_docs("Stocks plunge in broad sell-off as volatility index spikes", publisher="reuters.com")
    (signal,) = engine.analyze([doc.model_copy(update={"corroboration": 3})])
    assert signal.corroboration == 3 and signal.source_reliability == 1.0
