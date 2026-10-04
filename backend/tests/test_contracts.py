from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from prism.core.contracts import RiskSignal, SentimentResult
from prism.core.taxonomy import EventType


def _signal(**overrides) -> RiskSignal:
    base = dict(
        id="s1",
        doc_id="d1",
        source="test",
        timestamp=datetime.now(timezone.utc),
        sentiment_score=-0.5,
        event_type=EventType.CREDIT_EVENT,
        impact_score=7.5,
        confidence=0.8,
        headline="h",
    )
    return RiskSignal(**{**base, **overrides})


def test_valid_signal_serialises_event_as_label() -> None:
    assert _signal().model_dump(mode="json")["event_type"] == "Credit Event"


@pytest.mark.parametrize(
    "field,value",
    [("sentiment_score", 1.5), ("sentiment_score", -1.01), ("impact_score", 0.5), ("impact_score", 10.5), ("confidence", 1.2)],
)
def test_out_of_range_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        _signal(**{field: value})


def test_sentiment_result_bounds() -> None:
    with pytest.raises(ValidationError):
        SentimentResult(score=2.0, confidence=0.5)


@pytest.mark.parametrize("raw,expected", [("geopolitical", EventType.GEOPOLITICAL), ("Credit Event", EventType.CREDIT_EVENT), ("MARKET_SHOCK", EventType.MARKET_SHOCK)])
def test_event_type_parse(raw: str, expected: EventType) -> None:
    assert EventType.parse(raw) is expected


def test_event_type_parse_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        EventType.parse("not an event")
