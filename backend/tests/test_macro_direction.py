from __future__ import annotations

import pytest

from prism.core.contracts import SentimentResult
from prism.nlp.sentiment import MacroDirectionSentiment


class Fixed:
    """A base model that returns the same score for every text (like FinBERT reading macro news literally)."""

    name = "fixed"

    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, texts):
        return [SentimentResult(score=self.score, probabilities={}, confidence=0.9) for _ in texts]


@pytest.fixture
def rules_path(config_dir):
    return config_dir / "macro_direction.yaml"


def model(rules_path, score):
    return MacroDirectionSentiment.from_yaml(Fixed(score), rules_path)


@pytest.mark.parametrize(
    "text,direction",
    [
        ("US inflation hits 8.6%, a 40-year high, beating forecasts", -1),
        ("Consumer price index rises 0.5% in September, above economists' forecasts", -1),
        ("Fed signals further interest rate hikes", -1),
        ("Treasury yields hit a 16-year high", -1),
        ("Weak US jobs report raises recession fears", -1),
        ("US inflation cools more than expected", 1),
        ("US inflation comes in cooler than expected", 1),
        ("Fed cuts interest rates by half a percentage point", 0),  # cuts are ambiguous (March 2020)
        ("Top economist lowers recession risk for next year", 0),
        ("Apple launches a new iPhone", 0),
    ],
)
def test_direction_rules(rules_path, text, direction) -> None:
    assert model(rules_path, 0.0).direction(text) == direction


def test_literal_positive_reading_of_hot_inflation_is_flipped(rules_path) -> None:
    (result,) = model(rules_path, 0.9).predict(["US inflation hits 8.6%, a 40-year high, beating forecasts"])
    assert result.score == pytest.approx(-0.9) and result.adjustment == "macro-bearish"


def test_neutral_reading_gets_a_minimum_magnitude(rules_path) -> None:
    (result,) = model(rules_path, 0.0).predict(["Fed signals further interest rate hikes"])
    assert result.score == pytest.approx(-0.5)


def test_agreeing_or_unmatched_texts_are_untouched(rules_path) -> None:
    agree, other = model(rules_path, -0.8).predict(["Treasury yields hit a 16-year high", "Apple launches a new iPhone"])
    assert agree.score == pytest.approx(-0.8) and agree.adjustment is None
    assert other.score == pytest.approx(-0.8) and other.adjustment is None


def test_name_shows_the_layer(rules_path) -> None:
    assert model(rules_path, 0.0).name == "fixed+macro"
