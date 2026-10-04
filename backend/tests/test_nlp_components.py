from __future__ import annotations

import re

import numpy as np
import pytest

from prism.core.taxonomy import EventType
from prism.nlp.entities import DictionaryEntityLinker
from prism.nlp.events import EmbeddingEventClassifier, HybridEventClassifier, RuleEventClassifier
from prism.nlp.impact import WeightedImpactModel
from prism.nlp.sentiment import FinBertSentiment, LexiconSentiment
from prism.core.contracts import ImpactInput


# ================================================================ entity linking
@pytest.fixture
def linker(config_dir) -> DictionaryEntityLinker:
    return DictionaryEntityLinker.from_yaml(config_dir / "companies.yaml")


def tickers(linker: DictionaryEntityLinker, text: str) -> list[str]:
    return [m.ticker for m in linker.link(text)]


def test_links_company_by_name(linker) -> None:
    assert tickers(linker, "Tesla shares fell after a regulatory investigation") == ["TSLA"]


def test_alias_maps_to_canonical_company(linker) -> None:
    mentions = linker.link("Google unveils a new search feature")
    assert [(m.name, m.ticker) for m in mentions] == [("Alphabet", "GOOGL")]


def test_ambiguous_name_needs_finance_context(linker) -> None:
    assert tickers(linker, "She baked an Apple pie for the school fair") == []
    assert tickers(linker, "Apple shares rose after strong earnings") == ["AAPL"]
    assert tickers(linker, "Metadata and Meta keywords") == []
    assert tickers(linker, "Meta shares jump on advertising growth") == ["META"]


def test_cashtags_and_exchange_tags(linker) -> None:
    assert tickers(linker, "$TSLA and $NVDA both moving") == ["TSLA", "NVDA"]
    assert tickers(linker, "Microsoft Corp (NASDAQ: MSFT) rose") == ["MSFT"]


def test_multiple_names_ordered_by_first_mention_without_duplicates(linker) -> None:
    assert tickers(linker, "Goldman Sachs and Morgan Stanley beat; JPMorgan Chase and JPMorgan too") == ["GS", "MS", "JPM"]


def test_no_entity(linker) -> None:
    assert linker.link("Global markets slid on inflation worries") == []


# ================================================================ sentiment: lexicon
def test_lexicon_polarity() -> None:
    pos, neg, neu = LexiconSentiment().predict(
        ["Profits surge to a record on strong growth", "Shares plunge after fraud probe and weak guidance", "The board meets on Tuesday"]
    )
    assert pos.score > 0.5 and neg.score < -0.5 and neu.score == 0.0


def test_lexicon_negation_flips_polarity() -> None:
    (result,) = LexiconSentiment().predict(["Revenue did not beat estimates"])
    assert result.score < 0


def test_lexicon_probabilities_form_a_distribution_and_match_score() -> None:
    (result,) = LexiconSentiment().predict(["Stocks plunge as recession fears grow"])
    p = result.probabilities
    assert sum(p.values()) == pytest.approx(1.0)
    assert result.score == pytest.approx(p["positive"] - p["negative"], abs=1e-3)


def test_lexicon_is_unsure_when_no_word_matches() -> None:
    (result,) = LexiconSentiment().predict(["The board meets on Tuesday"])
    assert result.confidence < 0.5


# ================================================================ sentiment: FinBERT adapter (no model loaded)
class FakePipe:
    def __init__(self, outputs):
        self.outputs, self.calls = outputs, []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return self.outputs[: len(texts)]


def test_finbert_score_is_positive_minus_negative() -> None:
    pipe = FakePipe([[{"label": "negative", "score": 0.82}, {"label": "neutral", "score": 0.17}, {"label": "positive", "score": 0.01}]])
    (result,) = FinBertSentiment(pipe=pipe).predict(["Tesla shares plunge after regulatory investigation."])
    assert result.score == pytest.approx(-0.81)
    assert result.confidence == pytest.approx(0.82)
    assert pipe.calls[0][1]["truncation"] is True


def test_finbert_accepts_title_cased_labels_and_batches() -> None:
    row = [{"label": "Positive", "score": 0.7}, {"label": "Negative", "score": 0.1}, {"label": "Neutral", "score": 0.2}]
    pipe = FakePipe([row, row])
    results = FinBertSentiment(pipe=pipe).predict(["a", "b"])
    assert [r.score for r in results] == [pytest.approx(0.6)] * 2
    assert len(pipe.calls) == 1  # one batched call


def test_finbert_empty_input_does_not_load_a_model() -> None:
    assert FinBertSentiment().predict([]) == []


# ================================================================ event classification: rules
@pytest.fixture
def rules(config_dir) -> RuleEventClassifier:
    return RuleEventClassifier.from_yaml(config_dir / "event_rules.yaml")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Northbridge Capital files for Chapter 11 bankruptcy protection after missing bond payments", EventType.BANKRUPTCY),
        ("Rating agency downgrades Helix Motors to junk, citing rising default risk", EventType.CREDIT_EVENT),
        ("Escalating tensions between two major economies trigger new sanctions and tariffs", EventType.GEOPOLITICAL),
        ("Federal Reserve signals further interest rate hikes as inflation stays high", EventType.MACROECONOMIC),
        ("Consumer price index rises 0.5% in September, above economists' forecasts", EventType.MACROECONOMIC),
        ("Stocks plunge in broad sell-off as volatility index spikes", EventType.MARKET_SHOCK),
        ("Regulators open antitrust investigation into cloud pricing practices", EventType.REGULATORY),
        ("Jury orders Cobalt Energy to pay $800 million in patent infringement lawsuit verdict", EventType.LEGAL),
        ("Orion Systems agrees to acquire Vantage Analytics in $2.1 billion all-cash deal", EventType.MERGER_ACQUISITION),
        ("Apple posts record quarterly revenue as iPhone demand beats analyst estimates", EventType.EARNINGS),
        ("Microsoft launches new AI assistant for enterprise customers", EventType.PRODUCT_LAUNCH),
        ("Global chip shortage deepens as port delays disrupt factory shipments", EventType.SUPPLY_CHAIN),
        ("Veridian Pharma names new chief executive after abrupt departure of long-time CEO", EventType.LEADERSHIP_CHANGE),
        ("Hackers breach customer database at Meridian Bank, ransomware group claims responsibility", EventType.CYBERSECURITY),
    ],
)
def test_rules_classify_clear_cases(rules, text: str, expected: EventType) -> None:
    (result,) = rules.classify([text])
    assert result.event_type is expected
    assert result.matched
    assert 0.5 <= result.confidence <= 0.95
    assert sum(result.scores.values()) == pytest.approx(1.0)


def test_rules_report_no_evidence_instead_of_guessing(rules) -> None:
    (result,) = rules.classify(["Lumen Labs raises 50 million in a Series B round"])
    assert result.event_type is EventType.OTHER and not result.matched
    assert result.confidence < 0.5


def test_rules_case_sensitive_acronyms_do_not_misfire(rules) -> None:
    # "fed" the verb and "sec" the unit must not look like the Fed / SEC
    (result,) = rules.classify(["The cat fed the dog a few sec ago"])
    assert not result.matched


# ================================================================ event classification: embedding + hybrid (fake encoder)
class BagOfWordsEncoder:
    """Deterministic stand-in for a sentence-transformer: normalised word-count vectors."""

    def __init__(self, vocabulary):
        self.index = {w: i for i, w in enumerate(sorted(set(vocabulary)))}

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), len(self.index)))
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if word in self.index:
                    out[row, self.index[word]] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


PROTOTYPES = {
    EventType.BANKRUPTCY: ["company files for bankruptcy protection and liquidation"],
    EventType.MERGER_ACQUISITION: ["firm agrees to acquire rival in merger deal"],
    EventType.OTHER: ["community sponsorship event"],
}


@pytest.fixture
def embedding() -> EmbeddingEventClassifier:
    vocab = [w for sents in PROTOTYPES.values() for s in sents for w in re.findall(r"[a-z]+", s.lower())]
    return EmbeddingEventClassifier(PROTOTYPES, encoder=BagOfWordsEncoder(vocab), temperature=0.1)


def test_embedding_picks_nearest_prototype(embedding) -> None:
    bankrupt, merger = embedding.classify(["lender files bankruptcy protection", "bank agrees to acquire rival in merger"])
    assert bankrupt.event_type is EventType.BANKRUPTCY
    assert merger.event_type is EventType.MERGER_ACQUISITION
    assert sum(bankrupt.scores.values()) == pytest.approx(1.0)
    assert 0.0 < bankrupt.confidence <= 1.0


def test_hybrid_blends_when_rules_fire_and_falls_back_to_embedding_otherwise(rules, embedding) -> None:
    hybrid = HybridEventClassifier(rules, embedding)
    fired, silent = hybrid.classify(
        ["Northbridge Capital files for Chapter 11 bankruptcy protection", "firm agrees rival deal"]
    )
    assert fired.event_type is EventType.BANKRUPTCY and fired.method == "hybrid:rules+embedding"
    assert silent.event_type is EventType.MERGER_ACQUISITION and silent.method == "hybrid:embedding"


def test_hybrid_without_embedding_is_plain_rules(rules) -> None:
    (result,) = HybridEventClassifier(rules, None).classify(["Stocks plunge in broad sell-off"])
    assert result.event_type is EventType.MARKET_SHOCK and result.method == "rules"


# ================================================================ impact model
@pytest.fixture
def impact(config_dir) -> WeightedImpactModel:
    return WeightedImpactModel.from_yaml(config_dir / "impact.yaml")


def _inp(**kw) -> ImpactInput:
    base = dict(event_type=EventType.CREDIT_EVENT, sentiment_score=-0.5, scope="company", source_domain="reuters.com", corroboration=1)
    return ImpactInput(**{**base, **kw})


def test_hand_computed_score(impact) -> None:
    # raw = .35*1.0 + .25*0.8 + .20*1.0 + .10*1.0 + .10*1.0 = 0.95  ->  1 + 9*0.95
    result = impact.score(_inp(event_type=EventType.BANKRUPTCY, sentiment_score=-0.8, corroboration=5))
    assert result.score == pytest.approx(9.55)


def test_components_explain_the_score(impact) -> None:
    result = impact.score(_inp(corroboration=3))
    assert sum(result.components.values()) == pytest.approx(result.score - 1.0, abs=0.01)
    assert set(result.components) == {"severity", "sentiment_magnitude", "exposure", "source_reliability", "corroboration"}


def test_score_stays_in_range_at_the_extremes(impact) -> None:
    low = impact.score(_inp(event_type=EventType.OTHER, sentiment_score=0.0, scope="sector", source_domain="twitter", corroboration=1))
    high = impact.score(_inp(event_type=EventType.BANKRUPTCY, sentiment_score=-1.0, corroboration=50))
    assert 1.0 <= low.score < 3.5
    assert 9.0 < high.score <= 10.0


def test_score_is_monotonic_in_each_driver(impact) -> None:
    assert impact.score(_inp(event_type=EventType.BANKRUPTCY)).score > impact.score(_inp(event_type=EventType.PRODUCT_LAUNCH)).score
    assert impact.score(_inp(sentiment_score=-0.9)).score > impact.score(_inp(sentiment_score=-0.1)).score
    assert impact.score(_inp(corroboration=4)).score > impact.score(_inp(corroboration=1)).score
    assert impact.score(_inp(source_domain="reuters.com")).score > impact.score(_inp(source_domain="twitter")).score


def test_direction_does_not_matter_only_magnitude(impact) -> None:
    assert impact.score(_inp(sentiment_score=0.7)).score == impact.score(_inp(sentiment_score=-0.7)).score


def test_source_reliability_lookup(impact) -> None:
    assert impact.reliability("reuters.com") == 1.0
    assert impact.reliability("uk.reuters.com") == 1.0  # subdomain
    assert impact.reliability("twitter") == 0.4
    assert impact.reliability("someblog.example") == 0.5
    assert impact.reliability(None) == 0.5


def test_weights_must_sum_to_one(config_dir) -> None:
    import yaml

    cfg = yaml.safe_load((config_dir / "impact.yaml").read_text(encoding="utf-8"))
    cfg["weights"]["severity"] = 0.9
    with pytest.raises(ValueError, match="sum to 1.0"):
        WeightedImpactModel(cfg)
