from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from prism.config import BACKEND_ROOT, REPO_ROOT, Settings
from prism.core.contracts import Document, RawDocument, RiskSignal
from prism.core.taxonomy import EventType
from prism.etl.transform import normalize_batch
from prism.nlp.engine import RiskEngine
from prism.nlp.entities import DictionaryEntityLinker
from prism.nlp.events import RuleEventClassifier
from prism.nlp.impact import WeightedImpactModel
from prism.nlp.sentiment import LexiconSentiment
from prism.storage import SqlStore

CONFIG_DIR = BACKEND_ROOT / "config"
SAMPLE_REPLAY = REPO_ROOT / "data" / "replay" / "sample_news.jsonl"


@pytest.fixture
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture
def sample_replay() -> Path:
    return SAMPLE_REPLAY


@pytest.fixture
def engine() -> RiskEngine:
    """Real config, light backends (lexicon + rules): no model downloads."""
    return RiskEngine(
        entity_linker=DictionaryEntityLinker.from_yaml(CONFIG_DIR / "companies.yaml"),
        sentiment=LexiconSentiment(),
        events=RuleEventClassifier.from_yaml(CONFIG_DIR / "event_rules.yaml"),
        impact=WeightedImpactModel.from_yaml(CONFIG_DIR / "impact.yaml"),
    )


@pytest.fixture
def store() -> SqlStore:
    return SqlStore("sqlite://")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="sqlite://",
        data_dir=tmp_path,
        enabled_sources="replay",
        replay_file=SAMPLE_REPLAY,
        scheduler_enabled=False,
        write_signals_jsonl=True,
    )


def make_docs(*texts: str, source: str = "test", publisher: str | None = "reuters.com") -> list[Document]:
    docs, rejected = normalize_batch(RawDocument(source=source, body=t, publisher=publisher) for t in texts)
    assert not rejected, rejected
    return docs


def make_signal(
    *,
    event_type: EventType = EventType.GEOPOLITICAL,
    impact: float = 9.0,
    sentiment: float = -0.8,
    confidence: float = 0.9,
    signal_id: str = "sig-1",
) -> RiskSignal:
    return RiskSignal(
        id=signal_id,
        doc_id="doc-1",
        source="test",
        timestamp=datetime.now(timezone.utc),
        scope="market",
        entity="MARKET",
        sentiment_score=sentiment,
        event_type=event_type,
        impact_score=impact,
        confidence=confidence,
        headline="synthetic headline",
    )
