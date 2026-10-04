"""Composition root: the one place that knows which concrete implementation fills each slot.

Everything else depends on protocols from prism.core. To swap a backend, change a setting; to add
one, write a class and add an entry to the matching factory dict below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import yaml

from prism.config import REPO_ROOT, Settings, get_settings
from prism.core.interfaces import EventClassifier, SentimentModel, SignalConsumer, Source
from prism.etl.extract import GdeltSource, NewsApiSource, QuotaGuard, ReplaySource
from prism.etl.load import JsonlSignalSink
from prism.etl.pipeline import EtlPipeline
from prism.etl.scheduler import PollingScheduler
from prism.modules.stress import Portfolio, ScenarioBook, StressEngine, StressTrigger
from prism.nlp.engine import RiskEngine
from prism.nlp.entities import DictionaryEntityLinker
from prism.nlp.events import EmbeddingEventClassifier, HybridEventClassifier, RuleEventClassifier
from prism.nlp.impact import WeightedImpactModel
from prism.nlp.sentiment import FinBertSentiment, LexiconSentiment
from prism.storage import SqlStore

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    store: SqlStore
    engine: RiskEngine
    pipeline: EtlPipeline
    portfolio: Portfolio
    scenarios: ScenarioBook
    stress_engine: StressEngine
    scheduler: PollingScheduler | None = None


# ---------------------------------------------------------------------- NLP slots
def build_sentiment(settings: Settings) -> SentimentModel:
    factories = {
        "lexicon": lambda: LexiconSentiment(),
        "finbert": lambda: FinBertSentiment(settings.finbert_model),
    }
    return factories[settings.sentiment_backend]()


def build_events(settings: Settings) -> EventClassifier:
    rules = RuleEventClassifier.from_yaml(settings.config_dir / "event_rules.yaml")
    if settings.event_backend == "rules":
        return rules
    embedding = EmbeddingEventClassifier.from_yaml(
        settings.config_dir / "event_prototypes.yaml", model_name=settings.embedding_model
    )
    if settings.event_backend == "embedding":
        return embedding
    return HybridEventClassifier(rules, embedding)


def build_engine(settings: Settings) -> RiskEngine:
    return RiskEngine(
        entity_linker=DictionaryEntityLinker.from_yaml(settings.config_dir / "companies.yaml"),
        sentiment=build_sentiment(settings),
        events=build_events(settings),
        impact=WeightedImpactModel.from_yaml(settings.config_dir / "impact.yaml"),
    )


# ---------------------------------------------------------------------- ETL slots
def build_sources(settings: Settings, store: SqlStore) -> dict[str, Source]:
    cfg = yaml.safe_load((settings.config_dir / "sources.yaml").read_text(encoding="utf-8"))
    replay_path = settings.replay_file or (REPO_ROOT / cfg["replay"]["file"])
    factories = {
        "replay": lambda: ReplaySource(replay_path, rebase_time=settings.replay_rebase_time),
        "gdelt": lambda: GdeltSource(
            cfg["gdelt"]["queries"], timespan=cfg["gdelt"]["timespan"], max_records=cfg["gdelt"]["max_records"]
        ),
        "newsapi": lambda: NewsApiSource(
            settings.newsapi_key,
            cfg["newsapi"]["query"],
            language=cfg["newsapi"]["language"],
            page_size=cfg["newsapi"]["page_size"],
            quota=QuotaGuard(store, "newsapi", settings.newsapi_daily_budget),
        ),
    }
    unknown = [n for n in settings.source_names if n not in factories]
    if unknown:
        raise ValueError(f"Unknown source(s) in ENABLED_SOURCES: {unknown}. Available: {sorted(factories)}")
    return {name: factories[name]() for name in settings.source_names}


def build_container(
    settings: Settings | None = None,
    *,
    store: SqlStore | None = None,
    sources: Mapping[str, Source] | None = None,
    engine: RiskEngine | None = None,
) -> Container:
    settings = settings or get_settings()
    store = store or SqlStore(settings.database_url)
    engine = engine or build_engine(settings)

    portfolio = Portfolio.from_yaml(settings.config_dir / "portfolio.yaml")
    scenarios = ScenarioBook.from_yaml(settings.config_dir / "scenarios.yaml")
    stress_engine = StressEngine()

    consumers: list[SignalConsumer] = []
    if settings.write_signals_jsonl:
        consumers.append(JsonlSignalSink(settings.data_dir / "processed" / "signals.jsonl"))
    if settings.stress_auto_trigger:
        consumers.append(
            StressTrigger(
                portfolio, scenarios, stress_engine, on_result=lambda r: store.save_stress_run(r.model_dump(mode="json"))
            )
        )

    pipeline = EtlPipeline(
        sources=sources if sources is not None else build_sources(settings, store),
        store=store,
        analyzer=engine,
        consumers=consumers,
    )

    scheduler = None
    if settings.scheduler_enabled:
        poll = {
            "gdelt": settings.gdelt_poll_seconds,
            "newsapi": settings.newsapi_poll_seconds,
            "replay": settings.replay_poll_seconds,
        }
        scheduler = PollingScheduler(
            pipeline,
            intervals={name: poll[name] for name in pipeline.sources if name in poll},
            limits={"replay": settings.replay_batch_size},
        )

    return Container(
        settings=settings,
        store=store,
        engine=engine,
        pipeline=pipeline,
        portfolio=portfolio,
        scenarios=scenarios,
        stress_engine=stress_engine,
        scheduler=scheduler,
    )
