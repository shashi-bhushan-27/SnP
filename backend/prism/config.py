"""Runtime settings (environment / .env). Only the composition root and entry points read this."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    # --- storage / paths
    database_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'prism.db').as_posix()}"
    data_dir: Path = REPO_ROOT / "data"
    config_dir: Path = BACKEND_ROOT / "config"

    # --- NLP backends (defaults run with no model downloads; use finbert/hybrid for real runs)
    sentiment_backend: Literal["lexicon", "finbert"] = "lexicon"
    event_backend: Literal["rules", "embedding", "hybrid"] = "rules"
    entity_backend: Literal["dictionary", "spacy"] = "dictionary"
    spacy_model: str = "en_core_web_sm"
    finbert_model: str = "ProsusAI/finbert"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- sources
    enabled_sources: str = "replay"  # comma separated: replay,gdelt,newsapi
    newsapi_key: str | None = None
    newsapi_daily_budget: int = 80  # free plan allows 100/day; keep headroom
    replay_file: Path | None = None  # overrides config/sources.yaml
    replay_batch_size: int = 3
    replay_rebase_time: bool = True  # stamp replayed items "now" so dashboards look live

    # --- scheduler (live polling)
    scheduler_enabled: bool = False
    gdelt_poll_seconds: int = 300
    newsapi_poll_seconds: int = 1200
    replay_poll_seconds: int = 5

    # --- Module B scenarios
    scenario_source: Literal["history", "matrix"] = "history"  # history-calibrated (default) or scenarios.yaml shocks
    stress_quantile: float = 0.10  # 1-in-10 bad outcome among past events of the same kind
    stress_horizon: Literal["trough", "1d", "5d"] = "trough"
    analog_encoder: Literal["hashing", "embedding"] = "hashing"  # embedding = EMBEDDING_MODEL (shared with events)

    # --- downstream / API
    stress_auto_trigger: bool = True
    write_signals_jsonl: bool = True
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:8501"

    @property
    def source_names(self) -> list[str]:
        return [s.strip().lower() for s in self.enabled_sources.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
