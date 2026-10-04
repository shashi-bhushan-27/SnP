"""Replay adapter: streams a JSONL file a few records at a time.

Why it exists: live APIs are rate-limited, delayed and unreliable during a demo. Replay makes
the demo deterministic and lets historical/social datasets (e.g. labelled finance tweets)
flow through the exact same pipeline as live sources.

Each line is a JSON object: {"text": ..., "title"?: ..., "publisher"?: ..., "published_at"?: ISO}.
Any extra fields (labels, flags) are kept in the raw payload.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from prism.core.contracts import FetchResult, RawDocument, ensure_utc, utcnow
from prism.core.errors import SourceUnavailable


class ReplaySource:
    def __init__(self, path: Path | str, *, name: str = "replay", rebase_time: bool = True, loop: bool = False) -> None:
        self.name = name
        self.path = Path(path)
        self.rebase_time = rebase_time
        self.loop = loop
        self._records: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._records is None:
            if not self.path.exists():
                raise SourceUnavailable(f"Replay file not found: {self.path}")
            with self.path.open(encoding="utf-8") as fh:
                self._records = [json.loads(line) for line in fh if line.strip()]
        return self._records

    def fetch(self, cursor: str | None, limit: int) -> FetchResult:
        records = self._load()
        offset = int(cursor or 0)
        if offset >= len(records) and self.loop and records:
            offset = 0
        batch = records[offset : offset + limit]

        docs = []
        for index, record in enumerate(batch, start=offset):
            published = None
            if not self.rebase_time and record.get("published_at"):
                published = ensure_utc(datetime.fromisoformat(record["published_at"]))
            docs.append(
                RawDocument(
                    source=self.name,
                    external_id=f"{self.path.stem}:{index}",
                    title=record.get("title"),
                    body=record.get("text"),
                    published_at=published or utcnow(),
                    language=record.get("language", "en"),
                    publisher=record.get("publisher"),
                    payload=record,
                )
            )
        return FetchResult(documents=docs, next_cursor=str(offset + len(batch)))
