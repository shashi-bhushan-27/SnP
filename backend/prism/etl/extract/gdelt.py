"""GDELT DOC 2.0 adapter (https://api.gdeltproject.org/api/v2/doc/doc, mode=artlist).

Facts that shape this adapter:
  * the API returns titles/URLs/metadata only - never article text
  * the index refreshes about every 15 minutes
  * the limiter is stricter than the documented "1 request / 5 s": a 429 can close the
    gate for a minute or more, so requests are spaced and 429s back off exponentially
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Callable, Sequence

import httpx

from prism.core.contracts import FetchResult, RawDocument, ensure_utc, utcnow
from prism.core.errors import SourceUnavailable

log = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_SEEN_FORMAT = "%Y%m%dT%H%M%SZ"
_LANGUAGES = {"english": "en"}
_MIN_TIMESPAN_MIN = 15
_MAX_TIMESPAN_MIN = 60 * 24 * 90  # rolling 3-month window


def parse_seendate(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, _SEEN_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class GdeltSource:
    name = "gdelt"

    def __init__(
        self,
        queries: Sequence[str],
        *,
        timespan: str = "1h",
        max_records: int = 75,
        client: httpx.Client | None = None,
        min_interval: float = 6.0,
        max_retries: int = 2,
        backoff_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.queries = list(queries)
        self.timespan = timespan
        self.max_records = max_records
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    # ------------------------------------------------------------------ public
    def fetch(self, cursor: str | None, limit: int) -> FetchResult:
        by_url: dict[str, RawDocument] = {}
        for query in self.queries:
            for article in self._articles(query, cursor, limit):
                url = article.get("url")
                if url and url not in by_url:
                    by_url[url] = self._to_raw(article)

        docs = sorted(by_url.values(), key=lambda d: d.published_at or utcnow())[-limit:]
        newest = max((d.published_at for d in docs if d.published_at), default=None)
        next_cursor = newest.isoformat() if newest else cursor
        return FetchResult(documents=docs, next_cursor=next_cursor)

    # ------------------------------------------------------------------ internals
    def _timespan_param(self, cursor: str | None) -> str:
        if not cursor:
            return self.timespan
        since = ensure_utc(datetime.fromisoformat(cursor))
        minutes = math.ceil((utcnow() - since).total_seconds() / 60)
        return f"{min(max(minutes, _MIN_TIMESPAN_MIN), _MAX_TIMESPAN_MIN)}min"

    def _articles(self, query: str, cursor: str | None, limit: int) -> list[dict]:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": min(max(limit, 1), self.max_records, 250),
            "timespan": self._timespan_param(cursor),
        }
        response = self._get(params)
        try:
            data = response.json()
        except ValueError as exc:
            raise SourceUnavailable("GDELT returned a non-JSON body") from exc
        # GDELT answers `{}` when nothing matches
        return data.get("articles") or []

    def _get(self, params: dict) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            if self._last_request is not None:
                wait = self.min_interval - (self._clock() - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            try:
                response = self._client.get(GDELT_URL, params=params)
            except httpx.HTTPError as exc:
                raise SourceUnavailable(f"GDELT request failed: {exc}") from exc
            finally:
                self._last_request = self._clock()

            if response.status_code == 429:
                if attempt == self.max_retries:
                    raise SourceUnavailable("GDELT rate limit (HTTP 429) - try again later")
                delay = self.backoff_seconds * (2**attempt)
                log.warning("GDELT 429, backing off %.0fs (attempt %d)", delay, attempt + 1)
                self._sleep(delay)
                continue
            if response.is_error:
                raise SourceUnavailable(f"GDELT HTTP {response.status_code}")
            return response
        raise SourceUnavailable("GDELT unavailable")  # pragma: no cover

    @staticmethod
    def _to_raw(article: dict) -> RawDocument:
        language = (article.get("language") or "").strip().lower()
        return RawDocument(
            source="gdelt",
            external_id=article.get("url"),
            url=article.get("url"),
            title=article.get("title"),
            published_at=parse_seendate(article.get("seendate")),
            language=_LANGUAGES.get(language, language or None),
            publisher=article.get("domain"),
            payload=article,
        )
