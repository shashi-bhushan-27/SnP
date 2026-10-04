"""NewsAPI.org adapter (/v2/everything).

Free "Developer" plan: 100 requests/day, articles delayed 24 h, development use only
(not for staging/production). The QuotaGuard keeps polling inside the daily budget.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import httpx

from prism.core.contracts import FetchResult, RawDocument, ensure_utc
from prism.core.errors import SourceUnavailable
from prism.etl.extract.quota import QuotaGuard

NEWSAPI_URL = "https://newsapi.org/v2/everything"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class NewsApiSource:
    name = "newsapi"

    def __init__(
        self,
        api_key: str | None,
        query: str,
        *,
        language: str = "en",
        page_size: int = 50,
        search_in: str | None = "title,description",
        domains: Sequence[str] = (),
        client: httpx.Client | None = None,
        quota: QuotaGuard | None = None,
    ) -> None:
        self._api_key = api_key
        self.query = query
        self.language = language
        self.page_size = page_size
        self.search_in = search_in
        self.domains = list(domains)
        self.quota = quota
        self._client = client or httpx.Client(timeout=30.0)

    def fetch(self, cursor: str | None, limit: int) -> FetchResult:
        if not self._api_key:
            raise SourceUnavailable("NEWSAPI_KEY is not set")
        if self.quota:
            self.quota.acquire()

        params: dict[str, str | int] = {
            "q": self.query,
            "language": self.language,
            "sortBy": "publishedAt",
            "pageSize": min(max(limit, 1), self.page_size, 100),
        }
        if self.search_in:
            params["searchIn"] = self.search_in  # matching inside full content drags in off-topic stories
        if self.domains:
            params["domains"] = ",".join(self.domains)
        if cursor:
            params["from"] = cursor
        try:
            response = self._client.get(NEWSAPI_URL, params=params, headers={"X-Api-Key": self._api_key})
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"NewsAPI request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise SourceUnavailable("NewsAPI rejected the API key")
        if response.status_code in (426, 429):
            raise SourceUnavailable("NewsAPI rate limit reached")
        try:
            data = response.json()
        except ValueError as exc:
            raise SourceUnavailable("NewsAPI returned a non-JSON body") from exc
        if response.is_error or data.get("status") != "ok":
            raise SourceUnavailable(f"NewsAPI error: {data.get('message', response.status_code)}")

        docs = [self._to_raw(a) for a in data.get("articles", []) if a.get("url")]
        docs.sort(key=lambda d: d.published_at or _EPOCH)
        newest = max((d.published_at for d in docs if d.published_at), default=None)
        return FetchResult(documents=docs, next_cursor=newest.strftime("%Y-%m-%dT%H:%M:%S") if newest else cursor)

    @staticmethod
    def _to_raw(article: dict) -> RawDocument:
        published = article.get("publishedAt")
        body = " ".join(p for p in (article.get("description"), article.get("content")) if p)
        return RawDocument(
            source="newsapi",
            external_id=article["url"],
            url=article["url"],
            title=article.get("title"),
            body=body or None,
            published_at=ensure_utc(datetime.fromisoformat(published.replace("Z", "+00:00"))) if published else None,
            language="en",
            publisher=(article.get("source") or {}).get("name"),
            payload=article,
        )
