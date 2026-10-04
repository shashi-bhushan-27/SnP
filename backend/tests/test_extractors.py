from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from prism.core.errors import SourceUnavailable
from prism.etl.extract import GdeltSource, NewsApiSource, QuotaGuard, ReplaySource

FIXTURES = Path(__file__).parent / "fixtures"


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class MemoryState:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get_state(self, key):
        return self.data.get(key)

    def set_state(self, key, value):
        self.data[key] = value


# ================================================================ GDELT
GDELT_BODY = json.loads((FIXTURES / "gdelt_artlist.json").read_text(encoding="utf-8"))


def gdelt(handler, queries=("q",), **kwargs) -> GdeltSource:
    kwargs.setdefault("min_interval", 0)
    kwargs.setdefault("sleep", lambda _s: None)
    return GdeltSource(list(queries), client=client_for(handler), **kwargs)


def test_gdelt_parses_articles() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=GDELT_BODY)

    result = gdelt(handler).fetch(cursor=None, limit=10)
    assert seen["mode"] == "artlist" and seen["format"] == "json" and seen["timespan"] == "1h"
    assert len(result.documents) == 2
    first = result.documents[0]
    assert first.source == "gdelt" and first.language == "en"
    assert first.publisher == "example-news.com"
    assert first.published_at == datetime(2026, 10, 4, 18, 15, tzinfo=timezone.utc)  # oldest first
    assert result.next_cursor == "2026-10-04T18:30:00+00:00"


def test_gdelt_dedupes_the_same_url_across_queries() -> None:
    result = gdelt(lambda r: httpx.Response(200, json=GDELT_BODY), queries=("q1", "q2")).fetch(None, 10)
    assert len(result.documents) == 2


def test_gdelt_empty_result_keeps_the_cursor() -> None:
    result = gdelt(lambda r: httpx.Response(200, json={})).fetch(cursor="2026-10-04T18:30:00+00:00", limit=10)
    assert result.documents == []
    assert result.next_cursor == "2026-10-04T18:30:00+00:00"


def test_gdelt_cursor_becomes_a_timespan_of_at_least_15_minutes() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json={})

    recent = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    gdelt(handler).fetch(cursor=recent, limit=10)
    assert seen["timespan"] == "15min"

    older = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    gdelt(handler).fetch(cursor=older, limit=10)
    assert seen["timespan"] in {"180min", "181min"}


def test_gdelt_backs_off_exponentially_on_429_then_recovers() -> None:
    sleeps, statuses = [], iter([429, 429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, json=GDELT_BODY) if status == 200 else httpx.Response(429, text="Please limit requests")

    source = gdelt(handler, max_retries=2, backoff_seconds=60, sleep=sleeps.append)
    assert len(source.fetch(None, 10).documents) == 2
    assert sleeps == [60, 120]


def test_gdelt_gives_up_after_max_retries() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, text="Please limit requests")

    with pytest.raises(SourceUnavailable, match="429"):
        gdelt(handler, max_retries=2).fetch(None, 10)
    assert len(calls) == 3


def test_gdelt_spaces_consecutive_requests() -> None:
    sleeps = []
    source = gdelt(lambda r: httpx.Response(200, json={}), queries=("a", "b"), min_interval=6.0, sleep=sleeps.append, clock=lambda: 100.0)
    source.fetch(None, 10)
    assert sleeps == [6.0]  # nothing before the first request, a full gap before the second


def test_gdelt_rejects_non_json_bodies() -> None:
    with pytest.raises(SourceUnavailable, match="non-JSON"):
        gdelt(lambda r: httpx.Response(200, text="<html>oops</html>")).fetch(None, 10)


def test_gdelt_network_errors_are_source_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(SourceUnavailable):
        gdelt(handler).fetch(None, 10)


# ================================================================ NewsAPI
NEWSAPI_BODY = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "source": {"id": None, "name": "Example Wire"},
            "title": "Chipmaker unveils processor",
            "description": "The company said the chip doubles throughput.",
            "content": "The company said the chip doubles throughput for data centers… [+1200 chars]",
            "url": "https://example.com/a",
            "publishedAt": "2026-10-04T18:30:00Z",
        },
        {
            "source": {"id": None, "name": "Example News"},
            "title": "Fed holds rates",
            "description": None,
            "content": None,
            "url": "https://example.com/b",
            "publishedAt": "2026-10-04T18:10:00Z",
        },
    ],
}


def newsapi(handler, key="secret-key", **kwargs) -> NewsApiSource:
    return NewsApiSource(key, "stocks", client=client_for(handler), **kwargs)


def test_newsapi_parses_articles_and_sends_key_as_header() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"], captured["params"] = request.headers, dict(request.url.params)
        return httpx.Response(200, json=NEWSAPI_BODY)

    result = newsapi(handler).fetch(cursor="2026-10-04T10:00:00", limit=20)
    assert captured["headers"]["x-api-key"] == "secret-key"
    assert "apiKey" not in captured["params"] and captured["params"]["from"] == "2026-10-04T10:00:00"
    assert captured["params"]["sortBy"] == "publishedAt"
    assert captured["params"]["searchIn"] == "title,description"
    assert "domains" not in captured["params"]
    assert [d.title for d in result.documents] == ["Fed holds rates", "Chipmaker unveils processor"]  # oldest first
    assert result.documents[1].publisher == "Example Wire"
    assert "doubles throughput" in result.documents[1].body
    assert result.next_cursor == "2026-10-04T18:30:00"


def test_newsapi_domain_filter_and_full_content_search() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.url.params)
        return httpx.Response(200, json=NEWSAPI_BODY)

    newsapi(handler, search_in=None, domains=["cnbc.com", "ft.com"]).fetch(None, 10)
    assert captured["domains"] == "cnbc.com,ft.com"
    assert "searchIn" not in captured


def test_newsapi_without_a_key_is_unavailable_and_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(SourceUnavailable, match="NEWSAPI_KEY"):
        newsapi(handler, key=None).fetch(None, 10)


@pytest.mark.parametrize("status,match", [(401, "API key"), (429, "rate limit")])
def test_newsapi_http_errors(status: int, match: str) -> None:
    with pytest.raises(SourceUnavailable, match=match):
        newsapi(lambda r: httpx.Response(status, json={"status": "error", "message": "nope"})).fetch(None, 10)


def test_newsapi_error_payload_surfaces_the_message() -> None:
    with pytest.raises(SourceUnavailable, match="parametersMissing"):
        newsapi(lambda r: httpx.Response(200, json={"status": "error", "message": "parametersMissing"})).fetch(None, 10)


def test_quota_guard_stops_requests_when_the_daily_budget_is_used() -> None:
    state, calls = MemoryState(), []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=NEWSAPI_BODY)

    source = newsapi(handler, quota=QuotaGuard(state, "newsapi", daily_limit=2, today=lambda: date(2026, 10, 4)))
    source.fetch(None, 10)
    source.fetch(None, 10)
    with pytest.raises(SourceUnavailable, match="budget"):
        source.fetch(None, 10)
    assert len(calls) == 2


def test_quota_resets_the_next_day() -> None:
    state, today = MemoryState(), [date(2026, 10, 4)]
    guard = QuotaGuard(state, "newsapi", 1, today=lambda: today[0])
    guard.acquire()
    with pytest.raises(SourceUnavailable):
        guard.acquire()
    today[0] = date(2026, 10, 5)
    guard.acquire()


# ================================================================ replay
def test_replay_streams_in_batches_and_advances_the_cursor(sample_replay) -> None:
    source = ReplaySource(sample_replay)
    first = source.fetch(None, 5)
    assert len(first.documents) == 5 and first.next_cursor == "5"
    second = source.fetch(first.next_cursor, 5)
    assert second.documents[0].external_id == "sample_news:5"


def test_replay_at_the_end_returns_nothing_and_keeps_the_cursor(sample_replay) -> None:
    source = ReplaySource(sample_replay)
    total = len(sample_replay.read_text(encoding="utf-8").splitlines())
    result = source.fetch(str(total), 5)
    assert result.documents == [] and result.next_cursor == str(total)


def test_replay_can_loop(sample_replay) -> None:
    source = ReplaySource(sample_replay, loop=True)
    total = len(sample_replay.read_text(encoding="utf-8").splitlines())
    assert source.fetch(str(total), 2).documents[0].external_id == "sample_news:0"


def test_replay_rebases_timestamps_to_now_unless_told_otherwise(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"text": "Stocks rally on strong earnings", "published_at": "2020-01-02T03:04:05+00:00"}) + "\n", encoding="utf-8")
    rebased = ReplaySource(path).fetch(None, 1).documents[0]
    original = ReplaySource(path, rebase_time=False).fetch(None, 1).documents[0]
    assert abs((datetime.now(timezone.utc) - rebased.published_at).total_seconds()) < 30
    assert original.published_at == datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_replay_missing_file_is_source_unavailable(tmp_path) -> None:
    with pytest.raises(SourceUnavailable, match="not found"):
        ReplaySource(tmp_path / "missing.jsonl").fetch(None, 1)
