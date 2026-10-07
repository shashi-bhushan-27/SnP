"""GDELT 2.0 Global Knowledge Graph (GKG) raw-file adapter.

Why a second GDELT adapter: the DOC 2.0 search API rate-limits hard and blocked the development network
(HTTP 429 for days), while GDELT's raw 15-minute files are plain static downloads. One GKG file is ~5-6 MB
zipped and holds ~1,400 English articles with a page title, URL, source domain, GDELT themes, organisations
and tone. This adapter keeps only articles tagged with finance themes (config/sources.yaml).

Observed behaviour (Oct 2026) that shapes the code:
  * lastupdate.txt can list a slot whose file is not published yet (404 for up to ~1 hour), so slots are
    walked one by one and a 404 simply means "not yet" - the cursor never moves past a missing file
  * http:// URLs redirect to https://
  * GKG 2.1 rows have 27 tab-separated columns; the title is in <PAGE_TITLE> inside column 26
"""

from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator, Sequence

import httpx

from prism.core.contracts import FetchResult, RawDocument
from prism.core.errors import SourceUnavailable

BASE_URL = "https://data.gdeltproject.org/gdeltv2/"
SLOT = timedelta(minutes=15)
_SLOT_FORMAT = "%Y%m%d%H%M%S"
_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_PUBTIME_RE = re.compile(r"<PAGE_PRECISEPUBTIMESTAMP>(\d{14})</PAGE_PRECISEPUBTIMESTAMP>")
COLUMNS = 27
csv.field_size_limit(2**31 - 1)


def floor_slot(moment: datetime) -> datetime:
    moment = moment.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return moment.replace(minute=moment.minute - moment.minute % 15)


def parse_slot(value: str) -> datetime:
    return datetime.strptime(value, _SLOT_FORMAT).replace(tzinfo=timezone.utc)


def _themes(row: list[str]) -> list[str]:
    # V2 enhanced themes "THEME,offset;..." (col 8); fall back to V1 themes (col 7)
    raw = row[8] or row[7]
    seen: dict[str, None] = {}
    for item in raw.split(";"):
        if item:
            seen.setdefault(item.split(",")[0], None)
    return list(seen)


def _organisations(row: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in (row[14] or row[13]).split(";"):
        if item:
            seen.setdefault(item.split(",")[0].strip(), None)
    return list(seen)[:10]


def parse_gkg(content: bytes, theme_prefixes: Sequence[str]) -> Iterator[RawDocument]:
    """Yield finance-tagged articles from one zipped GKG file."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")
    prefixes = tuple(theme_prefixes)
    for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) < COLUMNS:
            continue
        themes = _themes(row)
        if prefixes and not any(t.startswith(prefixes) for t in themes):
            continue
        title_match = _TITLE_RE.search(row[26])
        if not title_match or not row[4].startswith("http"):
            continue
        published = _PUBTIME_RE.search(row[26])
        try:
            tone = float(row[15].split(",")[0])
        except ValueError:
            tone = None
        yield RawDocument(
            source="gdelt_gkg",
            external_id=row[4],
            url=row[4],
            title=html.unescape(title_match.group(1)).strip(),
            published_at=parse_slot(published.group(1) if published else row[1]),
            language="en",  # the main GKG stream is English (translated content ships separately)
            publisher=row[3] or None,
            payload={
                "gkg_record_id": row[0],
                "themes": [t for t in themes if t.startswith(prefixes)][:12] if prefixes else themes[:12],
                "organisations": _organisations(row),
                "tone": tone,
            },
        )


class GdeltGkgSource:
    name = "gdelt_gkg"

    def __init__(
        self,
        theme_prefixes: Sequence[str],
        *,
        client: httpx.Client | None = None,
        base_url: str = BASE_URL,
        max_files_per_fetch: int = 4,
        lookback_slots: int = 8,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.theme_prefixes = list(theme_prefixes)
        self._client = client or httpx.Client(timeout=120, follow_redirects=True)
        self.base_url = base_url
        self.max_files_per_fetch = max_files_per_fetch
        self.lookback_slots = lookback_slots
        self._now = now

    def _download(self, slot: datetime) -> bytes | None:
        url = f"{self.base_url}{slot.strftime(_SLOT_FORMAT)}.gkg.csv.zip"
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise SourceUnavailable(f"GDELT GKG download failed: {exc}") from exc
        if response.status_code == 404:
            return None  # listed but not published yet (or never published)
        if response.is_error:
            raise SourceUnavailable(f"GDELT GKG HTTP {response.status_code}")
        return response.content

    def _slots(self, cursor: str | None) -> list[datetime]:
        latest = floor_slot(self._now())
        if cursor is None:  # first run: newest available file only, searching back a little
            return [latest - i * SLOT for i in range(self.lookback_slots)]
        slots, slot = [], parse_slot(cursor) + SLOT
        while slot <= latest:
            slots.append(slot)
            slot += SLOT
        return slots

    def fetch(self, cursor: str | None, limit: int) -> FetchResult:
        docs: list[RawDocument] = []
        processed: str | None = None
        files = 0
        for slot in self._slots(cursor):
            content = self._download(slot)
            if content is None:
                if cursor is None:
                    continue  # first run walks backwards until it finds a published file
                break  # incremental runs stop at the first gap, so no file is ever skipped
            try:
                docs.extend(parse_gkg(content, self.theme_prefixes))
            except (zipfile.BadZipFile, UnicodeDecodeError):
                pass  # a corrupt file is skipped, but the cursor still moves past it
            processed = slot.strftime(_SLOT_FORMAT)
            files += 1
            if cursor is None or files >= self.max_files_per_fetch:
                break
        return FetchResult(documents=docs[:limit] if limit else docs, next_cursor=processed or cursor)
