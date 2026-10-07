"""GDELT GKG raw-file adapter: synthetic zipped GKG files served by a mock transport (no network)."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import httpx
import pytest

from prism.core.errors import SourceUnavailable
from prism.etl.extract import GdeltGkgSource
from prism.etl.extract.gdelt_gkg import floor_slot, parse_gkg

THEMES = ["ECON_STOCKMARKET", "ECON_BANKRUPTCY", "SANCTIONS"]


def gkg_row(record: str, url: str, title: str | None, themes: str, tone: str = "-3.5,1,4.5,5.5,20,0,600") -> str:
    cols = [""] * 27
    cols[0], cols[1], cols[2], cols[3], cols[4] = record, "20261007194500", "1", "example-news.com", url
    cols[7] = ";".join(t.split(",")[0] for t in themes.split(";"))
    cols[8] = themes
    cols[14] = "Northbridge Capital,120;Federal Reserve,300"
    cols[15] = tone
    if title is not None:
        cols[26] = f"<PAGE_PRECISEPUBTIMESTAMP>20261007193000</PAGE_PRECISEPUBTIMESTAMP><PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(cols)


def gkg_zip(*rows: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.gkg.csv", "\n".join(rows) + "\n")
    return buffer.getvalue()


FILE = gkg_zip(
    gkg_row("1", "https://example-news.com/a", "Northbridge Capital files for Chapter 11 bankruptcy", "ECON_BANKRUPTCY,10;TAX_FNCACT,40"),
    gkg_row("2", "https://example-news.com/b", "Local bakery wins a pie contest", "TAX_FOODSTAPLES,5"),
    gkg_row("3", "https://example-news.com/c", None, "ECON_STOCKMARKET,5"),  # no title -> skipped
    gkg_row("4", "https://example-news.com/d", "Stocks &amp; bonds slide as sanctions widen", "SANCTIONS,3;ECON_STOCKMARKET,9"),
    "too\tshort\trow",
)

NOW = datetime(2026, 10, 7, 20, 52, tzinfo=timezone.utc)  # latest slot 20:45


def source(published: dict[str, bytes], seen: list[str] | None = None, **kw) -> GdeltGkgSource:
    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if seen is not None:
            seen.append(name[:14])
        return httpx.Response(200, content=published[name[:14]]) if name[:14] in published else httpx.Response(404)

    return GdeltGkgSource(THEMES, client=httpx.Client(transport=httpx.MockTransport(handler)), now=lambda: NOW, **kw)


def test_parse_keeps_titled_finance_articles_only() -> None:
    docs = list(parse_gkg(FILE, THEMES))
    assert [d.title for d in docs] == [
        "Northbridge Capital files for Chapter 11 bankruptcy",
        "Stocks & bonds slide as sanctions widen",  # HTML entities decoded
    ]
    first = docs[0]
    assert first.source == "gdelt_gkg" and first.language == "en" and first.publisher == "example-news.com"
    assert first.published_at == datetime(2026, 10, 7, 19, 30, tzinfo=timezone.utc)  # precise publish time
    assert first.payload["tone"] == -3.5
    assert first.payload["themes"] == ["ECON_BANKRUPTCY"]
    assert first.payload["organisations"] == ["Northbridge Capital", "Federal Reserve"]


def test_empty_theme_list_keeps_everything_titled() -> None:
    assert len(list(parse_gkg(FILE, []))) == 3


def test_floor_slot() -> None:
    assert floor_slot(NOW) == datetime(2026, 10, 7, 20, 45, tzinfo=timezone.utc)


def test_first_run_walks_back_to_the_newest_published_file() -> None:
    seen: list[str] = []
    result = source({"20261007194500": FILE}, seen).fetch(None, 100)
    assert seen == ["20261007204500", "20261007203000", "20261007201500", "20261007200000", "20261007194500"]
    assert len(result.documents) == 2 and result.next_cursor == "20261007194500"


def test_incremental_run_stops_at_the_first_gap_and_never_skips_a_file() -> None:
    published = {"20261007200000": FILE, "20261007203000": FILE}  # 20:15 missing (not yet published)
    result = source(published).fetch("20261007194500", 100)
    assert result.next_cursor == "20261007200000"  # 20:30 is not consumed past the 20:15 gap
    assert len(result.documents) == 2


def test_incremental_run_caps_the_number_of_files() -> None:
    published = {f"202610071{m}": FILE for m in ("94500", "95000")}
    published.update({"20261007200000": FILE, "20261007201500": FILE, "20261007203000": FILE, "20261007204500": FILE})
    result = source(published, max_files_per_fetch=2).fetch("20261007194500", 100)
    assert result.next_cursor == "20261007201500" and len(result.documents) == 4


def test_nothing_new_keeps_the_cursor() -> None:
    result = source({}).fetch("20261007204500", 100)
    assert result.documents == [] and result.next_cursor == "20261007204500"


def test_limit_is_respected() -> None:
    assert len(source({"20261007194500": FILE}).fetch(None, 1).documents) == 1


def test_corrupt_file_is_skipped_but_consumed() -> None:
    result = source({"20261007200000": b"not a zip"}).fetch("20261007194500", 100)
    assert result.documents == [] and result.next_cursor == "20261007200000"


def test_network_errors_are_source_unavailable() -> None:
    def refuse(request):
        raise httpx.ConnectError("down")

    gkg = GdeltGkgSource(THEMES, client=httpx.Client(transport=httpx.MockTransport(refuse)), now=lambda: NOW)
    with pytest.raises(SourceUnavailable):
        gkg.fetch(None, 10)


def test_server_errors_are_source_unavailable() -> None:
    gkg = GdeltGkgSource(THEMES, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))), now=lambda: NOW)
    with pytest.raises(SourceUnavailable, match="503"):
        gkg.fetch("20261007194500", 10)
