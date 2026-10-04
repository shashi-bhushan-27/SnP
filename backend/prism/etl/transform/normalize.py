"""Bronze -> silver: turn a RawDocument into a clean, analyzable Document (or reject it)."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import Iterable
from urllib.parse import urlparse

from prism.core.contracts import Document, RawDocument, RejectedRecord, ensure_utc
from prism.core.ids import stable_id
from prism.etl.transform.language import is_english

MIN_CHARS = 15
MAX_TITLE_CHARS = 300

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_INVISIBLE_RE = re.compile("[​-‏⁠﻿]")
_TRUNCATED_RE = re.compile(r"\s*\[\+\d+ chars\]\s*$")  # NewsAPI truncation marker
_RETWEET_RE = re.compile(r"^RT\s+@\w+:\s*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RejectDocument(ValueError):
    """Raised inside to_document; the message becomes the reject reason."""


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = _TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")  # curly apostrophes (McDonald's)
    text = _INVISIBLE_RE.sub("", text)
    text = _TRUNCATED_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _RETWEET_RE.sub("", text.strip())
    return _WS_RE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Hash of the lower-cased alphanumeric tokens: insensitive to punctuation, case and spacing."""
    return hashlib.sha1(" ".join(_TOKEN_RE.findall(text.lower())).encode("utf-8")).hexdigest()


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else (host or None)


def to_document(raw: RawDocument) -> Document:
    body = clean_text(raw.body)
    title = clean_text(raw.title) or body[:MAX_TITLE_CHARS]
    text = body if (not body or title in body) else f"{title}. {body}"
    text = text or title
    if len(text) < MIN_CHARS:
        raise RejectDocument("too_short")
    if not is_english(text, raw.language):
        raise RejectDocument("non_english")

    digest = content_hash(text)
    return Document(
        id=stable_id(raw.source, raw.external_id or raw.url or digest),
        source=raw.source,
        source_domain=_domain(raw.url) or (raw.publisher.lower() if raw.publisher else None),
        url=raw.url,
        title=title,
        text=text,
        published_at=ensure_utc(raw.published_at or raw.fetched_at),
        language="en",
        content_hash=digest,
    )


def normalize_batch(raws: Iterable[RawDocument]) -> tuple[list[Document], list[RejectedRecord]]:
    docs: list[Document] = []
    rejects: list[RejectedRecord] = []
    for raw in raws:
        try:
            docs.append(to_document(raw))
        except RejectDocument as exc:
            rejects.append(RejectedRecord(source=raw.source, reason=str(exc), snippet=(raw.title or raw.body or "")[:120]))
    return docs, rejects
