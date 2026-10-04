"""Exact + near-duplicate handling.

Duplicates are not just dropped: when different outlets report the same story, the number of
distinct sources becomes `Document.corroboration`, which feeds the impact score.

Near-duplicates use Jaccard similarity over word 3-grams. That catches syndicated copies and
lightly edited headlines but NOT paraphrases - semantic clustering with sentence embeddings is
the planned upgrade (the embedding encoder already exists in nlp/events).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from prism.core.contracts import Document

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def shingles(text: str, size: int = 3) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) <= size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class _Kept:
    doc: Document
    shingles: set[str]
    sources: set[str] = field(default_factory=set)
    merged_hashes: set[str] = field(default_factory=set)


@dataclass
class DedupeResult:
    documents: list[Document]
    duplicates: int


class Deduper:
    def __init__(self, known_hashes: Iterable[str] = (), near_threshold: float = 0.75) -> None:
        self.known_hashes = set(known_hashes)
        self.near_threshold = near_threshold

    @staticmethod
    def _source_key(doc: Document) -> str:
        return doc.source_domain or doc.source

    def run(self, docs: Sequence[Document]) -> DedupeResult:
        kept: list[_Kept] = []
        by_hash: dict[str, _Kept] = {}
        duplicates = 0

        for doc in docs:
            key = self._source_key(doc)
            if doc.content_hash in by_hash:
                by_hash[doc.content_hash].sources.add(key)
                duplicates += 1
                continue
            if doc.content_hash in self.known_hashes:  # already loaded in an earlier run
                duplicates += 1
                continue

            sh = shingles(doc.text)
            match = next((k for k in kept if jaccard(sh, k.shingles) >= self.near_threshold), None)
            if match:
                match.sources.add(key)
                match.merged_hashes.add(doc.content_hash)  # remembered, so re-reads of the same window stay idempotent
                by_hash[doc.content_hash] = match
                duplicates += 1
                continue

            entry = _Kept(doc=doc, shingles=sh, sources={key})
            kept.append(entry)
            by_hash[doc.content_hash] = entry

        documents = [
            k.doc.model_copy(update={"corroboration": max(1, len(k.sources)), "merged_hashes": sorted(k.merged_hashes)})
            for k in kept
        ]
        return DedupeResult(documents=documents, duplicates=duplicates)
