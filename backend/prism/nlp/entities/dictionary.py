"""Dictionary-based entity linking: company names, aliases and cashtags -> ticker.

For a closed universe (~40 index constituents) this beats a generic NER model on precision and
needs no model download. Ambiguous names (Apple, Meta, Visa...) only count when the text also
carries finance context, which removes most fruit/metadata/travel false positives.
A spaCy NER fallback for unknown organisations can be added behind the same EntityLinker protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from prism.core.contracts import EntityMention

FINANCE_CONTEXT = frozenset(
    """shares share stock stocks earnings revenue profit profits quarterly analyst analysts investors
    nasdaq nyse dividend ceo cfo guidance market markets traders trading valuation bond bonds
    iphone ipad aws azure""".split()
)
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5}(?:\.[A-Z])?)\b")
_EXCHANGE_TAG_RE = re.compile(r"\b(?:NASDAQ|NYSE|AMEX)\s*:\s*([A-Z]{1,5}(?:\.[A-Z])?)\b")
_WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Company:
    name: str
    ticker: str
    sector: str | None = None
    aliases: tuple[str, ...] = ()
    ambiguous: bool = False


class DictionaryEntityLinker:
    def __init__(self, companies: Sequence[Company]) -> None:
        self._by_ticker = {c.ticker: c for c in companies}
        patterns: list[tuple[str, Company, re.Pattern[str]]] = []
        for company in companies:
            for alias in {company.name, *company.aliases}:
                patterns.append((alias, company, re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])")))
        self._patterns = sorted(patterns, key=lambda p: -len(p[0]))

    @classmethod
    def from_yaml(cls, path: Path | str) -> DictionaryEntityLinker:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        companies = [
            Company(
                name=c["name"],
                ticker=c["ticker"],
                sector=c.get("sector"),
                aliases=tuple(c.get("aliases", ())),
                ambiguous=bool(c.get("ambiguous", False)),
            )
            for c in data["companies"]
        ]
        return cls(companies)

    def link(self, text: str) -> list[EntityMention]:
        found: dict[str, tuple[int, EntityMention]] = {}

        def add(company: Company, position: int, matched: str, confidence: float) -> None:
            current = found.get(company.ticker)
            mention = EntityMention(
                name=company.name,
                ticker=company.ticker,
                sector=company.sector,
                scope="company",
                matched_text=matched,
                confidence=confidence,
            )
            if current is None or position < current[0]:
                found[company.ticker] = (position, mention)
            elif confidence > current[1].confidence:
                found[company.ticker] = (current[0], mention)

        for regex in (_CASHTAG_RE, _EXCHANGE_TAG_RE):
            for match in regex.finditer(text):
                company = self._by_ticker.get(match.group(1))
                if company:
                    add(company, match.start(), match.group(0), 1.0)

        has_context: bool | None = None
        for alias, company, pattern in self._patterns:
            match = pattern.search(text)
            if not match:
                continue
            if company.ambiguous:
                if has_context is None:
                    has_context = bool(FINANCE_CONTEXT & set(_WORD_RE.findall(text.lower())))
                if not has_context:
                    continue
            add(company, match.start(), alias, 0.8 if company.ambiguous else 0.95)

        return [mention for _, mention in sorted(found.values(), key=lambda item: item[0])]
