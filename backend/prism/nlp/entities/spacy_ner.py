"""spaCy NER on top of the dictionary linker.

The dictionary stays authoritative for tracked companies (it knows tickers and sectors); spaCy's ORG
entities add organisations the dictionary has never heard of ("Northbridge Capital files for Chapter 11"),
which then become company-scope signals without a ticker. Regulators, central banks, exchanges and news
outlets are tagged ORG by spaCy but are not companies, so they are ignored (`non_company_orgs` in
config/companies.yaml).

On live GDELT headlines (Title Case) the small English model tags fragments such as "Israeli Banks Fear
Expanded U.K." or "Circle Stock Falls" as ORG, and also people and universities. So an untracked ORG is accepted
only if its last word is a corporate suffix (Holdings, Bank, Energy, Inc ... `corporate_suffixes` in
companies.yaml).

spaCy and the model are imported lazily:
    pip install spacy && python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from prism.core.contracts import EntityMention
from prism.core.interfaces import EntityLinker
from prism.nlp.entities.dictionary import DictionaryEntityLinker

_POSSESSIVE_RE = re.compile(r"(?:'s|')$")
_LEADING_THE_RE = re.compile(r"^the\s+", re.IGNORECASE)


def clean_org(text: str) -> str:
    return _LEADING_THE_RE.sub("", _POSSESSIVE_RE.sub("", text.strip())).strip()


class SpacyEntityLinker:
    def __init__(
        self,
        dictionary: EntityLinker,
        *,
        model: str = "en_core_web_sm",
        nlp: Any = None,
        ignore: Iterable[str] = (),
        max_untracked: int = 2,
        confidence: float = 0.6,
        corporate_suffixes: Iterable[str] = (),
    ) -> None:
        self.dictionary = dictionary
        self.model = model
        self._nlp = nlp  # injectable for tests: callable(text) -> object with .ents
        self.ignore = {name.lower() for name in ignore}
        self.max_untracked = max_untracked
        self.confidence = confidence
        self.corporate_suffixes = {w.lower() for w in corporate_suffixes}

    @classmethod
    def from_yaml(cls, companies_path: Path | str, **kwargs: Any) -> SpacyEntityLinker:
        data = yaml.safe_load(Path(companies_path).read_text(encoding="utf-8"))
        return cls(
            DictionaryEntityLinker.from_yaml(companies_path),
            ignore=data.get("non_company_orgs", []),
            corporate_suffixes=data.get("corporate_suffixes", []),
            **kwargs,
        )

    def _pipeline(self) -> Any:
        if self._nlp is None:
            import spacy  # lazy

            self._nlp = spacy.load(self.model, disable=["parser", "lemmatizer"])
        return self._nlp

    def link(self, text: str) -> list[EntityMention]:
        mentions = list(self.dictionary.link(text))
        seen = {m.name.lower() for m in mentions} | {m.matched_text.lower() for m in mentions if m.matched_text}
        added = 0
        for ent in self._pipeline()(text).ents:
            if ent.label_ != "ORG" or added >= self.max_untracked:
                continue
            name = clean_org(ent.text)
            key = name.lower()
            if len(name) < 3 or key in self.ignore or any(key in s or s in key for s in seen):
                continue
            if self.corporate_suffixes and key.split()[-1].strip(".,") not in self.corporate_suffixes:
                continue
            mentions.append(
                EntityMention(name=name, scope="company", matched_text=ent.text, confidence=self.confidence)
            )
            seen.add(key)
            added += 1
        return mentions
