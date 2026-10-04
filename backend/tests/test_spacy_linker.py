"""SpacyEntityLinker with a fake spaCy pipeline (no model download needed)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from prism.nlp.entities import DictionaryEntityLinker, SpacyEntityLinker
from prism.nlp.entities.spacy_ner import clean_org


@dataclass
class Ent:
    text: str
    label_: str


class FakeNlp:
    """Returns the entities it was told to, regardless of the text (like a perfectly predictable NER)."""

    def __init__(self, *ents: Ent):
        self.ents, self.calls = list(ents), []

    def __call__(self, text):
        self.calls.append(text)
        return self


@pytest.fixture
def companies(config_dir):
    return config_dir / "companies.yaml"


def linker(companies, *ents, **kwargs) -> SpacyEntityLinker:
    return SpacyEntityLinker.from_yaml(companies, nlp=FakeNlp(*ents), **kwargs)


def test_untracked_organisation_becomes_a_company_mention_without_ticker(companies) -> None:
    (mention,) = linker(companies, Ent("Northbridge Capital", "ORG")).link(
        "Northbridge Capital files for Chapter 11 bankruptcy protection"
    )
    assert (mention.name, mention.ticker, mention.scope) == ("Northbridge Capital", None, "company")
    assert mention.confidence < 0.95  # less certain than a dictionary match


def test_dictionary_matches_win_and_are_not_duplicated(companies) -> None:
    mentions = linker(companies, Ent("Tesla", "ORG"), Ent("Tesla Inc", "ORG"), Ent("Helix Motors", "ORG")).link(
        "Tesla shares fall as Helix Motors cuts its outlook"
    )
    assert [(m.name, m.ticker) for m in mentions] == [("Tesla", "TSLA"), ("Helix Motors", None)]


def test_regulators_outlets_and_non_org_entities_are_ignored(companies) -> None:
    mentions = linker(
        companies,
        Ent("the Federal Reserve", "ORG"),
        Ent("SEC", "ORG"),
        Ent("Reuters", "ORG"),
        Ent("China", "GPE"),
        Ent("Wednesday", "DATE"),
    ).link("The Federal Reserve and the SEC said on Wednesday, Reuters reported, citing China")
    assert mentions == []


def test_untracked_mentions_are_capped(companies) -> None:
    ents = [Ent(name, "ORG") for name in ("Alpha Holdings", "Beta Partners", "Gamma Group")]
    mentions = linker(companies, *ents, max_untracked=2).link("Alpha Holdings, Beta Partners and Gamma Group merge")
    assert [m.name for m in mentions] == ["Alpha Holdings", "Beta Partners"]


@pytest.mark.parametrize(
    "raw,clean", [("Helix Motors'", "Helix Motors"), ("Northbridge's", "Northbridge"), ("The Orion Group", "Orion Group")]
)
def test_org_names_are_cleaned(raw: str, clean: str) -> None:
    assert clean_org(raw) == clean


def test_plugs_into_the_engine_and_produces_a_ticker_less_signal(engine, companies) -> None:
    from conftest import make_docs

    engine.entity_linker = linker(companies, Ent("Northbridge Capital", "ORG"))
    (signal,) = engine.analyze(make_docs("Northbridge Capital files for Chapter 11 bankruptcy protection"))
    assert (signal.entity, signal.ticker, signal.scope) == ("Northbridge Capital", None, "company")
    assert signal.event_type.value == "Bankruptcy"


def test_dictionary_linker_still_satisfies_the_same_protocol(companies) -> None:
    from prism.core.interfaces import EntityLinker

    assert isinstance(DictionaryEntityLinker.from_yaml(companies), EntityLinker)
    assert isinstance(linker(companies), EntityLinker)
