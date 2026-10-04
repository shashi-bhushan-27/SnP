from __future__ import annotations

import pytest

from prism.core.contracts import Document, RawDocument
from prism.etl.transform import Deduper, clean_text, content_hash, normalize_batch, to_document
from prism.etl.transform.language import is_english


# ---------------------------------------------------------------- cleaning
def test_clean_text_strips_html_urls_and_entities() -> None:
    assert clean_text("<p>Stocks &amp; bonds rally https://t.co/abc123</p>") == "Stocks & bonds rally"


def test_clean_text_removes_newsapi_truncation_marker() -> None:
    assert clean_text("Apple surges on record sales and the outlook [+1234 chars]") == "Apple surges on record sales and the outlook"


def test_clean_text_handles_retweets_and_curly_apostrophes() -> None:
    assert clean_text("RT @someone: McDonald’s beats estimates") == "McDonald's beats estimates"


def test_clean_text_none_is_empty() -> None:
    assert clean_text(None) == ""


def test_content_hash_ignores_case_and_punctuation() -> None:
    assert content_hash("Fed hikes rates!") == content_hash("fed  hikes, rates")
    assert content_hash("Fed hikes rates") != content_hash("Fed cuts rates")


# ---------------------------------------------------------------- normalisation
def test_title_and_body_are_combined_once() -> None:
    doc = to_document(RawDocument(source="x", title="Fed hikes", body="The Federal Reserve raised interest rates by a quarter point"))
    assert doc.text == "Fed hikes. The Federal Reserve raised interest rates by a quarter point"


def test_body_that_already_contains_title_is_not_duplicated() -> None:
    doc = to_document(RawDocument(source="x", title="Fed hikes rates", body="Fed hikes rates by a quarter point on Wednesday"))
    assert doc.text == "Fed hikes rates by a quarter point on Wednesday"


def test_document_id_is_stable_and_domain_comes_from_url() -> None:
    raw = RawDocument(source="gdelt", external_id="u1", url="https://www.reuters.com/a/b", title="Markets rally on strong earnings")
    first, second = to_document(raw), to_document(raw)
    assert first.id == second.id
    assert first.source_domain == "reuters.com"
    assert first.published_at.tzinfo is not None


def test_publisher_is_used_when_there_is_no_url() -> None:
    doc = to_document(RawDocument(source="replay", body="Markets rally on strong earnings", publisher="Bloomberg.com"))
    assert doc.source_domain == "bloomberg.com"


def test_normalize_batch_separates_rejects_with_reasons() -> None:
    docs, rejects = normalize_batch(
        [
            RawDocument(source="x", body="Markets rally on strong earnings today"),
            RawDocument(source="x", body="tiny"),
            RawDocument(source="x", body="Esta es una frase suficientemente larga", language="es"),
        ]
    )
    assert len(docs) == 1
    assert [r.reason for r in rejects] == ["too_short", "non_english"]


def test_language_detection() -> None:
    assert is_english("The Federal Reserve raised interest rates on Wednesday")
    assert not is_english("中央银行连续第五次加息，市场反应强烈")
    assert not is_english("anything at all", hint="de")
    assert is_english("anything at all", hint="English")


def test_terse_english_headlines_are_not_mistaken_for_foreign_text() -> None:
    # headlines have few function words; they must not be rejected for that
    assert is_english("Nvidia unveils next-generation data center chip promises major performance gains")
    assert is_english("Federal Reserve signals further interest rate hikes as inflation stays stubbornly high, officials say")
    assert is_english("Oil prices spike")


@pytest.mark.parametrize(
    "text",
    [
        "El banco central sube los tipos de interes por quinta vez consecutiva",
        "Die Zentralbank erhoht die Zinsen zum funften Mal in Folge und die Markte reagieren nicht",
        "La banque centrale relève les taux pour la cinquième fois dans une semaine",
    ],
)
def test_clearly_foreign_latin_script_text_is_rejected(text: str) -> None:
    assert not is_english(text)


# ---------------------------------------------------------------- de-duplication
def _doc(text: str, domain: str = "reuters.com") -> Document:
    return to_document(RawDocument(source="t", body=text, publisher=domain, external_id=f"{domain}:{text}"))


def test_exact_duplicates_are_dropped_and_count_distinct_sources() -> None:
    text = "Apple posts record quarterly revenue as iPhone demand beats analyst estimates"
    result = Deduper().run([_doc(text, "reuters.com"), _doc(text, "bloomberg.com"), _doc(text, "bloomberg.com")])
    assert len(result.documents) == 1
    assert result.duplicates == 2
    assert result.documents[0].corroboration == 2  # two distinct outlets, not three


def test_near_duplicates_merge_into_corroboration() -> None:
    a = "Federal Reserve signals further interest rate hikes as inflation stays stubbornly high"
    b = a + ", officials say"
    result = Deduper().run([_doc(a, "reuters.com"), _doc(b, "wsj.com")])
    assert len(result.documents) == 1
    assert result.documents[0].corroboration == 2
    assert result.documents[0].source_domain == "reuters.com"  # first seen wins
    assert result.documents[0].merged_hashes == [_doc(b, "wsj.com").content_hash]  # remembered for later runs


def test_distinct_stories_are_kept() -> None:
    result = Deduper().run([_doc("Fed raises interest rates by a quarter point"), _doc("Nvidia unveils a new data center chip")])
    assert len(result.documents) == 2
    assert result.duplicates == 0


def test_previously_loaded_hashes_are_dropped() -> None:
    doc = _doc("Fed raises interest rates by a quarter point")
    result = Deduper(known_hashes={doc.content_hash}).run([doc])
    assert result.documents == []
    assert result.duplicates == 1
