from __future__ import annotations

import re

# Distinctive function words per language. Headlines are terse, so we never *require* English
# stopwords; we only reject text where another language clearly outscores English.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the and of to in for with as at by from that this is are was were has have will after over says said amid on".split()
    ),
    "es": frozenset("el la los las del que por con para una es su como mas sube".split()),
    "fr": frozenset("le la les des du et est pour dans avec sur une qui que au aux ce".split()),
    "de": frozenset("der die das und ist nicht mit fur von zu den dem ein eine auf im zum zur sich auch".split()),
    "it": frozenset("il lo gli della delle dei per con che di sono piu non".split()),
    "pt": frozenset("os um uma do da dos das em para com nao que mais".split()),
}
_TOKEN_RE = re.compile(r"[a-z']+")


def is_english(text: str, hint: str | None = None) -> bool:
    """Cheap English check. A provider-supplied language hint wins; otherwise: reject non-Latin
    scripts, and Latin-script text where another language's stopwords clearly outnumber English.

    Placeholder for a real detector (fastText / langdetect) if non-English noise shows up.
    """
    if hint:
        return hint.strip().lower().startswith("en")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < 4:
        return True  # too little evidence to call it foreign
    scores = {lang: sum(t in words for t in tokens) for lang, words in _STOPWORDS.items()}
    best_other = max(score for lang, score in scores.items() if lang != "en")
    return not (best_other >= 2 and best_other > scores["en"])
