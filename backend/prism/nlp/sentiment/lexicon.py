"""Small finance lexicon baseline.

Two jobs: (1) a zero-download fallback so the system runs anywhere (CI, first run, offline), and
(2) the ablation baseline that FinBERT must beat in the evaluation. It is deliberately simple.
Output matches FinBERT's contract: score = P(positive) - P(negative).
"""

from __future__ import annotations

import re
from typing import Sequence

from prism.core.contracts import SentimentResult

POSITIVE = frozenset(
    """beat beats surge surges surged soar soars soared jump jumps jumped rally rallies gain gains rise rises
    rose record upgrade upgrades upgraded outperform growth profit profits strong boost boosts expands expansion
    approval approved wins win raises raised optimistic rebound recovery bullish breakthrough partnership
    dividend buyback discovery""".split()
)
NEGATIVE = frozenset(
    """miss misses missed plunge plunges plunged tumble tumbles tumbled fall falls fell drop drops dropped
    slump slumps sink sinks crash downgrade downgrades downgraded underperform loss losses weak cut cuts layoffs
    lawsuit sues sued probe investigation fraud default defaults bankruptcy bankrupt recall warns warning
    fear fears selloff slowdown recession sanctions tariffs tensions escalate escalating rattled rattling
    breach hack hacked ransomware outage shortage delay delays halts fine fined bearish concern concerns
    turmoil crisis collapse junk inflation unemployment volatility""".split()
)
_NEGATORS = frozenset({"not", "no", "never", "without", "fails", "failed", "unable"})
_TOKEN_RE = re.compile(r"[a-z']+")


class LexiconSentiment:
    name = "lexicon"

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> SentimentResult:
        tokens = _TOKEN_RE.findall(text.lower().replace("-", ""))
        pos = neg = 0
        for i, token in enumerate(tokens):
            polarity = 1 if token in POSITIVE else -1 if token in NEGATIVE else 0
            if polarity and any(t in _NEGATORS for t in tokens[max(0, i - 2) : i]):
                polarity = -polarity
            if polarity > 0:
                pos += 1
            elif polarity < 0:
                neg += 1

        total = pos + neg
        denom = total + 1.0
        p_pos, p_neg, p_neu = pos / denom, neg / denom, 1.0 / denom
        # no lexicon hit means "unknown", not "confidently neutral"
        confidence = 0.4 if total == 0 else min(0.9, 0.5 + 0.1 * total)
        return SentimentResult(
            score=round(p_pos - p_neg, 4),
            probabilities={"positive": p_pos, "negative": p_neg, "neutral": p_neu},
            confidence=confidence,
        )
