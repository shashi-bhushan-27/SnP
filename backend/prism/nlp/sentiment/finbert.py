"""FinBERT sentiment (ProsusAI/finbert): BERT further pre-trained on financial text and
fine-tuned on Financial PhraseBank. Output classes: positive / negative / neutral.

    score      = P(positive) - P(negative)            in [-1, 1]
    confidence = max class probability

Heavy dependencies (torch, transformers) are imported lazily, so importing this module is free.
Install with `pip install -r backend/requirements-ml.txt`.
"""

from __future__ import annotations

from typing import Any, Sequence

from prism.core.contracts import SentimentResult


class FinBertSentiment:
    name = "finbert"

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        *,
        batch_size: int = 16,
        max_length: int = 256,
        pipe: Any = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._pipe = pipe  # injectable for tests

    def _classifier(self) -> Any:
        if self._pipe is None:
            from transformers import pipeline  # lazy: keeps base install light

            self._pipe = pipeline("text-classification", model=self.model_name, top_k=None)
        return self._pipe

    def predict(self, texts: Sequence[str]) -> list[SentimentResult]:
        if not texts:
            return []
        outputs = self._classifier()(
            list(texts), batch_size=self.batch_size, truncation=True, max_length=self.max_length
        )
        return [self._to_result(out) for out in outputs]

    @staticmethod
    def _to_result(output: Any) -> SentimentResult:
        entries = [output] if isinstance(output, dict) else output
        probs = {str(e["label"]).lower(): float(e["score"]) for e in entries}
        pos, neg = probs.get("positive", 0.0), probs.get("negative", 0.0)
        return SentimentResult(
            score=max(-1.0, min(1.0, round(pos - neg, 4))),
            probabilities=probs,
            confidence=max(probs.values()) if probs else 0.0,
        )
