"""Zero-shot event classification by nearest prototype in sentence-embedding space.

Why not an NLI zero-shot pipeline: recent benchmarks (BTZSC, 2026) find embedding models give the
best accuracy/latency trade-off and NLI cross-encoders plateau, and an NLI model needs one forward
pass per (text, label) pair. Here each text is encoded once and compared with the prototype
sentences in config/event_prototypes.yaml - editable without retraining.

sentence-transformers is imported lazily (requirements-ml.txt); an `encoder` can be injected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from prism.core.contracts import EventResult
from prism.core.taxonomy import EventType


class EmbeddingEventClassifier:
    name = "embedding"

    def __init__(
        self,
        prototypes: Mapping[EventType, Sequence[str]],
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        temperature: float = 0.05,
        encoder: Any = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self._encoder = encoder  # anything with .encode(list[str], normalize_embeddings=True)
        self._classes = list(prototypes)
        self._sentences = [s for event in self._classes for s in prototypes[event]]
        self._slices: list[np.ndarray] = []
        start = 0
        for event in self._classes:
            n = len(prototypes[event])
            self._slices.append(np.arange(start, start + n))
            start += n
        self._matrix: np.ndarray | None = None

    @classmethod
    def from_yaml(cls, path: Path | str, **kwargs: Any) -> EmbeddingEventClassifier:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls({EventType.parse(key): sentences for key, sentences in raw.items()}, **kwargs)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._encoder = SentenceTransformer(self.model_name)
        return np.asarray(self._encoder.encode(list(texts), normalize_embeddings=True), dtype=float)

    def classify(self, texts: Sequence[str]) -> list[EventResult]:
        if not texts:
            return []
        if self._matrix is None:
            self._matrix = self._encode(self._sentences)
        sims = self._encode(texts) @ self._matrix.T  # cosine similarity (embeddings are normalised)

        results = []
        for row in sims:
            per_class = np.array([row[idx].max() for idx in self._slices])
            z = per_class / self.temperature
            probs = np.exp(z - z.max())
            probs /= probs.sum()
            top = int(probs.argmax())
            results.append(
                EventResult(
                    event_type=self._classes[top],
                    confidence=float(probs[top]),
                    scores={event.value: float(p) for event, p in zip(self._classes, probs)},
                    method=self.name,
                )
            )
        return results
