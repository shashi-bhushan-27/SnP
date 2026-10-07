"""Text encoders (TextEncoder protocol).

    SentenceEncoder  sentence-transformers model (default BAAI/bge-small-en-v1.5, 33M params); lazy import
    HashingEncoder   dependency-free hashed TF bag of unigrams + bigrams; a weak but deterministic fallback
                     used when the ML extras are not installed (tests, CI)
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Any, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("the a an and or of to in on for with as at by from is are was were be its it this that".split())


class HashingEncoder:
    name = "hashing"

    def __init__(self, dims: int = 4096) -> None:
        self.dims = dims

    def _features(self, text: str) -> Counter:
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return Counter(zlib.crc32(g.encode("utf-8")) % self.dims for g in grams)

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True) -> np.ndarray:
        out = np.zeros((len(texts), self.dims))
        for row, text in enumerate(texts):
            for index, count in self._features(text).items():
                out[row, index] = 1.0 + math.log(count)
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out /= np.where(norms == 0, 1.0, norms)
        return out


class SentenceEncoder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", model: Any = None) -> None:
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self._model = model

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=normalize_embeddings), dtype=float)
