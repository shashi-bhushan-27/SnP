from __future__ import annotations

import hashlib


def stable_id(*parts: object, length: int = 16) -> str:
    """Deterministic short id: the same inputs always give the same id (idempotent loads)."""
    digest = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return digest[:length]
