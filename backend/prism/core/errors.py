from __future__ import annotations


class PrismError(Exception):
    """Base class for expected, recoverable PRISM errors."""


class SourceUnavailable(PrismError):
    """A source cannot be read right now (rate limit, missing key, network). The pipeline skips it."""
