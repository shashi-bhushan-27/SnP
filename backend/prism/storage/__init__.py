"""Persistence (the Load target). SQLite by default, PostgreSQL via DATABASE_URL.

Depends only on prism.core: downstream modules hand results over as plain dicts, so storage never
imports them.
"""

from prism.storage.store import SqlStore

__all__ = ["SqlStore"]
