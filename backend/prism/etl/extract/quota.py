from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable

from prism.core.errors import SourceUnavailable
from prism.core.interfaces import KeyValueState


class QuotaGuard:
    """Persistent per-day request budget (e.g. NewsAPI's free plan: 100 requests/day)."""

    def __init__(
        self,
        state: KeyValueState,
        name: str,
        daily_limit: int,
        today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
    ) -> None:
        self.state = state
        self.name = name
        self.daily_limit = daily_limit
        self._today = today

    def acquire(self) -> None:
        key = f"quota:{self.name}:{self._today().isoformat()}"
        used = int(self.state.get_state(key) or 0)
        if used >= self.daily_limit:
            raise SourceUnavailable(f"{self.name} daily budget of {self.daily_limit} requests is used up")
        self.state.set_state(key, str(used + 1))
