"""Live polling: one asyncio loop per source, pipeline runs serialized through a single lock.

Intervals are per source because each has different limits (GDELT refreshes every ~15 min and
rate-limits hard; NewsAPI's free plan allows 100 requests/day; replay can tick every few seconds).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Mapping

from prism.etl.pipeline import EtlPipeline

log = logging.getLogger(__name__)


class PollingScheduler:
    def __init__(
        self,
        pipeline: EtlPipeline,
        intervals: Mapping[str, float],
        limits: Mapping[str, int] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.intervals = dict(intervals)
        self.limits = dict(limits or {})
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        for source, interval in self.intervals.items():
            self._tasks.append(asyncio.create_task(self._loop(source, interval), name=f"poll-{source}"))
        log.info("scheduler started for: %s", ", ".join(self.intervals) or "no sources")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _loop(self, source: str, interval: float) -> None:
        while True:
            try:
                async with self._lock:
                    await asyncio.to_thread(self.pipeline.run_once, [source], self.limits.get(source))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep polling
                log.exception("poll of %s failed", source)
            await asyncio.sleep(interval)
