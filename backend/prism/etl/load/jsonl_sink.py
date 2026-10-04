from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from prism.core.contracts import RiskSignal


class JsonlSignalSink:
    """File output for the engine: appends every new signal as one JSON line.

    Implements SignalConsumer, so it is wired exactly like any downstream module.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def on_signals(self, signals: Sequence[RiskSignal]) -> None:
        if not signals:
            return
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            for signal in signals:
                fh.write(signal.model_dump_json() + "\n")
