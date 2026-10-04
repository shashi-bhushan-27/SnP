"""Command line entry points.

    python -m prism.cli run --sources replay --limit 25      one ETL run
    python -m prism.cli analyze "Fed signals more rate hikes"  Risk Engine on ad-hoc text
    python -m prism.cli stress --event Geopolitical --impact 9  stress test on demand
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from prism.bootstrap import build_container, build_engine
from prism.config import get_settings
from prism.core.contracts import RawDocument
from prism.core.taxonomy import EventType
from prism.etl.transform import normalize_batch


def _print(obj) -> None:
    payload = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
    print(json.dumps(payload, indent=2, default=str))


def cmd_run(args: argparse.Namespace) -> None:
    container = build_container()
    for i in range(args.repeat):
        if i:
            time.sleep(args.interval)
        report = container.pipeline.run_once(args.sources or None, args.limit)
        _print(report)


def cmd_analyze(args: argparse.Namespace) -> None:
    engine = build_engine(get_settings())
    docs, rejected = normalize_batch(RawDocument(source="cli", body=t) for t in args.texts)
    _print({"signals": [s.model_dump(mode="json") for s in engine.analyze(docs)], "rejected": [r.model_dump() for r in rejected]})


def cmd_stress(args: argparse.Namespace) -> None:
    container = build_container()
    scenario = container.scenarios.for_event(EventType.parse(args.event))
    if scenario is None:
        sys.exit(f"No scenario for event type {args.event!r}")
    _print(container.stress_engine.run(container.portfolio, scenario, impact_score=args.impact))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="prism")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the ETL pipeline")
    run.add_argument("--sources", nargs="*", help="subset of enabled sources")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--repeat", type=int, default=1, help="number of runs")
    run.add_argument("--interval", type=float, default=5.0, help="seconds between repeated runs")
    run.set_defaults(func=cmd_run)

    analyze = sub.add_parser("analyze", help="analyze text with the Risk Engine")
    analyze.add_argument("texts", nargs="+")
    analyze.set_defaults(func=cmd_analyze)

    stress = sub.add_parser("stress", help="run a stress test")
    stress.add_argument("--event", required=True)
    stress.add_argument("--impact", type=float, default=None)
    stress.set_defaults(func=cmd_stress)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
