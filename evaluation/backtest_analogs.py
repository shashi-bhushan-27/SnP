#!/usr/bin/env python
"""Back-test the historical-analog scenario engine against the alternatives.

    python evaluation/backtest_analogs.py [--horizon trough|1d|5d]

Question: when a new event arrives, which method best predicts the market reaction that actually followed -
and therefore the portfolio P&L a stress test should show?

Methods (for each held-out event, using only the other events):
    static_matrix   the hand-written shock for the event type (config/scenarios.yaml), zero if none
    global_mean     average reaction to all other events
    type_mean       average reaction to other events of the same type (falls back to global_mean)
    analog          nearest past events by headline meaning (bge-small), softmax-weighted (the PRISM method)
    analog_hashing  same retrieval with a bag-of-words encoder instead of sentence embeddings

Protocols:
    leave-one-out   every other event is available
    time-respecting only events BEFORE the held-out one (what a live system would have known); scored on
                    events with at least 15 predecessors

Metrics: mean absolute error of equity (pp), 10y yield (bp), Baa spread (bp), sign accuracy of the yield move,
and the error of the resulting portfolio P&L on the default synthetic book (the number a risk manager acts on),
plus the Spearman rank correlation between predicted and realised P&L.

Writes evaluation/results/analog_backtest.{md,json}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402

from prism.config import Settings  # noqa: E402
from prism.modules.stress import Portfolio, ScenarioBook, Shock, StressEngine  # noqa: E402
from prism.modules.stress.analogs import AnalogLibrary  # noqa: E402
from prism.nlp.embeddings import HashingEncoder, SentenceEncoder  # noqa: E402

MIN_HISTORY = 15
PURGE_DAYS = 10  # events this close to the held-out one share its market window -> excluded (purged CV)


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run(horizon: str) -> dict:
    settings = Settings(_env_file=None)
    cfg = settings.config_dir
    portfolio = Portfolio.from_yaml(cfg / "portfolio.yaml")
    book = ScenarioBook.from_yaml(cfg / "scenarios.yaml")
    engine = StressEngine()
    any_scenario = book.all()[0]

    def pnl(shock: Shock) -> float:
        return engine.run(portfolio, any_scenario, shock=shock).pnl

    path = cfg / "analog_library.json"
    analog = AnalogLibrary.from_json(path, SentenceEncoder(settings.embedding_model), horizon=horizon)
    hashing = AnalogLibrary.from_json(path, HashingEncoder(), horizon=horizon, min_similarity=0.0)
    events = analog.events

    def purged(i: int) -> set[str]:
        return {x.id for x in events if abs((x.reaction_date - events[i].reaction_date).days) <= PURGE_DAYS}

    def predict(method: str, i: int, before) -> Shock:
        e = events[i]
        exclude = purged(i)
        if method == "static_matrix":
            scenario = book.for_event(e.event_type)
            return scenario.shock if scenario else Shock()
        if method == "global_mean":
            return analog.global_mean(exclude=exclude, before=before)
        if method == "type_mean":
            return analog.type_mean(e.event_type, exclude=exclude, before=before) or analog.global_mean(exclude=exclude, before=before)
        lib = {"analog": analog, "analog_hashing": hashing, "analog_soft": soft}[method]
        return lib.scenario(e.title, e.event_type, exclude=exclude, before=before).shock

    methods = ["static_matrix", "global_mean", "type_mean", "analog_hashing", "analog", "analog_soft"]
    soft = AnalogLibrary.from_json(path, analog.encoder, horizon=horizon, k=50, temperature=0.10,
                                   same_type_bonus=0.15, min_similarity=0.0)
    soft._matrix = analog._embeddings()
    report = {"horizon": horizon, "n_events": len(events), "protocols": {}}
    for protocol in ("leave_one_out", "time_respecting"):
        rows = []
        for i, e in enumerate(events):
            before = e.reaction_date if protocol == "time_respecting" else None
            if before is not None and sum(x.reaction_date < before for x in events) < MIN_HISTORY:
                continue
            actual = e.moves[horizon]
            rows.append((i, actual, {m: predict(m, i, before) for m in methods}))

        results = {}
        for m in methods:
            eq = [abs(p[m].equity_pct - a.equity_pct) * 100 for _, a, p in rows]
            rt = [abs(p[m].rate_bps - a.rate_bps) for _, a, p in rows]
            cs = [abs(p[m].credit_spread_bps - a.credit_spread_bps) for _, a, p in rows]
            moved = [(p[m].rate_bps, a.rate_bps) for _, a, p in rows if abs(a.rate_bps) >= 5]
            pred_pnl = [pnl(p[m]) for _, _, p in rows]
            real_pnl = [pnl(a) for _, a, _ in rows]
            results[m] = {
                "equity_mae_pp": float(np.mean(eq)),
                "rate_mae_bp": float(np.mean(rt)),
                "credit_mae_bp": float(np.mean(cs)),
                "rate_sign_accuracy": float(np.mean([np.sign(p) == np.sign(a) for p, a in moved])) if moved else None,
                "pnl_mae_usd": float(np.mean(np.abs(np.array(pred_pnl) - np.array(real_pnl)))),
                "pnl_spearman": spearman(pred_pnl, real_pnl) if m != "global_mean" and len(set(pred_pnl)) > 1 else None,  # LOO global mean is anti-correlated by construction
            }
        report["protocols"][protocol] = {"n_scored": len(rows), "results": results}

        # stress severity: does the scenario cover the realised loss, and by how much does it over-reserve?
        coverage = {}
        for label, q in (("history q20 (same type)", 0.20), ("history q10 (same type)", 0.10), ("history q10 (all events)", 0.10)):
            hits, excess = [], []
            for i, actual, _ in rows:
                e = events[i]
                before = e.reaction_date if protocol == "time_respecting" else None
                pool = [x for x in events if x.id not in purged(i) and (before is None or x.reaction_date < before)]
                if "same type" in label:
                    same = [x for x in pool if x.event_type == e.event_type]
                    pool = same if len(same) >= 5 else pool
                stress_pnl = float(np.quantile([pnl(x.moves[horizon]) for x in pool], q))
                real = pnl(actual)
                hits.append(real >= stress_pnl)
                excess.append(real - stress_pnl)
            coverage[label] = {"coverage": float(np.mean(hits)), "mean_excess_usd": float(np.mean(excess))}
        hits = [pnl(a) >= pnl(p["static_matrix"]) for _, a, p in rows]
        excess = [pnl(a) - pnl(p["static_matrix"]) for _, a, p in rows]
        coverage["hand-written matrix"] = {"coverage": float(np.mean(hits)), "mean_excess_usd": float(np.mean(excess))}
        report["protocols"][protocol]["coverage"] = coverage

    by_type = {}
    for t in sorted({e.event_type for e in events}, key=lambda t: t.value):
        group = [e.moves[horizon] for e in events if e.event_type == t]
        static = book.for_event(t)
        by_type[t.value] = {
            "n": len(group),
            "median_equity_pct": float(np.median([g.equity_pct for g in group])),
            "median_rate_bps": float(np.median([g.rate_bps for g in group])),
            "share_rate_down": float(np.mean([g.rate_bps < 0 for g in group])),
            "median_credit_bps": float(np.median([g.credit_spread_bps for g in group])),
            "matrix": static.shock.model_dump() if static else None,
        }
    report["by_type"] = by_type
    return report


def to_markdown(reports: list[dict]) -> str:
    out = [
        "# Historical-analog scenario back-test\n",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `evaluation/backtest_analogs.py` "
        f"on {reports[0]['n_events']} curated events (2008-2025, `backend/config/analog_library.json`).\n",
        "Each method predicts the held-out event's market reaction; errors are against what actually happened. "
        "P&L error = |P&L of the predicted shock - P&L of the realised shock| on the default $60M synthetic book.\n",
    ]
    names = {
        "static_matrix": "Hand-written matrix (scenarios.yaml)",
        "global_mean": "Average of all past events",
        "type_mean": "Average of same event type",
        "analog_hashing": "Analog retrieval, bag-of-words",
        "analog": "Analog retrieval, sentence embeddings (top-5)",
        "analog_soft": "Analog-weighted history (all events, type bonus)",
    }
    for rep in reports:
        for protocol, block in rep["protocols"].items():
            out += [
                f"\n## Horizon `{rep['horizon']}` - {protocol.replace('_', '-')} ({block['n_scored']} events scored)\n",
                "| Method | P&L error | P&L rank corr. | Equity MAE | 10y MAE | Spread MAE | 10y sign correct |",
                "|---|---|---|---|---|---|---|",
            ]
            for m, r in block["results"].items():
                sign = f"{100 * r['rate_sign_accuracy']:.0f}%" if r["rate_sign_accuracy"] is not None else "-"
                corr = f"{r['pnl_spearman']:.2f}" if r["pnl_spearman"] is not None else "-"
                out.append(
                    f"| {names[m]} | ${r['pnl_mae_usd'] / 1e6:.2f}M | {corr} | {r['equity_mae_pp']:.2f} pp | "
                    f"{r['rate_mae_bp']:.1f} bp | {r['credit_mae_bp']:.1f} bp | {sign} |"
                )
            out += [
                f"\n### Stress severity - horizon `{rep['horizon']}`, {protocol.replace('_', '-')}\n",
                "Coverage = share of events whose real loss was no worse than the scenario's; excess = how much more "
                "severe than reality the scenario was on average (capital held against losses that never came).\n",
                "| Scenario rule | Coverage | Mean excess severity |",
                "|---|---|---|",
            ]
            for label, c in block["coverage"].items():
                out.append(f"| {label} | {100 * c['coverage']:.0f}% | ${c['mean_excess_usd'] / 1e6:.2f}M |")
        out += [
            f"\n### What history says vs the hand-written matrix - horizon `{rep['horizon']}`\n",
            "| Event type | Events | Median equity | Median 10y | 10y fell | Median Baa spread | Matrix (equity / 10y / spread) |",
            "|---|---|---|---|---|---|---|",
        ]
        for t, b in rep["by_type"].items():
            m = b["matrix"]
            mtx = f"{100 * m['equity_pct']:+.0f}% / {m['rate_bps']:+.0f} bp / {m['credit_spread_bps']:+.0f} bp" if m else "none"
            out.append(
                f"| {t} | {b['n']} | {100 * b['median_equity_pct']:+.1f}% | {b['median_rate_bps']:+.0f} bp | "
                f"{100 * b['share_rate_down']:.0f}% | {b['median_credit_bps']:+.0f} bp | {mtx} |"
            )
    out.append(
        f"\nCaveats: {reports[0]['n_events']} events is a small sample, so differences of a few percent are noise; neighbours within 10 days of a held-out event are purged because they share its market window; the library was curated by hand "
        "(selection bias toward famous events); titles are written outcome-free so the held-out headline does not reveal its "
        "own market reaction. Sign accuracy counts only events where the 10y yield moved at least 5 bp."
    )
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", choices=["trough", "1d", "5d"], nargs="*", default=["trough", "1d"])
    args = parser.parse_args()
    reports = [run(h) for h in args.horizon]
    results = REPO_ROOT / "evaluation" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "analog_backtest.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    markdown = to_markdown(reports)
    (results / "analog_backtest.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
