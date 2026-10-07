#!/usr/bin/env python
"""Compute realised market reactions for the curated analog events and write the analog library.

    python scripts/build_analog_library.py

Input : backend/config/analog_events.yaml        (dates, titles, event types - curated by hand)
Output: backend/config/analog_library.json       (committed, so the app needs no network at runtime)

For each event, with t = first trading session on/after the event date and base = the previous close:
    equity_pct         S&P 500 (^GSPC, Yahoo Finance) close-to-close return
    rate_bps           change in the 10-year Treasury yield (FRED DGS10), basis points
    credit_spread_bps  change in Moody's Baa minus 10-year Treasury (FRED BAA10Y), basis points
over horizons of 1 and 5 trading days, plus "trough": the day of the lowest S&P close within the first
5 sessions, with every factor measured on that same day (a coherent peak-to-trough stress state, the usual
convention for historical scenario replay). FRED values are taken as of each date (holidays forward-filled).
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "backend" / "config"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"
HORIZONS = {"1d": 1, "5d": 5}


def fred_series(series: str, start: date, end: date) -> dict[date, float]:
    response = httpx.get(FRED_URL.format(series=series, start=start, end=end), timeout=60, follow_redirects=True)
    response.raise_for_status()
    values = {}
    for row in csv.DictReader(io.StringIO(response.text)):
        raw = row.get(series, ".")
        if raw not in (".", ""):
            values[date.fromisoformat(row["observation_date"])] = float(raw)
    return values


def as_of(series: dict[date, float], day: date) -> float:
    """Latest observation on or before `day` (FRED has gaps on holidays)."""
    for back in range(0, 10):
        value = series.get(day - timedelta(days=back))
        if value is not None:
            return value
    raise ValueError(f"no observation within 10 days before {day}")


def spx_closes(start: date, end: date) -> list[tuple[date, float]]:
    import yfinance as yf

    frame = yf.download("^GSPC", start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                        progress=False, auto_adjust=False)
    closes = frame["Close"].squeeze()
    return [(ts.date(), float(value)) for ts, value in closes.items()]


def slug(event: dict) -> str:
    words = re.findall(r"[a-z0-9]+", event["title"].lower())[:4]
    return f"{event['date']}-{'-'.join(words)}"


def main() -> None:
    curated = yaml.safe_load((CONFIG / "analog_events.yaml").read_text(encoding="utf-8"))["events"]
    dates = [e["date"] if isinstance(e["date"], date) else date.fromisoformat(e["date"]) for e in curated]
    start, end = min(dates) - timedelta(days=20), max(dates) + timedelta(days=20)

    spx = spx_closes(start, end)
    sessions = [d for d, _ in spx]
    close = dict(spx)
    dgs10 = fred_series("DGS10", start, end)
    baa10y = fred_series("BAA10Y", start, end)

    events, warnings = [], []
    for event, event_date in zip(curated, dates):
        i = next(idx for idx, d in enumerate(sessions) if d >= event_date)
        base_day, reaction_day = sessions[i - 1], sessions[i]
        if reaction_day != event_date:
            warnings.append(f"{event_date}: not a trading day, using {reaction_day}")
        moves = {}
        for name, h in HORIZONS.items():
            end_day = sessions[i + h - 1]
            moves[name] = {
                "equity_pct": round(close[end_day] / close[base_day] - 1, 5),
                "rate_bps": round(100 * (as_of(dgs10, end_day) - as_of(dgs10, base_day)), 1),
                "credit_spread_bps": round(100 * (as_of(baa10y, end_day) - as_of(baa10y, base_day)), 1),
            }
        window = sessions[i : i + 5]
        trough_day = min(window, key=lambda d: close[d])
        moves["trough"] = {
            "equity_pct": round(close[trough_day] / close[base_day] - 1, 5),
            "rate_bps": round(100 * (as_of(dgs10, trough_day) - as_of(dgs10, base_day)), 1),
            "credit_spread_bps": round(100 * (as_of(baa10y, trough_day) - as_of(baa10y, base_day)), 1),
            "days": window.index(trough_day) + 1,
        }
        if abs(moves["1d"]["equity_pct"]) < 0.002 and abs(moves["5d"]["equity_pct"]) < 0.005:
            warnings.append(f"{event_date}: tiny equity reaction ({moves['1d']['equity_pct']:+.2%}) - check the date")
        events.append(
            {
                "id": slug(event),
                "date": event_date.isoformat(),
                "reaction_date": reaction_day.isoformat(),
                "title": event["title"],
                "event_type": event["event_type"],
                "moves": moves,
            }
        )

    library = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "equity": "S&P 500 index (^GSPC) daily close, Yahoo Finance",
            "rates": "10-Year Treasury Constant Maturity Rate (DGS10), FRED",
            "credit": "Moody's Seasoned Baa Corporate Bond Yield Relative to 10-Year Treasury (BAA10Y), FRED",
        },
        "horizons_trading_days": {**HORIZONS, "trough": "lowest S&P close within 5 sessions"},
        "events": events,
    }
    out = CONFIG / "analog_library.json"
    out.write_text(json.dumps(library, indent=1) + "\n", encoding="utf-8")

    print(f"{len(events)} events -> {out.relative_to(REPO_ROOT)}")
    print(f"{'date':<11} {'type':<14} {'eq 1d':>7} | trough: {'eq':>7} {'10y':>5} {'cs':>5} {'day':>3}  title")
    for e in events:
        m1, mt = e["moves"]["1d"], e["moves"]["trough"]
        print(f"{e['reaction_date']:<11} {e['event_type'][:14]:<14} {m1['equity_pct']:>+7.2%} | {mt['equity_pct']:>+15.2%} "
              f"{mt['rate_bps']:>+5.0f} {mt['credit_spread_bps']:>+5.0f} {mt['days']:>3}  {e['title'][:52]}")
    for w in warnings:
        print("WARN", w)


if __name__ == "__main__":
    main()
