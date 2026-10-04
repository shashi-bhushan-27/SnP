"""HTTP client for the PRISM API plus pure data shaping for the dashboard (no Streamlit in here)."""

from __future__ import annotations

from typing import Any, Sequence

import httpx
import pandas as pd

NEGATIVE_BAND = -0.2
POSITIVE_BAND = 0.2
HIGH_IMPACT = 7.0

IMPACT_LABELS = {
    "severity": "Event severity",
    "sentiment_magnitude": "Sentiment strength",
    "exposure": "Exposure",
    "source_reliability": "Source reliability",
    "corroboration": "Corroboration",
}
ASSET_TYPE_LABELS = {"loan": "Loans", "bond": "Bonds", "equity": "Equities", "derivative": "Derivatives"}


class ApiError(RuntimeError):
    pass


class PrismClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", client: httpx.Client | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the PRISM API at {self.base_url} ({exc.__class__.__name__})") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{method} {path} failed ({response.status_code}): {detail}")
        return response.json()

    # ----------------------------------------------------------------- reads
    def health(self) -> dict:
        return self._call("GET", "/health")

    def stats(self) -> dict:
        return self._call("GET", "/api/stats")

    def signals(self, **filters: Any) -> list[dict]:
        params = {k: v for k, v in filters.items() if v not in (None, "", [])}
        return self._call("GET", "/api/risk-signals", params=params)

    def portfolio(self) -> dict:
        return self._call("GET", "/api/portfolio")

    def scenarios(self) -> list[dict]:
        return self._call("GET", "/api/scenarios")

    def stress_runs(self, limit: int = 20) -> list[dict]:
        return self._call("GET", "/api/stress-runs", params={"limit": limit})

    def runs(self, limit: int = 10) -> list[dict]:
        return self._call("GET", "/api/pipeline/runs", params={"limit": limit})

    def rejects(self, limit: int = 50) -> list[dict]:
        return self._call("GET", "/api/pipeline/rejects", params={"limit": limit})

    # ----------------------------------------------------------------- actions
    def run_pipeline(self, sources: Sequence[str] | None = None, limit: int | None = None) -> dict:
        return self._call("POST", "/api/pipeline/run", json={"sources": list(sources) if sources else None, "limit": limit})

    def stress_test(self, event_type: str, impact_score: float | None = None, persist: bool = False) -> dict:
        body = {"event_type": event_type, "impact_score": impact_score, "persist": persist}
        return self._call("POST", "/api/stress-test", json=body)

    def analyze(self, texts: Sequence[str], persist: bool = False) -> dict:
        return self._call("POST", "/api/analyze", json={"texts": list(texts), "persist": persist})


# --------------------------------------------------------------------- shaping
def direction(score: float) -> str:
    if score <= NEGATIVE_BAND:
        return "negative"
    if score >= POSITIVE_BAND:
        return "positive"
    return "neutral"


def signals_frame(signals: Sequence[dict]) -> pd.DataFrame:
    columns = [
        "time", "who", "event_type", "sentiment_score", "impact_score", "confidence", "headline",
        "source", "scope", "direction", "high_risk", "id",
    ]
    if not signals:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(signals)
    df["time"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["who"] = df["ticker"].fillna(df["entity"]).fillna("MARKET")
    df["direction"] = df["sentiment_score"].map(direction)
    df["high_risk"] = (df["impact_score"] >= HIGH_IMPACT) & (df["sentiment_score"] <= NEGATIVE_BAND)
    return df.sort_values(["time", "impact_score"], ascending=[False, False])[columns + ["explanation"]].reset_index(drop=True)


def impact_breakdown(signal: dict) -> pd.DataFrame:
    """Impact = 1.0 base + per-feature points; rows in a fixed order so bars never reshuffle."""
    points = (signal.get("explanation") or {}).get("impact_points", {})
    rows = [("Base", 1.0)] + [(label, float(points.get(key, 0.0))) for key, label in IMPACT_LABELS.items()]
    return pd.DataFrame(rows, columns=["component", "points"])


def waterfall_rows(result: dict) -> pd.DataFrame:
    """Before -> change per asset class -> after, in a fixed asset-class order."""
    rows = [("Before", result["value_before"], "absolute")]
    for key, label in ASSET_TYPE_LABELS.items():
        if key in result.get("by_type", {}):
            rows.append((label, result["by_type"][key], "relative"))
    rows.append(("After", result["value_after"], "total"))
    return pd.DataFrame(rows, columns=["step", "value", "measure"])


def asset_table(result: dict) -> pd.DataFrame:
    df = pd.DataFrame(result.get("by_asset", []))
    if df.empty:
        return df
    df["asset_type"] = df["asset_type"].map(ASSET_TYPE_LABELS).fillna(df["asset_type"])
    return df[["name", "asset_type", "value_before", "pnl", "pnl_pct", "value_after"]].sort_values("pnl")


def money(value: float, digits: int = 2) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.{digits}f}B"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.{digits}f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.0f}K"
    return f"{sign}${value:.0f}"
