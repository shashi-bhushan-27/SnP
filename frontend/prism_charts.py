"""Plotly figure builders. Colors are the validated reference palette (dataviz skill):
single series = blue slot 1; polarity (loss/gain, negative/positive) = red/blue poles with a
neutral gray. Both sets pass the palette validator (light surface, all pairs)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from prism_client import HIGH_IMPACT, money

BLUE = "#2a78d6"  # single series / gains / positive
RED = "#e34948"  # losses / negative
NEUTRAL = "#898781"  # neutral, totals
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
DIRECTION_COLORS = {"negative": RED, "neutral": NEUTRAL, "positive": BLUE}


def _style(fig: go.Figure, height: int = 320) -> go.Figure:
    axis = dict(
        gridcolor=GRID, linecolor=AXIS, zeroline=False, automargin=True,  # automargin: never clip tick labels
        tickfont=dict(color=MUTED), title_font=dict(color=INK_2),
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        margin=dict(l=8, r=16, t=16, b=8),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        bargap=0.35,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def waterfall_figure(rows: pd.DataFrame) -> go.Figure:
    """Portfolio value before -> P&L per asset class -> after (USD millions)."""
    millions = rows["value"] / 1e6
    labels = [money(v) if m != "relative" else ("+" if v > 0 else "") + money(v) for v, m in zip(rows["value"], rows["measure"])]
    fig = go.Figure(
        go.Waterfall(
            x=rows["step"],
            y=millions,
            measure=rows["measure"],
            text=labels,
            textposition="outside",
            textfont=dict(color=INK_2),
            increasing=dict(marker=dict(color=BLUE)),
            decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=NEUTRAL)),
            connector=dict(line=dict(color=AXIS, width=1)),
            hovertemplate="%{x}: %{text}<extra></extra>",
        )
    )
    low = min(rows.loc[rows["measure"] != "relative", "value"].min(), rows["value"].cumsum().min()) / 1e6
    high = rows.loc[rows["measure"] == "absolute", "value"].max() / 1e6
    fig.update_yaxes(title_text="USD millions", range=[low * 0.9, high * 1.04])
    return _style(fig, height=340)


def impact_breakdown_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df["points"],
            y=df["component"],
            orientation="h",
            marker=dict(color=BLUE, cornerradius=4),
            text=[f"{p:.2f}" for p in df["points"]],
            textposition="outside",
            textfont=dict(color=INK_2),
            hovertemplate="%{y}: %{x:.2f} points<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="impact points (sum = score)", range=[0, max(3.5, df["points"].max() * 1.25)])
    return _style(fig, height=260)


def events_figure(df: pd.DataFrame) -> go.Figure:
    counts = df["event_type"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker=dict(color=BLUE, cornerradius=4),
            hovertemplate="%{y}: %{x} signals<extra></extra>",
        )
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(title_text="signals", dtick=1 if len(counts) and counts.max() <= 10 else None)
    return _style(fig, height=max(240, 28 * len(counts) + 60))


def risk_timeline_figure(df: pd.DataFrame) -> go.Figure:
    """Impact over time; color = sentiment direction (legend + hover carry the same information)."""
    fig = go.Figure()
    for name in ("negative", "neutral", "positive"):
        part = df[df["direction"] == name]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["time"],
                y=part["impact_score"],
                mode="markers",
                name=f"{name} sentiment",
                marker=dict(size=11, color=DIRECTION_COLORS[name], line=dict(color=SURFACE, width=2)),
                customdata=part[["who", "event_type", "sentiment_score", "headline"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>impact %{y:.2f} · sentiment %{customdata[2]:+.2f}"
                    "<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=HIGH_IMPACT, line=dict(color=AXIS, width=1))
    fig.add_annotation(
        xref="paper", x=0, y=HIGH_IMPACT, yshift=10, text="stress trigger threshold", showarrow=False,
        font=dict(color=MUTED, size=11), xanchor="left",
    )
    fig.update_yaxes(title_text="impact score", range=[0.5, 10.5])
    return _style(fig, height=320)


def distribution_figure(outcomes: pd.DataFrame, basis_id: str, expected_pnl: float) -> go.Figure:
    """P&L today of every past event of this kind (worst left); the stress basis in red, the average as a line."""
    colors = [RED if i == basis_id else NEUTRAL for i in outcomes["id"]]
    fig = go.Figure(
        go.Bar(
            x=list(range(1, len(outcomes) + 1)),
            y=outcomes["P&L today"] / 1e6,
            marker=dict(color=colors, cornerradius=3),
            customdata=outcomes[["date", "event"]].values,
            hovertemplate="%{customdata[1]}<br>%{customdata[0]}<br>P&L today: $%{y:.2f}M<extra></extra>",
        )
    )
    fig.add_hline(y=expected_pnl / 1e6, line=dict(color=INK_2, width=1))
    fig.add_annotation(xref="paper", x=1, y=expected_pnl / 1e6, yshift=10, xanchor="right", showarrow=False,
                       text="average past outcome", font=dict(color=MUTED, size=11))
    fig.update_xaxes(title_text="past events of this kind, worst to best", showticklabels=False, showgrid=False)
    fig.update_yaxes(title_text="P&L on today's book (USD M)")
    return _style(fig, height=280)
