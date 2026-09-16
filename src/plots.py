"""Plotly figure builders for the dashboard.

Design rules kept consistently across every chart:

* One y-axis per chart -- never two scales on one plot.  CTL, ATL and TSB share
  a chart only because they are all in the same load-point unit.
* Colors carry identity, not magnitude: a fixed, colorblind-checked categorical
  order (blue, orange, aqua) assigned per series, never cycled or re-ranked.
* Thin marks, solid hairline gridlines, recessive axes, and direct labels on the
  endpoint that matters instead of a number on every point.
* Pace axes are reversed and ticked in ``m:ss`` so "higher" always means faster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.consistency import SWEET_SPOT
from src.performance import format_pace

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = {"fitness": "#2a78d6", "fatigue": "#eb6834", "form": "#1baf7a"}
GOOD = "#0ca30c"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base(
    fig: go.Figure, height: int = 380, y_title: str = "", x_title: str = "", right: int = 16
) -> go.Figure:
    """Apply the shared chrome: surface, hairline grid, recessive axes.

    The top margin leaves room for the title *and* a horizontal legend beneath
    it, so the two never collide; ``right`` widens the gutter on charts that
    carry endpoint labels.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        margin=dict(l=8, r=right, t=76, b=8),
        hovermode="x unified",
        title=dict(x=0, xanchor="left", y=0.97, yanchor="top",
                   font=dict(size=15, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    )
    fig.update_xaxes(
        title_text=x_title, showgrid=False, showline=True, linecolor=AXIS,
        linewidth=1, ticks="outside", tickcolor=AXIS, tickfont=dict(color=MUTED),
    )
    fig.update_yaxes(
        title_text=y_title, gridcolor=GRID, gridwidth=1, griddash="solid",
        zeroline=False, showline=False, tickfont=dict(color=MUTED),
    )
    return fig


def _pace_axis(fig: go.Figure, paces: pd.Series) -> None:
    """Reverse the y-axis and tick it every 15 s of pace, labelled ``m:ss``."""
    finite = pd.Series(paces).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return
    low = np.floor(finite.min() * 4) / 4
    high = np.ceil(finite.max() * 4) / 4
    ticks = np.arange(low, high + 0.25, 0.25)
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=ticks,
        ticktext=[format_pace(t).replace(" /km", "") for t in ticks],
    )


def _gutter_labels(fig: go.Figure, entries: list[tuple[float, str, str]], span: float) -> None:
    """Label each series' latest value in the right gutter.

    Labels are pinned to the paper edge (so they never stretch the x-range) and
    nudged apart when two series converge.  They are also the accessibility
    relief for the lower-contrast series hue: identity is never color alone.
    """
    entries = sorted([e for e in entries if np.isfinite(e[0])], key=lambda e: -e[0])
    min_gap = 0.075 * span if span > 0 else 1.0
    placed: list[float] = []
    for value, text, color in entries:
        y = value if not placed else min(value, placed[-1] - min_gap)
        placed.append(y)
        fig.add_annotation(
            xref="paper", x=1.005, y=y, text=f"<b>{text}</b>", showarrow=False,
            xanchor="left", font=dict(color=color, size=12), bgcolor=SURFACE, borderpad=2,
        )


def fitness_freshness_figure(ff: pd.DataFrame) -> go.Figure:
    """Fitness (CTL), Fatigue (ATL) and Form (TSB) -- all in load points."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ff["date"], y=ff["ctl"], name="Fitness (42-day)", mode="lines",
        line=dict(color=SERIES["fitness"], width=2), fill="tozeroy",
        fillcolor="rgba(42,120,214,0.08)", hovertemplate="Fitness %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ff["date"], y=ff["atl"], name="Fatigue (7-day)", mode="lines",
        line=dict(color=SERIES["fatigue"], width=2),
        hovertemplate="Fatigue %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ff["date"], y=ff["tsb"], name="Form (Fitness - Fatigue)", mode="lines",
        line=dict(color=SERIES["form"], width=2),
        hovertemplate="Form %{y:+.1f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)

    if not ff.empty:
        last = ff.iloc[-1]
        span = float(ff[["ctl", "atl", "tsb"]].to_numpy().max()
                     - ff[["ctl", "atl", "tsb"]].to_numpy().min())
        _gutter_labels(fig, [
            (float(last["ctl"]), f"Fitness {last['ctl']:.0f}", SERIES["fitness"]),
            (float(last["atl"]), f"Fatigue {last['atl']:.0f}", SERIES["fatigue"]),
            (float(last["tsb"]), f"Form {last['tsb']:+.0f}", SERIES["form"]),
        ], span)
    fig.update_layout(title="Fitness &amp; Freshness")
    return _base(fig, height=420, y_title="Load points", right=96)


def pace_distance_figure(activities: pd.DataFrame, frontier: pd.DataFrame) -> go.Figure:
    """Every run as distance vs pace (log x), with the best-effort frontier lit up."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=activities["distance_km"], y=activities["pace_min_per_km"], name="Every run",
        mode="markers", marker=dict(color=MUTED, size=8, opacity=0.55,
                                    line=dict(width=2, color=SURFACE)),
        customdata=activities["date"].dt.strftime("%d %b %Y"),
        hovertemplate="%{customdata}<br>%{x:.1f} km at %{y:.2f} min/km<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frontier["distance_km"], y=frontier["pace_min_per_km"], name="Best-effort frontier",
        mode="lines+markers", line=dict(color=SERIES["fitness"], width=2),
        marker=dict(color=SERIES["fitness"], size=10, line=dict(width=2, color=SURFACE)),
        customdata=frontier["date"].dt.strftime("%d %b %Y"),
        hovertemplate="%{customdata}<br>%{x:.1f} km at %{y:.2f} min/km<extra></extra>",
    ))
    fig.update_layout(title="Pace vs distance", hovermode="closest")
    _base(fig, height=420, y_title="Pace (min/km)", x_title="Distance (km)")
    fig.update_xaxes(type="log", tickvals=[3, 5, 10, 15, 21, 30, 42],
                     ticktext=["3", "5", "10", "15", "21", "30", "42"])
    _pace_axis(fig, activities["pace_min_per_km"])
    return fig


def _volume_figure(frame: pd.DataFrame, x: str, title: str, hover_fmt: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=frame[x], y=frame["distance_km"], marker=dict(color=SERIES["fitness"], cornerradius=4),
        customdata=frame["runs"],
        hovertemplate=f"{hover_fmt}<br>%{{y:.1f}} km over %{{customdata}} runs<extra></extra>",
    ))
    fig.update_layout(title=title, bargap=0.25)
    return _base(fig, height=340, y_title="Distance (km)")


def weekly_volume_figure(weekly: pd.DataFrame) -> go.Figure:
    """Distance per calendar week; empty weeks stay visible as gaps."""
    return _volume_figure(weekly, "week_start", "Weekly distance", "Week of %{x|%d %b %Y}")


def monthly_volume_figure(monthly: pd.DataFrame) -> go.Figure:
    """Distance per calendar month."""
    return _volume_figure(monthly, "month_start", "Monthly distance", "%{x|%b %Y}")


def acwr_figure(acwr: pd.DataFrame) -> go.Figure:
    """Acute:chronic workload ratio with the 0.8-1.3 sweet spot shaded."""
    low, high = SWEET_SPOT
    fig = go.Figure()
    fig.add_hrect(y0=low, y1=high, fillcolor=GOOD, opacity=0.10, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=acwr["date"], y=acwr["acwr"], name="ACWR", mode="lines",
        line=dict(color=SERIES["fitness"], width=2), connectgaps=False,
        hovertemplate="ACWR %{y:.2f}<extra></extra>",
    ))
    fig.add_annotation(
        x=acwr["date"].min() if not acwr.empty else 0, y=high, yanchor="bottom", xanchor="left",
        text=f"sweet spot {low:g}-{high:g}", showarrow=False,
        font=dict(color=GOOD, size=11), bgcolor=SURFACE, borderpad=2,
    )
    fig.update_layout(title="Acute:chronic workload ratio", showlegend=False)
    return _base(fig, height=340, y_title="7-day load / 28-day load")


def rolling_pace_figure(rolling: pd.DataFrame, window_days: int = 30) -> go.Figure:
    """Distance-weighted rolling average pace -- the "am I getting faster?" chart."""
    fig = go.Figure(go.Scatter(
        x=rolling["date"], y=rolling["rolling_pace"], mode="lines",
        line=dict(color=SERIES["fitness"], width=2), name="Rolling pace",
        hovertemplate="%{y:.2f} min/km<extra></extra>",
    ))
    fig.update_layout(title=f"Rolling {window_days}-day average pace", showlegend=False)
    _base(fig, height=340, y_title="Pace (min/km)")
    _pace_axis(fig, rolling["rolling_pace"])
    return fig


def heart_rate_figure(hr: pd.DataFrame, window_days: int = 30) -> go.Figure:
    """Per-run average heart rate with its rolling mean; runs without HR are absent."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hr["date"], y=hr["avg_hr"], name="Per run", mode="markers",
        marker=dict(color=MUTED, size=8, opacity=0.55, line=dict(width=2, color=SURFACE)),
        hovertemplate="%{y:.0f} bpm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hr["date"], y=hr["rolling_hr"], name=f"{window_days}-day average", mode="lines",
        line=dict(color=SERIES["fitness"], width=2),
        hovertemplate="%{y:.1f} bpm average<extra></extra>",
    ))
    fig.update_layout(title="Average heart rate over time")
    return _base(fig, height=340, y_title="Heart rate (bpm)")
