"""Plotly figure builders for the dashboard.

Design rules kept consistently across every chart:

* One y-axis per chart -- never two scales on one plot.  CTL, ATL and TSB share
  a chart only because they are all in the same load-point unit.
* Colors carry identity, not magnitude: a fixed, colorblind-checked categorical
  order (blue, orange, aqua) assigned per series, never cycled or re-ranked.
* Thin marks, solid hairline gridlines, recessive axes, and direct labels on the
  endpoint that matters instead of a number on every point.
* Pace axes are reversed and ticked in ``m:ss`` so "higher" always means faster.
* Distances and paces arrive in kilometres; every builder that shows one takes
  a :class:`~src.units.Units` and converts only what it draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.consistency import SWEET_SPOT
from src.performance import format_pace
from src.units import METRIC, Units

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
        ticktext=[format_pace(t, suffix="") for t in ticks],
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


def fitness_freshness_figure(ff: pd.DataFrame, ctl_days: int = 42, atl_days: int = 7) -> go.Figure:
    """Fitness (CTL), Fatigue (ATL) and Form (TSB) -- all in load points."""
    fitness_name, fatigue_name = fitness_freshness_title(ctl_days, atl_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ff["date"], y=ff["ctl"], name=fitness_name, mode="lines",
        line=dict(color=SERIES["fitness"], width=2), fill="tozeroy",
        fillcolor="rgba(42,120,214,0.08)", hovertemplate="Fitness %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ff["date"], y=ff["atl"], name=fatigue_name, mode="lines",
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


_DISTANCE_TICKS = {"km": [3, 5, 10, 15, 21, 30, 42], "mi": [2, 3, 5, 10, 13.1, 20, 26.2]}


def pace_distance_figure(
    activities: pd.DataFrame, frontier: pd.DataFrame, units: Units = METRIC
) -> go.Figure:
    """Every run as distance vs pace (log x), with the best-effort frontier lit up."""
    hover = (f"%{{customdata}}<br>%{{x:.1f}} {units.distance_label} at "
             f"%{{y:.2f}} {units.pace_label}<extra></extra>")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=units.distance(activities["distance_km"]), y=units.pace(activities["pace_min_per_km"]),
        name="Every run", mode="markers",
        marker=dict(color=MUTED, size=8, opacity=0.55, line=dict(width=2, color=SURFACE)),
        customdata=activities["date"].dt.strftime("%d %b %Y"), hovertemplate=hover,
    ))
    fig.add_trace(go.Scatter(
        x=units.distance(frontier["distance_km"]), y=units.pace(frontier["pace_min_per_km"]),
        name="Best-effort frontier",
        mode="lines+markers", line=dict(color=SERIES["fitness"], width=2),
        marker=dict(color=SERIES["fitness"], size=10, line=dict(width=2, color=SURFACE)),
        customdata=frontier["date"].dt.strftime("%d %b %Y"), hovertemplate=hover,
    ))
    fig.update_layout(title="Pace vs distance", hovermode="closest")
    _base(fig, height=420, y_title=f"Pace ({units.pace_label})",
          x_title=f"Distance ({units.distance_label})")
    ticks = _DISTANCE_TICKS[units.distance_label]
    fig.update_xaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])
    _pace_axis(fig, units.pace(activities["pace_min_per_km"]))
    return fig


def _volume_figure(
    frame: pd.DataFrame, x: str, title: str, hover_fmt: str, units: Units
) -> go.Figure:
    label = units.distance_label
    fig = go.Figure(go.Bar(
        x=frame[x], y=units.distance(frame["distance_km"]),
        marker=dict(color=SERIES["fitness"], cornerradius=4), customdata=frame["runs"],
        hovertemplate=f"{hover_fmt}<br>%{{y:.1f}} {label} over %{{customdata}} runs<extra></extra>",
    ))
    fig.update_layout(title=title, bargap=0.25)
    return _base(fig, height=340, y_title=f"Distance ({label})")


def weekly_volume_figure(weekly: pd.DataFrame, units: Units = METRIC) -> go.Figure:
    """Distance per calendar week; empty weeks stay visible as gaps."""
    return _volume_figure(weekly, "week_start", "Weekly distance", "Week of %{x|%d %b %Y}",
                          units)


def weekly_hours_figure(weekly: pd.DataFrame) -> go.Figure:
    """Moving time per calendar week, in hours."""
    fig = go.Figure(go.Bar(
        x=weekly["week_start"], y=weekly["moving_time_h"],
        marker=dict(color=SERIES["fitness"], cornerradius=4), customdata=weekly["runs"],
        hovertemplate=("Week of %{x|%d %b %Y}<br>%{y:.1f} h over %{customdata} runs"
                       "<extra></extra>"),
    ))
    fig.update_layout(title="Hours run per week", bargap=0.25)
    return _base(fig, height=320, y_title="Hours")


def weekly_runs_figure(weekly: pd.DataFrame) -> go.Figure:
    """Number of runs per calendar week."""
    fig = go.Figure(go.Bar(
        x=weekly["week_start"], y=weekly["runs"],
        marker=dict(color=SERIES["fitness"], cornerradius=4),
        hovertemplate="Week of %{x|%d %b %Y}<br>%{y:.0f} runs<extra></extra>",
    ))
    fig.update_layout(title="Runs per week", bargap=0.25)
    _base(fig, height=320, y_title="Runs")
    fig.update_yaxes(dtick=1)
    return fig


def monthly_volume_figure(monthly: pd.DataFrame, units: Units = METRIC) -> go.Figure:
    """Distance per calendar month."""
    return _volume_figure(monthly, "month_start", "Monthly distance", "%{x|%b %Y}", units)


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


def rolling_pace_figure(
    rolling: pd.DataFrame, window_days: int = 30, units: Units = METRIC
) -> go.Figure:
    """Distance-weighted rolling average pace -- the "am I getting faster?" chart."""
    pace = units.pace(rolling["rolling_pace"])
    fig = go.Figure(go.Scatter(
        x=rolling["date"], y=pace, mode="lines",
        line=dict(color=SERIES["fitness"], width=2), name="Rolling pace",
        hovertemplate=f"%{{y:.2f}} {units.pace_label}<extra></extra>",
    ))
    fig.update_layout(title=f"Rolling {window_days}-day average pace", showlegend=False)
    _base(fig, height=340, y_title=f"Pace ({units.pace_label})")
    _pace_axis(fig, pace)
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


# Ordinal blue ramp for Z1-Z5 (light -> dark = easy -> hard); validated as one
# hue with monotone lightness and a light end that still clears the surface.
ZONE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def zone_distribution_figure(zones: pd.DataFrame) -> go.Figure:
    """Share of heart-rate-recorded time per zone, one bar per zone."""
    fig = go.Figure(go.Bar(
        x=zones["label"], y=zones["share"] * 100.0,
        marker=dict(color=ZONE_COLORS[:len(zones)], cornerradius=4),
        customdata=np.stack([zones["bpm"], zones["hours"], zones["runs"]], axis=-1),
        text=[f"{share:.0%}" for share in zones["share"]], textposition="outside",
        textfont=dict(color=INK_SECONDARY),
        hovertemplate=("%{x} (%{customdata[0]} bpm)<br>%{y:.0f}% of time, "
                       "%{customdata[1]:.1f} h over %{customdata[2]} runs<extra></extra>"),
    ))
    fig.update_layout(title="Time in heart-rate zone", showlegend=False,
                      hovermode="closest", bargap=0.25)
    _base(fig, height=360, y_title="Share of time (%)")
    fig.update_yaxes(range=[0, max(float(zones["share"].max()) * 115.0, 10.0)])
    return fig


def monthly_zone_figure(monthly: pd.DataFrame) -> go.Figure:
    """Hours per zone per month, stacked easy-at-the-bottom."""
    fig = go.Figure()
    for color, (_, frame) in zip(ZONE_COLORS, monthly.groupby("zone", sort=True)):
        name = frame["label"].iloc[0]
        fig.add_trace(go.Bar(
            x=frame["month_start"], y=frame["hours"], name=name,
            marker=dict(color=color, line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{name}: %{{y:.1f}} h<extra></extra>",
        ))
    fig.update_layout(title="Monthly hours by zone", barmode="stack", bargap=0.2)
    fig.update_xaxes(hoverformat="%b %Y")
    return _base(fig, height=380, y_title="Hours")


def run_zone_figure(breakdown: pd.DataFrame, run_labels: pd.Series) -> go.Figure:
    """One 100%-stacked bar per run, newest on top, easy zones on the left.

    ``breakdown`` is long-form per-run zone time from
    :func:`src.heart_rate.run_zone_breakdown`; ``run_labels`` maps activity_id
    to the text shown on the axis.
    """
    order = (breakdown.drop_duplicates("activity_id")
             .sort_values("date", ascending=True)["activity_id"].tolist())
    fig = go.Figure()
    for color, (_, frame) in zip(ZONE_COLORS, breakdown.groupby("zone", sort=True)):
        frame = frame.set_index("activity_id").loc[order]
        name = frame["label"].iloc[0]
        fig.add_trace(go.Bar(
            y=[run_labels[a] for a in order], x=frame["share"] * 100.0, name=name,
            orientation="h", customdata=frame["minutes"],
            marker=dict(color=color, line=dict(width=1, color=SURFACE)),
            hovertemplate=(f"{name}: %{{x:.0f}}% (%{{customdata:.0f}} min)"
                           "<extra>%{y}</extra>"),
        ))
    fig.update_layout(title="Zone mix run by run", barmode="stack", bargap=0.25,
                      hovermode="closest")
    _base(fig, height=max(320, 120 + 22 * len(order)), x_title="Share of run (%)")
    fig.update_xaxes(range=[0, 100], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SECONDARY))
    return fig


def recovery_timeline_figure(
    joined: pd.DataFrame, column: str, baseline: pd.Series, title: str, y_title: str
) -> go.Figure:
    """Nightly sleep score or HRV as dots, with a trailing 7-day baseline line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=joined["date"], y=joined[column], name="Nightly", mode="markers",
        marker=dict(color=MUTED, size=7, opacity=0.5, line=dict(width=1, color=SURFACE)),
        hovertemplate="%{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=joined["date"], y=baseline, name="7-day average", mode="lines",
        line=dict(color=SERIES["fitness"], width=2),
        hovertemplate="%{y:.1f} average<extra></extra>",
    ))
    fig.update_layout(title=title)
    fig.update_xaxes(hoverformat="%d %b %Y")
    return _base(fig, height=340, y_title=y_title)


def recovery_scatter_figure(
    joined: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, r: float
) -> go.Figure:
    """One dot per day, training metric vs recovery metric, with a least-squares line."""
    pair = joined[["date", x, y]].dropna()
    fig = go.Figure(go.Scatter(
        x=pair[x], y=pair[y], name="Day", mode="markers",
        marker=dict(color=MUTED, size=8, opacity=0.55, line=dict(width=1, color=SURFACE)),
        customdata=pair["date"].dt.strftime("%d %b %Y"),
        hovertemplate="%{customdata}<br>%{x:.0f} -> %{y:.0f}<extra></extra>",
    ))
    if len(pair) >= 2 and pair[x].std() > 0:
        slope, intercept = np.polyfit(pair[x], pair[y], 1)
        xs = np.array([pair[x].min(), pair[x].max()])
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, name="Trend", mode="lines",
            line=dict(color=SERIES["fitness"], width=2), hoverinfo="skip",
        ))
    r_text = f"r = {r:+.2f}" if np.isfinite(r) else "r not available"
    fig.update_layout(title=f"{y_title} vs {x_title.lower()} ({r_text}, {len(pair)} days)",
                      hovermode="closest", showlegend=False)
    return _base(fig, height=400, y_title=y_title, x_title=x_title)


def race_plan_figure(projection: pd.DataFrame, race_ready: tuple[float, float],
                     ctl_days: int = 42, atl_days: int = 7, history_days: int = 90) -> go.Figure:
    """The last ``history_days`` of Fitness, Fatigue and Form, then the plan dotted to race day."""
    fitness_name, fatigue_name = fitness_freshness_title(ctl_days, atl_days)
    history = projection[~projection["projected"]].tail(history_days)
    # The projected lines start on the last recorded day so the two halves join.
    future = pd.concat([history.tail(1), projection[projection["projected"]]])
    fig = go.Figure()
    fig.add_hrect(y0=race_ready[0], y1=race_ready[1], fillcolor=GOOD, opacity=0.08,
                  line_width=0)
    for column, name, color in (("ctl", fitness_name, SERIES["fitness"]),
                                ("atl", fatigue_name, SERIES["fatigue"]),
                                ("tsb", "Form (Fitness - Fatigue)", SERIES["form"])):
        label = name.split(" (")[0]
        fig.add_trace(go.Scatter(
            x=history["date"], y=history[column], name=name, mode="lines", legendgroup=column,
            line=dict(color=color, width=2),
            hovertemplate=f"{label} %{{y:.1f}}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=future["date"], y=future[column], name=f"{label} (planned)", mode="lines",
            legendgroup=column, showlegend=False, line=dict(color=color, width=2, dash="dot"),
            hovertemplate=f"{label} %{{y:.1f}} planned<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)

    race = projection.iloc[-1]
    fig.add_vline(x=race["date"], line_color=INK_SECONDARY, line_width=1, line_dash="dash")
    fig.add_annotation(x=race["date"], y=1, yref="paper", text="<b>Race day</b>",
                       showarrow=False, xanchor="right", yanchor="top",
                       font=dict(color=INK_SECONDARY, size=12), bgcolor=SURFACE)
    shown = pd.concat([history, future])[["ctl", "atl", "tsb"]].to_numpy()
    _gutter_labels(fig, [
        (float(race["ctl"]), f"Fitness {race['ctl']:.0f}", SERIES["fitness"]),
        (float(race["atl"]), f"Fatigue {race['atl']:.0f}", SERIES["fatigue"]),
        (float(race["tsb"]), f"Form {race['tsb']:+.0f}", SERIES["form"]),
    ], float(shown.max() - shown.min()))
    fig.update_layout(title="Projected Fitness &amp; Freshness to race day")
    return _base(fig, height=420, y_title="Load points", right=96)


def fitness_freshness_title(ctl_days: int, atl_days: int) -> tuple[str, str]:
    """Legend names for the Fitness and Fatigue lines at the given time constants."""
    return f"Fitness ({ctl_days}-day)", f"Fatigue ({atl_days}-day)"


# --------------------------------------------------------------------------
# GPS-stream charts
# --------------------------------------------------------------------------

_CURVE_TICKS = [0.4, 1, 1.609, 3, 5, 10, 21.1, 42.2]
_CURVE_LABELS = ["400 m", "1K", "1 mi", "3K", "5K", "10K", "HM", "M"]


def pace_curve_figure(
    all_time: pd.DataFrame, recent: pd.DataFrame, recent_days: int, units: Units = METRIC
) -> go.Figure:
    """Best pace at every distance from 400 m up: the running power curve."""
    fig = go.Figure()
    for frame, name, color, dash in (
        (all_time, "All-time best", SERIES["fitness"], "solid"),
        (recent, f"Last {recent_days} days", SERIES["fatigue"], "solid"),
    ):
        if frame.empty:
            continue
        fig.add_trace(go.Scatter(
            x=units.distance(frame["distance_km"]), y=units.pace(frame["pace_min_per_km"]),
            name=name, mode="lines+markers", line=dict(color=color, width=2, dash=dash),
            marker=dict(color=color, size=8, line=dict(width=2, color=SURFACE)),
            customdata=np.stack([
                [units.format_pace(p) for p in frame["pace_min_per_km"]],
                pd.to_datetime(frame["date"]).dt.strftime("%d %b %Y"),
            ], axis=-1),
            hovertemplate=(f"{name}<br>%{{x:.2f}} {units.distance_label} at %{{customdata[0]}}"
                           "<br>set %{customdata[1]}<extra></extra>"),
        ))
    fig.update_layout(title="Pace curve: best pace held over each distance",
                      hovermode="closest")
    _base(fig, height=400, y_title=f"Pace ({units.pace_label})", x_title="Distance")
    fig.update_xaxes(type="log", tickvals=[units.distance(t) for t in _CURVE_TICKS],
                     ticktext=_CURVE_LABELS)
    _pace_axis(fig, units.pace(pd.concat([all_time["pace_min_per_km"],
                                          recent["pace_min_per_km"]])))
    return fig


def zone_method_figure(average_method: pd.DataFrame, stream_method: pd.DataFrame) -> go.Figure:
    """Share of time per zone by the average-HR shortcut vs second-by-second HR."""
    fig = go.Figure()
    for frame, name, color in (
        (average_method, "Whole run at its average HR", SERIES["fatigue"]),
        (stream_method, "Second by second", SERIES["fitness"]),
    ):
        fig.add_trace(go.Bar(
            x=frame["label"], y=frame["share"] * 100.0, name=name,
            marker=dict(color=color, cornerradius=4, line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.0f}}% of time<extra></extra>",
        ))
    fig.update_layout(title="Time in zone: two ways to count it", barmode="group",
                      bargap=0.25, hovermode="closest")
    return _base(fig, height=360, y_title="Share of time (%)")


def splits_figure(splits: pd.DataFrame, units: Units = METRIC) -> go.Figure:
    """Pace per kilometre (or mile) for one run, heart rate in the hover."""
    hr_text = [f"{hr:.0f} bpm" if np.isfinite(hr) else "no HR"
               for hr in splits["avg_hr"].astype("float64")]
    pace = units.pace(splits["pace_min_per_km"].astype("float64"))
    fig = go.Figure(go.Scatter(
        x=splits["split"], y=pace, mode="lines+markers",
        line=dict(color=SERIES["fitness"], width=2),
        marker=dict(color=SERIES["fitness"], size=9, line=dict(width=2, color=SURFACE)),
        customdata=np.stack([[units.format_pace(p) for p in splits["pace_min_per_km"]], hr_text,
                             units.distance(splits["distance_m"] / 1000.0)], axis=-1),
        hovertemplate=(f"Split %{{x}} (%{{customdata[2]:.2f}} {units.distance_label})"
                       "<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"),
    ))
    fig.update_layout(title=f"Pace per {units.split_label.lower()}", showlegend=False,
                      hovermode="closest")
    _base(fig, height=340, y_title=f"Pace ({units.pace_label})", x_title=units.split_label)
    fig.update_xaxes(dtick=1 if len(splits) <= 25 else 5)
    _pace_axis(fig, pace)
    return fig


def split_hr_figure(splits: pd.DataFrame, units: Units = METRIC) -> go.Figure:
    """Average heart rate per kilometre (or mile) for one run."""
    fig = go.Figure(go.Scatter(
        x=splits["split"], y=splits["avg_hr"], mode="lines+markers",
        line=dict(color=SERIES["fatigue"], width=2),
        marker=dict(color=SERIES["fatigue"], size=9, line=dict(width=2, color=SURFACE)),
        hovertemplate="Split %{x}<br>%{y:.0f} bpm<extra></extra>",
    ))
    fig.update_layout(title=f"Heart rate per {units.split_label.lower()}", showlegend=False,
                      hovermode="closest")
    _base(fig, height=340, y_title="Heart rate (bpm)", x_title=units.split_label)
    fig.update_xaxes(dtick=1 if len(splits) <= 25 else 5)
    return fig


def _route_zoom(lat: pd.Series, lon: pd.Series, width_px: int = 640, height_px: int = 420) -> float:
    """Web-map zoom level that fits the route: a 256 px world tile doubles per level."""
    centre = np.radians(float(lat.mean()))
    lon_span = max(float(lon.max() - lon.min()), 1e-4)
    lat_span = max(float(lat.max() - lat.min()) / np.cos(centre), 1e-4)
    zoom = min(np.log2(width_px * 360.0 / (256.0 * lon_span)),
               np.log2(height_px * 360.0 / (256.0 * lat_span)))
    return float(np.clip(zoom - 0.3, 1.0, 17.0))


def route_map_figure(
    route: pd.DataFrame, color_by: str = "pace", units: Units = METRIC, basemap: bool = False
) -> go.Figure:
    """One run's route, coloured by pace or heart rate along the way.

    The colour is the sequential blue ramp, darker for faster or harder, with
    the scale clipped to the 5th-95th percentile so one GPS spike cannot wash
    out the rest of the run.  Without ``basemap`` the track is drawn on plain
    axes at equal-distance aspect (away from the equator a degree of longitude
    is shorter than one of latitude) and nothing is fetched.  With it, the
    track sits on CARTO street tiles, which the *browser* downloads -- so the
    run's area is sent to CARTO's servers.
    """
    by_hr = color_by == "hr" and route["hr"].notna().any()
    values = route["hr"] if by_hr else units.pace(route["pace_min_per_km"])
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    low, high = ((float(finite.quantile(0.05)), float(finite.quantile(0.95)))
                 if not finite.empty else (0.0, 1.0))
    if high <= low:
        high = low + 1.0
    scale = [[i / (len(ZONE_COLORS) - 1), color] for i, color in enumerate(ZONE_COLORS)]
    colorbar = dict(thickness=12, outlinewidth=0, tickfont=dict(color=MUTED),
                    title=dict(text="bpm" if by_hr else units.pace_label))
    if not by_hr:
        ticks = np.linspace(low, high, 5)
        colorbar.update(tickmode="array", tickvals=ticks,
                        ticktext=[format_pace(t, suffix="") for t in ticks])
    marker = dict(color=values.clip(low, high), colorscale=scale, cmin=low, cmax=high,
                  reversescale=not by_hr, size=7, colorbar=colorbar)
    hover = [
        f"{units.format_distance(d / 1000.0, 2)}<br>{units.format_pace(p)}"
        + (f"<br>{hr:.0f} bpm" if np.isfinite(hr) else "")
        for d, p, hr in zip(route["distance_m"], route["pace_min_per_km"],
                            route["hr"].astype("float64"))
    ]
    ends = route.iloc[[0, -1]]
    title = f"Route, coloured by {'heart rate' if by_hr else 'pace'}"

    if basemap:
        fig = go.Figure()
        fig.add_trace(go.Scattermap(
            lat=route["lat"], lon=route["lon"], mode="markers", marker=marker,
            text=hover, hovertemplate="%{text}<extra></extra>", name="Route",
        ))
        fig.add_trace(go.Scattermap(
            lat=ends["lat"], lon=ends["lon"], mode="markers+text", text=["Start", "Finish"],
            textposition="top right", marker=dict(color=INK, size=11),
            textfont=dict(color=INK, size=12), hoverinfo="skip", name="Ends",
        ))
        fig.update_layout(
            map=dict(style="carto-positron", zoom=_route_zoom(route["lat"], route["lon"]),
                     center=dict(lat=float(route["lat"].mean()), lon=float(route["lon"].mean()))),
            title=title, showlegend=False,
        )
        _base(fig, height=460)
        fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), hovermode="closest")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=route["lon"], y=route["lat"], mode="lines", hoverinfo="skip",
        line=dict(color=AXIS, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=route["lon"], y=route["lat"], mode="markers", marker=marker,
        text=hover, hovertemplate="%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ends["lon"], y=ends["lat"], mode="markers+text", text=["Start", "Finish"],
        textposition="top right", textfont=dict(color=INK, size=12), hoverinfo="skip",
        marker=dict(color=INK, size=11, line=dict(width=2, color=SURFACE)),
    ))
    fig.update_layout(title=title, showlegend=False, hovermode="closest")
    _base(fig, height=460)
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x",
                     scaleratio=1.0 / np.cos(np.radians(float(route["lat"].mean()))))
    return fig


# --------------------------------------------------------------------------
# Personal-model charts
# --------------------------------------------------------------------------

def riegel_fit_figure(fit, default_exponent: float, units: Units = METRIC) -> go.Figure:
    """Best efforts as pace vs distance, with the fitted and default Riegel curves.

    Riegel's ``T = a * D ** b`` is pace ``= a * D ** (b - 1) / 60`` -- a straight
    line on log distance.  The default curve is pinned to the same geometric
    centre as the fit, so the two differ only in slope.
    """
    points = fit.points
    distances = np.geomspace(points["distance_km"].min(), points["distance_km"].max(), 50)
    centre = float(np.exp(np.log(points["distance_km"]).mean()))
    default_a = fit.predict(centre) / centre ** default_exponent

    point_pace = points["time_s"] / 60.0 / points["distance_km"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=units.distance(distances),
        y=units.pace(default_a * distances ** default_exponent / 60.0 / distances),
        name=f"Riegel default ({default_exponent:.2f})", mode="lines",
        line=dict(color=MUTED, width=2, dash="dash"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=units.distance(distances),
        y=units.pace(np.array([fit.predict(d) / 60.0 / d for d in distances])),
        name=f"Fitted to you ({fit.exponent:.3f})", mode="lines",
        line=dict(color=SERIES["fitness"], width=2), hoverinfo="skip",
    ))
    labels = (points["distance"] if "distance" in points
              else points["distance_km"].map(units.format_distance))
    fig.add_trace(go.Scatter(
        x=units.distance(points["distance_km"]), y=units.pace(point_pace),
        name="Your best efforts", mode="markers",
        marker=dict(color=SERIES["fitness"], size=10, line=dict(width=2, color=SURFACE)),
        customdata=np.stack([labels, [units.format_pace(p) for p in point_pace]], axis=-1),
        hovertemplate="%{customdata[0]}: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(title="How your pace fades with distance", hovermode="closest")
    _base(fig, height=400, y_title=f"Pace ({units.pace_label})", x_title="Distance")
    fig.update_xaxes(type="log", tickvals=[units.distance(t) for t in _CURVE_TICKS],
                     ticktext=_CURVE_LABELS)
    _pace_axis(fig, units.pace(point_pace))
    return fig


def performance_model_figure(performance: pd.DataFrame, modelled: pd.DataFrame) -> go.Figure:
    """Each run's performance index with the fitted fitness-fatigue model through it."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=performance["date"], y=performance["performance"], name="Each run", mode="markers",
        marker=dict(color=MUTED, size=8, opacity=0.55, line=dict(width=2, color=SURFACE)),
        hovertemplate="%{x|%d %b %Y}<br>%{y:+.1f}% vs expected<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=modelled["date"], y=modelled["predicted"], name="Model", mode="lines",
        line=dict(color=SERIES["fitness"], width=2),
        hovertemplate="Model %{y:+.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_layout(title="Running faster than your heart rate predicts")
    return _base(fig, height=380, y_title="Speed vs expected (%)")


def time_constant_surface_figure(fit, default: tuple[int, int]) -> go.Figure:
    """Model fit (R²) over every pair of time constants, best and default marked."""
    grid = fit.surface.pivot(index="tau2", columns="tau1", values="r2")
    fig = go.Figure(go.Heatmap(
        x=grid.columns, y=grid.index, z=grid.to_numpy(),
        colorscale=[[0.0, "#f0efec"], [1.0, ZONE_COLORS[-1]]],
        colorbar=dict(title=dict(text="R²"), thickness=12, outlinewidth=0),
        hovertemplate="Fitness %{x} d, fatigue %{y} d<br>R² %{z:.3f}<extra></extra>",
        hoverongaps=False,
    ))
    for (x, y), name, symbol in (((fit.tau1, fit.tau2), "Best fit", "star"),
                                 (default, "Dashboard default", "circle-open")):
        fig.add_trace(go.Scatter(
            x=[x], y=[y], name=f"{name} ({x} / {y} d)", mode="markers",
            marker=dict(color=INK, size=14, symbol=symbol, line=dict(width=2, color=INK)),
            hovertemplate=f"{name}: {x} / {y} days<extra></extra>",
        ))
    fig.update_layout(title="How well each pair of time constants fits", hovermode="closest")
    return _base(fig, height=400, y_title="Fatigue time constant (days)",
                 x_title="Fitness time constant (days)")


def impulse_figure(fit) -> go.Figure:
    """Modelled effect of one training session on performance, day by day."""
    impulse = fit.impulse[fit.impulse["day"] <= 90]
    fig = go.Figure(go.Scatter(
        x=impulse["day"], y=impulse["effect"] * 100.0, mode="lines",
        line=dict(color=SERIES["fitness"], width=2), name="Net effect",
        hovertemplate="Day %{x}<br>%{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    if fit.peak_day is not None:
        peak = impulse.loc[impulse["day"] == fit.peak_day, "effect"].iloc[0] * 100.0
        fig.add_annotation(x=fit.peak_day, y=peak, text=f"peak, day {fit.peak_day}",
                           showarrow=False, yanchor="bottom", font=dict(color=INK_SECONDARY))
    fig.update_layout(title="What one session does, day by day", showlegend=False)
    return _base(fig, height=340, y_title="Effect (per 100 load points)",
                 x_title="Days after the session")


# --------------------------------------------------------------------------
# Export-column charts
# --------------------------------------------------------------------------

def relative_effort_figure(scored: pd.DataFrame, r: float) -> go.Figure:
    """This project's load score against Strava's own Relative Effort, one dot per run."""
    pair = scored[["date", "load", "relative_effort"]].dropna()
    fig = go.Figure(go.Scatter(
        x=pair["relative_effort"], y=pair["load"], mode="markers", name="Run",
        marker=dict(color=MUTED, size=8, opacity=0.55, line=dict(width=1, color=SURFACE)),
        customdata=pair["date"].dt.strftime("%d %b %Y"),
        hovertemplate="%{customdata}<br>Relative Effort %{x:.0f}, load %{y:.0f}<extra></extra>",
    ))
    if len(pair) >= 2 and pair["relative_effort"].std() > 0:
        slope, intercept = np.polyfit(pair["relative_effort"], pair["load"], 1)
        xs = np.array([pair["relative_effort"].min(), pair["relative_effort"].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Trend",
                                 line=dict(color=SERIES["fitness"], width=2), hoverinfo="skip"))
    fig.update_layout(title=f"This app's load vs Strava's Relative Effort (r = {r:+.2f})",
                      showlegend=False, hovermode="closest")
    return _base(fig, height=380, y_title="Load points (this app)",
                 x_title="Relative Effort (Strava)")


def gear_figure(gear: pd.DataFrame, retire_km: float, units: Units = METRIC) -> go.Figure:
    """Distance on each shoe, with the retirement mileage marked."""
    ordered = gear.sort_values("distance_km")
    colors = [SERIES["fitness"] if active else MUTED for active in ordered["active"]]
    fig = go.Figure(go.Bar(
        y=ordered["gear"], x=units.distance(ordered["distance_km"]), orientation="h",
        marker=dict(color=colors, cornerradius=4),
        customdata=np.stack([ordered["runs"], ordered["status"]], axis=-1),
        hovertemplate=(f"%{{y}}<br>%{{x:.0f}} {units.distance_label} over %{{customdata[0]}} "
                       "runs<br>%{customdata[1]}<extra></extra>"),
    ))
    fig.add_vline(x=units.distance(retire_km), line_color=INK_SECONDARY, line_width=1,
                  line_dash="dash",
                  annotation_text=f"retire at {units.format_distance(retire_km, 0)}",
                  annotation_position="top", annotation_font_color=INK_SECONDARY)
    fig.update_layout(title="Distance per shoe", showlegend=False, hovermode="closest",
                      bargap=0.3)
    _base(fig, height=max(240, 70 + 46 * len(ordered)),
          x_title=f"Distance ({units.distance_label})")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# Fixed hue per sport, in categorical order, so a sport keeps its colour
# whichever others the period contains.  Green is skipped: it is the "good"
# status colour on the ACWR chart.  Sports beyond these share muted grey.
SPORT_COLORS = {
    "Run": "#2a78d6", "Ride": "#eb6834", "Swim": "#1baf7a", "Strength": "#eda100",
    "Walk": "#e87ba4", "Hike": "#6250d6", "Row": "#e34948",
}


def _sport_order(sports) -> list[str]:
    """The palette's order first, then any other sport alphabetically."""
    present = set(sports)
    return ([s for s in SPORT_COLORS if s in present]
            + sorted(present - set(SPORT_COLORS)))


def sport_weekly_figure(weekly: pd.DataFrame, value: str = "hours") -> go.Figure:
    """Hours (or load) per calendar week, stacked by sport, runs at the bottom."""
    label = {"hours": ("Hours per week by sport", "Hours", "{y:.1f} h"),
             "load": ("Training load per week by sport", "Load points", "{y:.0f} points")}
    title, y_title, fmt = label[value]
    fig = go.Figure()
    for sport in _sport_order(weekly["sport"]):
        frame = weekly[weekly["sport"] == sport]
        fig.add_trace(go.Bar(
            x=frame["week_start"], y=frame[value], name=sport,
            marker=dict(color=SPORT_COLORS.get(sport, MUTED), line=dict(width=2, color=SURFACE)),
            hovertemplate=f"{sport}: %{fmt}<extra></extra>",
        ))
    fig.update_layout(title=title, barmode="stack", bargap=0.2)
    fig.update_xaxes(hoverformat="Week of %d %b %Y")
    return _base(fig, height=380, y_title=y_title)


def sport_load_figure(summary: pd.DataFrame) -> go.Figure:
    """Each sport's share of total training load, largest at the top."""
    ordered = summary.sort_values("load")
    fig = go.Figure(go.Bar(
        y=ordered["sport"], x=ordered["load_share"] * 100.0, orientation="h",
        marker=dict(color=[SPORT_COLORS.get(s, MUTED) for s in ordered["sport"]],
                    cornerradius=4),
        text=[f"{share:.0%}" for share in ordered["load_share"]], textposition="outside",
        textfont=dict(color=INK_SECONDARY),
        customdata=np.stack([ordered["sessions"], ordered["hours"]], axis=-1),
        hovertemplate=("%{y}: %{x:.0f}% of load<br>%{customdata[0]} sessions, "
                       "%{customdata[1]:.1f} h<extra></extra>"),
    ))
    fig.update_layout(title="Share of training load", showlegend=False, hovermode="closest",
                      bargap=0.3)
    _base(fig, height=max(240, 90 + 46 * len(ordered)), x_title="Share of load (%)")
    fig.update_xaxes(range=[0, max(float(ordered["load_share"].max()) * 118.0, 10.0)])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK_SECONDARY))
    return fig
