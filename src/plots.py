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
from plotly.subplots import make_subplots

from src.consistency import SWEET_SPOT
from src.performance import format_duration, format_pace
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
_EARTH_M_PER_DEG = 6_371_008.8 * np.pi / 180.0


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


def _pace_axis(fig: go.Figure, paces: pd.Series, **where) -> None:
    """Reverse the y-axis and tick it every 15 s of pace, labelled ``m:ss``.

    ``where`` (``row=``, ``col=``) picks one subplot's axis instead of all.
    """
    finite = pd.Series(paces).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return
    low = np.floor(finite.min() * 4) / 4
    high = np.ceil(finite.max() * 4) / 4
    step = 0.25 if high - low <= 3 else 1.0
    ticks = np.arange(low, high + step, step)
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=ticks,
        ticktext=[format_pace(t, suffix="") for t in ticks],
        **where,
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
    """Distance-weighted rolling average pace -- the "am I getting faster?" chart.

    When ``rolling`` carries a ``rolling_gap`` column with any values, the
    grade-adjusted pace is drawn alongside as a second line.
    """
    pace = units.pace(rolling["rolling_pace"])
    has_gap = bool("rolling_gap" in rolling and rolling["rolling_gap"].notna().any())
    fig = go.Figure(go.Scatter(
        x=rolling["date"], y=pace, mode="lines",
        line=dict(color=SERIES["fitness"], width=2), name="Pace",
        customdata=[units.format_pace(p) for p in rolling["rolling_pace"]],
        hovertemplate="Pace %{customdata}<extra></extra>",
    ))
    shown = [pace]
    if has_gap:
        gap = units.pace(rolling["rolling_gap"])
        shown.append(gap)
        fig.add_trace(go.Scatter(
            x=rolling["date"], y=gap, mode="lines",
            line=dict(color=SERIES["fatigue"], width=2), name="Grade-adjusted pace",
            customdata=[units.format_pace(p) for p in rolling["rolling_gap"]],
            hovertemplate="GAP %{customdata}<extra></extra>",
        ))
    fig.update_layout(title=f"Rolling {window_days}-day average pace", showlegend=has_gap)
    _base(fig, height=340, y_title=f"Pace ({units.pace_label})")
    _pace_axis(fig, pd.concat(shown))
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


def zone_distribution_figure(zones: pd.DataFrame, title: str = "Time in heart-rate zone",
                             range_column: str = "bpm", range_suffix: str = " bpm"
                             ) -> go.Figure:
    """Share of recorded time per zone, one bar per zone.

    ``range_column`` holds each zone's boundaries as text (``bpm`` for heart
    rate, ``pace`` for pace zones), shown in the hover with ``range_suffix``.
    """
    fig = go.Figure(go.Bar(
        x=zones["label"], y=zones["share"] * 100.0,
        marker=dict(color=ZONE_COLORS[:len(zones)], cornerradius=4),
        customdata=np.stack([zones[range_column], zones["hours"], zones["runs"]], axis=-1),
        text=[f"{share:.0%}" for share in zones["share"]], textposition="outside",
        textfont=dict(color=INK_SECONDARY),
        hovertemplate=(f"%{{x}} (%{{customdata[0]}}{range_suffix})<br>%{{y:.0f}}% of time, "
                       "%{customdata[1]:.1f} h over %{customdata[2]} runs<extra></extra>"),
    ))
    fig.update_layout(title=title, showlegend=False,
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


def zone_method_figure(average_method: pd.DataFrame, stream_method: pd.DataFrame,
                       average_name: str = "Whole run at its average HR") -> go.Figure:
    """Share of time per zone by the whole-run-average shortcut vs second by second."""
    fig = go.Figure()
    for frame, name, color in (
        (average_method, average_name, SERIES["fatigue"]),
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
    """Pace per kilometre (or mile) for one run, heart rate in the hover.

    Grade-adjusted pace is drawn as a second line when the splits have it.
    """
    hr_text = [f"{hr:.0f} bpm" if np.isfinite(hr) else "no HR"
               for hr in splits["avg_hr"].astype("float64")]
    pace = units.pace(splits["pace_min_per_km"].astype("float64"))
    gap_raw = (splits["gap_pace_min_per_km"].astype("float64")
               if "gap_pace_min_per_km" in splits else pd.Series(np.nan, index=splits.index))
    has_gap = bool(gap_raw.notna().any())
    gap_text = [f"GAP {units.format_pace(g)}" if np.isfinite(g) else "" for g in gap_raw]
    fig = go.Figure(go.Scatter(
        x=splits["split"], y=pace, mode="lines+markers", name="Pace",
        line=dict(color=SERIES["fitness"], width=2),
        marker=dict(color=SERIES["fitness"], size=9, line=dict(width=2, color=SURFACE)),
        customdata=np.stack([[units.format_pace(p) for p in splits["pace_min_per_km"]], hr_text,
                             units.distance(splits["distance_m"] / 1000.0), gap_text], axis=-1),
        hovertemplate=(f"Split %{{x}} (%{{customdata[2]:.2f}} {units.distance_label})"
                       "<br>Pace %{customdata[0]}<br>%{customdata[3]}<br>%{customdata[1]}"
                       "<extra></extra>"),
    ))
    shown = [pace]
    if has_gap:
        gap = units.pace(gap_raw)
        shown.append(gap)
        fig.add_trace(go.Scatter(
            x=splits["split"], y=gap, mode="lines+markers", name="Grade-adjusted pace",
            line=dict(color=SERIES["fatigue"], width=2),
            marker=dict(color=SERIES["fatigue"], size=8, line=dict(width=2, color=SURFACE)),
            hoverinfo="skip",
        ))
    fig.update_layout(title=f"Pace per {units.split_label.lower()}", showlegend=has_gap,
                      hovermode="closest")
    _base(fig, height=340, y_title=f"Pace ({units.pace_label})", x_title=units.split_label)
    fig.update_xaxes(dtick=1 if len(splits) <= 25 else 5)
    _pace_axis(fig, pd.concat(shown))
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


# Pace heat: fastest (hot red) -> median (neutral) -> slowest (cool blue).  Validated
# for adjacent-pair separation (normal vision and CVD); the light neutral middle is
# kept visible on a white map by the dark casing drawn under the route.
PACE_HEAT_COLORS = ["#8c1616", "#d93b3a", "#f4908f", "#e2e1dc", "#7fb0ee", "#2a78d6", "#0f3a72"]
NO_PACE_COLOR = AXIS


def _band_colors(n: int) -> list[str]:
    """``n`` diverging band colours, fastest first (``n`` is odd; 7 uses the full scale)."""
    if n == len(PACE_HEAT_COLORS):
        return list(PACE_HEAT_COLORS)
    picks = np.linspace(0, len(PACE_HEAT_COLORS) - 1, n).round().astype(int)
    return [PACE_HEAT_COLORS[i] for i in picks]


def _segment_path(segs: pd.DataFrame) -> tuple[list, list]:
    """Lat and lon lists that draw each segment, with consecutive ones joined.

    One trace per colour band keeps the figure light; ``None`` breaks the line
    wherever the next segment of this band does not start where the last ended.
    """
    lats: list = []
    lons: list = []
    last_end = None
    for s in segs.itertuples():
        if last_end != (s.lat0, s.lon0):
            if lats:
                lats.append(None)
                lons.append(None)
            lats.append(s.lat0)
            lons.append(s.lon0)
        lats.append(s.lat1)
        lons.append(s.lon1)
        last_end = (s.lat1, s.lon1)
    return lats, lons


RETRACE_SHARE = 0.15   # this much of a route doubling back on itself gets keep-right lanes
LANE_PX = 5.0          # how far each direction is pushed off the centre line, in pixels


def _retraced_share(east: np.ndarray, north: np.ndarray, near_m: float = 40.0,
                    apart_m: float = 400.0) -> float:
    """Share of points with a point at least ``apart_m`` further along within ``near_m``."""
    along = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(east), np.diff(north)))])
    step = max(len(east) // 150, 1)  # thin long tracks; the share barely changes
    e, n, a = east[::step], north[::step], along[::step]
    d = np.hypot(e[:, None] - e[None, :], n[:, None] - n[None, :])
    far_along = np.abs(a[:, None] - a[None, :]) >= apart_m
    return float(((d <= near_m) & far_along).any(axis=1).mean())


def _keep_right(segs: pd.DataFrame, lat_mid: float, plot_px: float = 460.0) -> pd.DataFrame:
    """Shift a route that doubles back to the right of its direction of travel.

    On an out-and-back the way home is drawn over the way out, hiding half the
    run.  Pushing every point a few pixels to its right (at the map's scale)
    splits the two directions into side-by-side lanes, like traffic.  Routes
    that never double back are returned untouched, so they stay on the road.
    """
    if len(segs) < 3:
        return segs
    k = _EARTH_M_PER_DEG
    cos = np.cos(np.radians(lat_mid))
    lat = np.append(segs["lat0"].to_numpy(), segs["lat1"].iloc[-1])
    lon = np.append(segs["lon0"].to_numpy(), segs["lon1"].iloc[-1])
    east, north = lon * k * cos, lat * k
    if _retraced_share(east, north) < RETRACE_SHARE:
        return segs
    span = max(float(np.ptp(east)), float(np.ptp(north)), 1.0)
    offset = LANE_PX * span / plot_px
    # Direction at each point from a few points either side, so GPS jitter cannot flip it.
    w = 3
    pe, pn = np.pad(east, w, mode="edge"), np.pad(north, w, mode="edge")
    de, dn = pe[2 * w:] - pe[:-2 * w], pn[2 * w:] - pn[:-2 * w]
    length = np.hypot(de, dn)
    length[length == 0] = 1.0
    east = east + offset * dn / length   # (dn, -de) points to the right of travel
    north = north - offset * de / length
    out = segs.copy()
    out["lat0"], out["lat1"] = north[:-1] / k, north[1:] / k
    out["lon0"], out["lon1"] = east[:-1] / (k * cos), east[1:] / (k * cos)
    return out


def pace_heatmap_figure(heat, extremes: pd.DataFrame, units: Units = METRIC,
                        basemap: bool = False) -> go.Figure:
    """The route painted by pace: hot where you were fast, cool where you eased off.

    ``heat`` comes from :func:`src.pace_heat.pace_heat`, ``extremes`` from
    :func:`src.pace_heat.split_extremes`.  Three linked views, one colour key:

    * the **map** -- a continuous line, each stretch coloured by its pace band
      (centred on the run's own median), with the fastest and slowest split
      labelled where they happened;
    * the **distribution** beside it -- how much of the run's distance fell in
      each band, which doubles as the legend;
    * the **pace ribbon** underneath -- the same colours laid out along the
      distance axis, so you can see *when* in the run each band came, ticked
      at every kilometre or mile.

    Nothing needs scrubbing: the shape of the run is readable without a hover.
    """
    segs, bands = heat.segments, heat.bands
    colors = _band_colors(len(bands))
    lat_mid = float(np.nanmean(np.concatenate([segs["lat0"], segs["lat1"]])))
    segs = _keep_right(segs, lat_mid)
    lon_mid = float(np.nanmean(np.concatenate([segs["lon0"], segs["lon1"]])))

    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.7, 0.3], row_heights=[0.8, 0.2],
        horizontal_spacing=0.12, vertical_spacing=0.09,  # the gap holds the band labels
        specs=[[{"type": "map" if basemap else "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
    )
    point = go.Scattermap if basemap else go.Scatter

    def where(lats, lons) -> dict:
        return dict(lat=lats, lon=lons) if basemap else dict(x=lons, y=lats)

    # The map: a dark casing under everything, then one line trace per band.
    all_lat, all_lon = _segment_path(segs)
    fig.add_trace(point(**where(all_lat, all_lon), mode="lines", hoverinfo="skip",
                        line=dict(color=INK_SECONDARY, width=8), opacity=0.55), row=1, col=1)
    gaps = segs[segs["band"] < 0]
    if not gaps.empty:
        fig.add_trace(point(**where(*_segment_path(gaps)), mode="lines", hoverinfo="skip",
                            line=dict(color=NO_PACE_COLOR, width=5)), row=1, col=1)
    for b in bands.itertuples():
        in_band = segs[segs["band"] == b.band]
        if in_band.empty:
            continue
        fig.add_trace(point(**where(*_segment_path(in_band)), mode="lines", hoverinfo="skip",
                            line=dict(color=colors[b.band], width=5)), row=1, col=1)

    # Hover: map lines do not take hover, so an invisible dot sits on each stretch.
    paced = segs[segs["band"] >= 0]
    diff_s = (paced["pace"] - heat.median) * 60.0
    hover = [
        f"{units.format_distance(d / 1000.0, 2)}<br>"
        f"<b>{format_pace(p, suffix=' /' + units.distance_label)}</b><br>"
        + (f"{abs(ds):.0f} s {'faster' if ds < 0 else 'slower'} than median"
           if abs(ds) >= 1 else "at the median pace")
        for d, p, ds in zip(paced["distance_m"], paced["pace"], diff_s)
    ]
    fig.add_trace(point(
        **where((paced["lat0"] + paced["lat1"]) / 2, (paced["lon0"] + paced["lon1"]) / 2),
        mode="markers", marker=dict(size=14, color=INK, opacity=0),
        text=hover, hovertemplate="%{text}<extra></extra>",
    ), row=1, col=1)

    # Start / finish, then the fastest and slowest split, labelled in place.
    ends_lat = [segs["lat0"].iloc[0], segs["lat1"].iloc[-1]]
    ends_lon = [segs["lon0"].iloc[0], segs["lon1"].iloc[-1]]
    ends_text = ["Start", "Finish"]
    apart_m = np.hypot((ends_lat[1] - ends_lat[0]) * _EARTH_M_PER_DEG,
                       (ends_lon[1] - ends_lon[0]) * _EARTH_M_PER_DEG * np.cos(np.radians(lat_mid)))
    if apart_m < 100.0:  # a loop or an out-and-back: one marker, one label
        ends_lat, ends_lon, ends_text = ends_lat[:1], ends_lon[:1], ["Start / Finish"]
    fig.add_trace(point(**where(ends_lat, ends_lon), mode="markers+text",
                        text=ends_text, textposition="top right",
                        textfont=dict(color=INK, size=12), hoverinfo="skip",
                        marker=dict(color=INK, size=10)), row=1, col=1)
    if not extremes.empty:
        label = units.split_label.lower().replace("kilometre", "km")
        texts = [f"<b>{e.kind.capitalize()} {label} {e.split}</b> "
                 f"{format_pace(e.pace, suffix='')}" for e in extremes.itertuples()]
        # The northern label sits above its marker and the other below, so two neighbouring
        # splits never collide, and centred text does not run off the side of the panel.
        north = extremes["lat"].to_numpy() >= extremes["lat"].max()
        # One trace each: map traces take a single text position, not one per point.
        for e, text, n in zip(extremes.itertuples(), texts, north):
            fig.add_trace(point(
                **where([e.lat], [e.lon]), mode="markers+text", text=[text],
                textposition="top center" if n else "bottom center",
                textfont=dict(color=INK, size=12), hoverinfo="skip",
                marker=dict(size=13, color=colors[0] if e.kind == "fastest" else colors[-1])),
                row=1, col=1)

    # Distribution = legend: share of distance in each band, fastest on top.
    shares = bands["share"].to_numpy() * 100.0
    middle = len(bands) // 2
    fig.add_trace(go.Bar(
        x=shares, y=list(bands["label"]), orientation="h",
        marker=dict(color=colors, cornerradius=4, line=dict(color=INK_SECONDARY, width=0.5)),
        text=[f"{s:.0f}%" + (" · median" if i == middle else "") for i, s in enumerate(shares)],
        textposition="outside", cliponaxis=False, textfont=dict(color=INK_SECONDARY, size=12),
        hovertemplate="%{y}<br>%{x:.0f}% of the distance<extra></extra>",
    ), row=1, col=2)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK, size=12),
                     ticks="", row=1, col=2)
    fig.update_xaxes(visible=False, range=[0, max(float(shares.max()), 1.0) * 1.7],
                     row=1, col=2)

    # Ribbon: every stretch as a coloured block along the distance axis.
    starts = units.distance((segs["distance_m"] - segs["length_m"] / 2) / 1000.0)
    widths = units.distance(segs["length_m"] / 1000.0)
    block_colors = [colors[b] if b >= 0 else NO_PACE_COLOR for b in segs["band"]]
    ribbon_hover = [
        f"{units.format_distance(d / 1000.0, 2)}<br>"
        + (format_pace(p, suffix=" /" + units.distance_label) if b >= 0 else "no pace")
        for d, p, b in zip(segs["distance_m"], segs["pace"], segs["band"])
    ]
    fig.add_trace(go.Bar(
        x=widths, base=starts, y=["Pace"] * len(segs), orientation="h",
        marker=dict(color=block_colors, line=dict(width=0)), width=0.9,
        text=ribbon_hover, hovertemplate="%{text}<extra></extra>", textposition="none",
    ), row=2, col=1)
    total = float(units.distance((segs["distance_m"] + segs["length_m"] / 2).iloc[-1] / 1000.0))
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(range=[0, total], dtick=1 if total <= 30 else 5, showgrid=False,
                     row=2, col=1)
    if not extremes.empty:
        for e in extremes.itertuples():
            x = e.split - 0.5  # the middle of that split, in display distance
            fig.add_annotation(x=x, y=1.0, xref="x2" if basemap else "x3",
                               yref=("y2" if basemap else "y3") + " domain",
                               yanchor="bottom", showarrow=False,
                               text=f"{e.kind}", font=dict(size=11, color=MUTED))

    title = (f"Pace heat map  ·  median {format_pace(heat.median, suffix=' /' + units.distance_label)}"
             f", bands of {heat.step * 60:.0f} s")
    _base(fig, height=640)
    fig.update_layout(title=title, showlegend=False, hovermode="closest", bargap=0,
                      margin=dict(l=8, r=8, t=48, b=48))
    fig.update_xaxes(title_text=f"Distance ({units.distance_label})", row=2, col=1)
    if basemap:
        lats = pd.concat([segs["lat0"], segs["lat1"]])
        lons = pd.concat([segs["lon0"], segs["lon1"]])
        fig.update_layout(map=dict(style="carto-positron", zoom=_route_zoom(lats, lons, 520, 460),
                                   center=dict(lat=lat_mid, lon=lon_mid)))
    else:
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, scaleanchor="x",
                         scaleratio=1.0 / np.cos(np.radians(lat_mid)), row=1, col=1)
    return fig


# --------------------------------------------------------------------------
# Most frequent routes
# --------------------------------------------------------------------------

# The validated categorical order; a route keeps its colour (by route number) in any period.
ROUTE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                "#6250d6", "#e34948")


def route_color(route_id: int) -> str:
    """Routes 1-8 get the categorical hues in order; later ones are drawn muted."""
    return ROUTE_COLORS[route_id - 1] if 1 <= route_id <= len(ROUTE_COLORS) else MUTED


def routes_overview_figure(table: pd.DataFrame, tracks: dict, units: Units = METRIC,
                           basemap: bool = False, selected: int | None = None) -> go.Figure:
    """Every frequent route on one map, coloured and numbered, the selected one on top.

    ``table`` is :func:`src.routes.route_table`; each route is drawn from its
    reference run's track in ``tracks``.  The number sits at the route's
    farthest point from the start, where the routes fan out, so labels rarely
    overlap.  The selected route is drawn thicker and the others fade back.
    """
    point = go.Scattermap if basemap else go.Scatter

    def where(lats, lons) -> dict:
        return dict(lat=lats, lon=lons) if basemap else dict(x=lons, y=lats)

    fig = go.Figure()
    order = sorted(table.itertuples(), key=lambda r: r.route_id == selected)  # selected last
    all_lat, all_lon = [], []
    for r in order:
        track = tracks.get(r.reference_id)
        if track is None or track.empty:
            continue
        lat, lon = track["lat"].to_numpy(), track["lon"].to_numpy()
        all_lat.append(lat)
        all_lon.append(lon)
        chosen = selected is None or r.route_id == selected
        label = (f"<b>{r.route_id}. {r.name}</b><br>{units.format_distance(r.distance_km)} "
                 f"{r.shape} · {r.runs} runs")
        fig.add_trace(point(**where(lat, lon), mode="lines", name=f"{r.route_id}. {r.name}",
                            legendrank=r.route_id,
                            line=dict(color=route_color(r.route_id), width=5 if chosen else 3),
                            opacity=1.0 if chosen else 0.35, text=label, hoverinfo="text"))
        far = int(np.argmax((lat - lat[0]) ** 2 + ((lon - lon[0]) * np.cos(np.radians(lat[0]))) ** 2))
        fig.add_trace(point(**where([lat[far]], [lon[far]]), mode="markers+text",
                            text=[f"<b>{r.route_id}</b>"], textposition="middle center",
                            textfont=dict(color=INK, size=11),
                            marker=dict(size=20, color=route_color(r.route_id),
                                        opacity=1.0 if chosen else 0.5),
                            hovertext=label, hoverinfo="text", showlegend=False))
    fig.update_layout(title="Your most frequent routes", hovermode="closest",
                      legend=dict(orientation="h", y=-0.02, yanchor="top", x=0))
    _base(fig, height=480)
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8),
                      legend=dict(orientation="h", y=-0.02, yanchor="top", x=0))
    if not all_lat:
        return fig
    lats, lons = pd.Series(np.concatenate(all_lat)), pd.Series(np.concatenate(all_lon))
    if basemap:
        fig.update_layout(map=dict(style="carto-positron", zoom=_route_zoom(lats, lons, 640, 400),
                                   center=dict(lat=float(lats.mean()), lon=float(lons.mean()))))
    else:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x",
                         scaleratio=1.0 / np.cos(np.radians(float(lats.mean()))))
    return fig


def route_history_figure(efforts: pd.DataFrame, units: Units = METRIC) -> go.Figure:
    """Every run on one route: its pace by date, the PR as it stood, and the PR itself.

    ``efforts`` is :func:`src.routes.route_efforts`.  The step line is the
    route PR at each point in time, so each drop is a new PR; the all-time PR
    is ringed and labelled.
    """
    pace = units.pace(efforts["pace_min_per_km"])
    best = units.pace(efforts["best_so_far"])
    hover = [f"<b>{units.format_pace(p)}</b><br>{format_duration(t)} for "
             f"{units.format_distance(d, 2)}<br>{dt:%d %b %Y} · {n}"
             for p, t, d, dt, n in zip(efforts["pace_min_per_km"], efforts["moving_time_s"],
                                       efforts["distance_km"], efforts["date"], efforts["name"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=efforts["date"], y=best, mode="lines", name="Route PR at the time",
                             line=dict(color=INK_SECONDARY, width=1.5, shape="hv"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=efforts["date"], y=pace, mode="markers", name="Run",
                             marker=dict(color=SERIES["fitness"], size=9,
                                         line=dict(width=2, color=SURFACE)),
                             text=hover, hovertemplate="%{text}<extra></extra>"))
    pr = efforts[efforts["is_pr"]]
    if not pr.empty:
        p = pr.iloc[0]
        y = units.pace(p["pace_min_per_km"])
        fig.add_trace(go.Scatter(
            x=[p["date"]], y=[y], mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=2.5, color=INK)),
            hoverinfo="skip", showlegend=False))
        # A callout below-left, on a surface chip, so neighbouring runs cannot cover it.
        fig.add_annotation(x=p["date"], y=y, ax=-44, ay=34, showarrow=True, arrowwidth=1,
                           arrowcolor=INK_SECONDARY, bgcolor=SURFACE, borderpad=2,
                           text=f"<b>PR</b> {format_pace(y, suffix='')} · {p['date']:%d %b %Y}",
                           font=dict(color=INK, size=12))
    fig.update_layout(title="Every run on this route", hovermode="closest")
    _base(fig, height=340, y_title=units.pace_label)
    _pace_axis(fig, pd.concat([pace, best]))
    return fig


RUNNER = SERIES["fatigue"]


def _replay_readout(frame: pd.Series, marks: pd.DataFrame, units: Units) -> str:
    """The line above the replay: where the runner is, the clock, the pace, the last split."""
    parts = [f"<b>{units.format_distance(frame['distance_m'] / 1000.0, 2)}</b>",
             f"{format_duration(frame['moving_s'])} elapsed",
             f"{units.format_pace(frame['pace_min_per_km'])} now"]
    if np.isfinite(frame["hr"]):
        parts.append(f"{frame['hr']:.0f} bpm")
    passed = marks[marks["moving_s"] <= frame["moving_s"] + 1e-6]
    if not passed.empty:
        last = passed.iloc[-1]
        parts.append(f"{units.split_label} {int(last['split'])} split "
                     f"{format_pace(units.pace(last['pace_min_per_km']), suffix='')}")
    return "  ·  ".join(parts)


def route_replay_figure(
    route: pd.DataFrame, frames: pd.DataFrame, marks: pd.DataFrame,
    units: Units = METRIC, basemap: bool = False, duration_s: float = 20.0,
) -> go.Figure:
    """An animated replay: a runner dot crossing the route, its pace drawn as it goes.

    ``frames`` and ``marks`` come from :func:`src.replay.replay_frames` and
    :func:`src.replay.milestones`.  Frames are evenly spaced in moving time, so
    at a steady frame rate the dot moves as fast as the run did.  The top
    panel is the route (the stretch already run coloured by pace, darker for
    faster, with a numbered marker at each kilometre or mile); the bottom one
    is pace against distance, filling in behind a cursor, with each split's
    pace written over its marker line.  Play, Pause and a scrubber sit below.
    The whole replay lasts about ``duration_s`` seconds whatever the run's
    length.  ``basemap`` puts the route on CARTO tiles, as in
    :func:`route_map_figure`.
    """
    paces = units.pace(route["pace_min_per_km"]).replace([np.inf, -np.inf], np.nan)
    finite = paces.dropna()
    low, high = ((float(finite.quantile(0.05)), float(finite.quantile(0.95)))
                 if not finite.empty else (0.0, 1.0))
    if high <= low:
        high = low + 1.0
    # The pace line is clipped a little wider than the colours, so a stop's spike stays on the chart.
    lo_line, hi_line = ((float(finite.quantile(0.02)), float(finite.quantile(0.98)))
                        if not finite.empty else (low, high))
    shown = paces.clip(lo_line, hi_line).to_numpy()
    route_x = units.distance(route["distance_m"].to_numpy() / 1000.0)
    frame_x = units.distance(frames["distance_m"].to_numpy() / 1000.0)
    frame_pace = units.pace(frames["pace_min_per_km"]).clip(lo_line, hi_line).to_numpy()
    lat, lon = route["lat"].to_numpy(), route["lon"].to_numpy()

    point = go.Scattermap if basemap else go.Scatter

    def where(lats, lons) -> dict:
        return dict(lat=lats, lon=lons) if basemap else dict(x=lons, y=lats)

    scale = [[i / (len(ZONE_COLORS) - 1), color] for i, color in enumerate(ZONE_COLORS)]
    ticks = np.linspace(low, high, 5)
    trail_marker = dict(
        colorscale=scale, cmin=low, cmax=high, reversescale=True, size=6,
        colorbar=dict(thickness=10, outlinewidth=0, len=0.6, y=1.0, yanchor="top",
                      tickfont=dict(color=MUTED), title=dict(text=units.pace_label),
                      tickmode="array", tickvals=ticks,
                      ticktext=[format_pace(t, suffix="") for t in ticks]),
    )

    def moving_traces(i: int) -> list:
        """The four traces that change from frame to frame, at frame ``i``."""
        frame = frames.iloc[i]
        k = int(frame["trail"])
        trail_lat = np.append(lat[:k], frame["lat"])
        trail_lon = np.append(lon[:k], frame["lon"])
        trail_pace = np.append(np.clip(paces.to_numpy()[:k], low, high),
                               np.clip(units.pace(frame["pace_min_per_km"]), low, high))
        return [
            point(**where(trail_lat, trail_lon), mode="markers", hoverinfo="skip",
                  marker=dict(trail_marker, color=trail_pace)),
            point(**where([frame["lat"]], [frame["lon"]]), mode="markers", hoverinfo="skip",
                  marker=dict(color=RUNNER, size=16)),
            go.Scatter(x=np.append(route_x[:k], frame_x[i]),
                       y=np.append(shown[:k], frame_pace[i]), mode="lines",
                       hoverinfo="skip", line=dict(color=SERIES["fitness"], width=2)),
            go.Scatter(x=[frame_x[i]], y=[frame_pace[i]], mode="markers", hoverinfo="skip",
                       marker=dict(color=RUNNER, size=11, line=dict(width=2, color=SURFACE))),
        ]

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.68, 0.32], vertical_spacing=0.07,
        specs=[[{"type": "map" if basemap else "xy"}], [{"type": "xy"}]],
    )
    # 0: the whole route, faint, so you can see where the runner is headed.
    fig.add_trace(point(**where(lat, lon), mode="lines", hoverinfo="skip",
                        line=dict(color=AXIS, width=3)), row=1, col=1)
    # 1 and 3 (map) and 5-6 (pace chart) move; 4 is the whole pace line, faint.
    first = moving_traces(0)
    fig.add_trace(first[0], row=1, col=1)
    # 2: a numbered marker at each kilometre or mile, above the trail, below the runner.
    mark_hover = [
        f"{units.split_label} {int(m.split)}<br>reached at {format_duration(m.moving_s)}"
        f"<br>split {format_duration(m.split_s)} "
        f"({units.format_pace(m.pace_min_per_km)})"
        for m in marks.itertuples()
    ]
    fig.add_trace(point(
        **where(marks["lat"], marks["lon"]), mode="markers+text",
        text=[str(int(s)) for s in marks["split"]], textposition="top center",
        textfont=dict(color=INK, size=12), marker=dict(color=INK, size=8),
        hovertext=mark_hover, hovertemplate="%{hovertext}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(first[1], row=1, col=1)
    fig.add_trace(go.Scatter(x=route_x, y=shown, mode="lines", line=dict(color=GRID, width=1.5),
                             hoverinfo="skip"), row=2, col=1)
    fig.add_trace(first[2], row=2, col=1)
    fig.add_trace(first[3], row=2, col=1)

    # add_vline cannot be used: it inspects every trace's x-axis, and map traces have none.
    xref, yref = ("x", "y") if basemap else ("x2", "y2")
    label_splits = len(marks) <= 16  # past that the pace labels would overlap
    previous = 0.0
    for m in marks.itertuples():
        x = units.distance(m.distance_m / 1000.0)
        fig.add_shape(type="line", xref=xref, yref=f"{yref} domain", x0=x, x1=x, y0=0, y1=1,
                      line=dict(color=AXIS, width=1, dash="dot"), layer="below")
        middle, previous = (previous + x) / 2.0, x  # each split's pace sits over that split
        if label_splits:
            fig.add_annotation(xref=xref, yref=f"{yref} domain", x=middle, y=1.0,
                               yanchor="bottom",
                               showarrow=False, font=dict(size=10, color=MUTED),
                               text=format_pace(units.pace(m.pace_min_per_km), suffix=""))

    readouts = [_replay_readout(frames.iloc[i], marks, units) for i in range(len(frames))]
    fig.frames = [
        go.Frame(name=str(i), data=moving_traces(i), traces=[1, 3, 5, 6],
                 layout=dict(title=dict(text=readouts[i])))
        for i in range(len(frames))
    ]
    step_ms = max(int(duration_s * 1000 / max(len(frames), 1)), 16)
    play = dict(frame=dict(duration=step_ms, redraw=True), transition=dict(duration=0),
                fromcurrent=True, mode="immediate")
    still = dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0),
                 mode="immediate")

    _base(fig, height=700)
    fig.update_layout(
        title=dict(text=readouts[0] if readouts else "", font=dict(size=14, color=INK)),
        showlegend=False, hovermode="closest",
        margin=dict(l=8, r=8, t=48, b=110),
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0, xanchor="left", y=-0.115, yanchor="top", pad=dict(t=0, r=8),
            bgcolor=SURFACE, bordercolor=AXIS, font=dict(color=INK),
            buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                     dict(label="❚❚ Pause", method="animate", args=[[None], still])],
        )],
        sliders=[dict(
            active=0, x=0.17, len=0.83, y=-0.105, yanchor="top", pad=dict(t=0),
            currentvalue=dict(visible=False), ticklen=0, minorticklen=0,
            bordercolor=AXIS, bgcolor=GRID, activebgcolor=RUNNER, font=dict(size=1),
            steps=[dict(method="animate", label="", args=[[str(i)], still])
                   for i in range(len(frames))],
        )],
    )
    if basemap:
        fig.update_layout(map=dict(
            style="carto-positron", zoom=_route_zoom(route["lat"], route["lon"], height_px=440),
            center=dict(lat=float(route["lat"].mean()), lon=float(route["lon"].mean()))))
    else:
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, scaleanchor="x",
                         scaleratio=1.0 / np.cos(np.radians(float(route["lat"].mean()))),
                         row=1, col=1)
    fig.update_xaxes(title_text=f"Distance ({units.distance_label})", row=2, col=1)
    fig.update_yaxes(title_text=units.pace_label, row=2, col=1)
    _pace_axis(fig, pd.Series(shown), row=2, col=1)
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


# One colour per athlete, you first; a fixed order so a friend keeps their colour.
ATHLETE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")


def athlete_color(index: int) -> str:
    return ATHLETE_COLORS[index % len(ATHLETE_COLORS)]


def athlete_lines_figure(frame: pd.DataFrame, names: list[str], y: str, title: str,
                         y_title: str, hover_fmt: str, scale: float = 1.0) -> go.Figure:
    """One line per athlete from a long ``DataFrame[date, athlete, <y>]``, labelled at the end."""
    fig = go.Figure()
    entries = []
    for index, name in enumerate(names):
        rows = frame[frame["athlete"] == name]
        values = rows[y] * scale
        color = athlete_color(index)
        fig.add_trace(go.Scatter(
            x=rows["date"], y=values, name=name, mode="lines", line=dict(color=color, width=2),
            hovertemplate=f"{name}: %{{y:{hover_fmt}}}<extra></extra>",
        ))
        if not rows.empty:
            entries.append((float(values.iloc[-1]), f"{name} {values.iloc[-1]:{hover_fmt}}",
                            color))
    span = float((frame[y] * scale).max() - (frame[y] * scale).min()) if not frame.empty else 0.0
    _gutter_labels(fig, entries, span)
    fig.update_layout(title=title)
    return _base(fig, height=380, y_title=y_title, right=120)


def weekly_battle_figure(battles: pd.DataFrame, names: list[str], title: str, y_title: str,
                         hover_fmt: str, suffix: str = "", scale: float = 1.0) -> go.Figure:
    """One weekly total per athlete, bars side by side; ``scale`` converts to display units."""
    fig = go.Figure()
    for index, name in enumerate(names):
        fig.add_trace(go.Bar(
            x=battles["week_start"], y=battles[name] * scale, name=name,
            marker=dict(color=athlete_color(index), cornerradius=3),
            hovertemplate=f"{name}: %{{y:{hover_fmt}}}{suffix}<extra></extra>",
        ))
    fig.update_layout(title=title, barmode="group", bargap=0.2, bargroupgap=0.05)
    fig.update_xaxes(hoverformat="Week of %d %b %Y")
    return _base(fig, height=360, y_title=y_title)


def shared_runs_figure(shared: pd.DataFrame, you: str, friend: str,
                       units: Units = METRIC) -> go.Figure:
    """Seconds per km (or mile) by which you were faster (up) or slower (down) on each run."""
    gap = units.pace(shared["pace_gap_s"])
    fig = go.Figure(go.Bar(
        x=shared["date"], y=gap,
        marker=dict(color=[athlete_color(0) if g >= 0 else athlete_color(1) for g in gap],
                    cornerradius=3),
        customdata=shared["name"],
        hovertemplate=("%{x|%d %b %Y} · %{customdata}<br>%{y:+.0f} s/"
                       f"{units.distance_label}<extra></extra>"),
    ))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_layout(title=f"Pace gap on runs together (up = {you} faster, down = "
                            f"{friend} faster)", showlegend=False, hovermode="closest",
                      bargap=0.3)
    return _base(fig, height=320, y_title=f"Seconds per {units.distance_label}")


# Terrain bands: low-chroma tints, so they sit behind the lines without being
# mistaken for a series colour.  Identity is also carried by the legend and the
# "+12 m" / "-12 m" labels, never by tint alone.
CLIMB_FILL = "#f3dcc8"
DESCENT_FILL = "#d6e4f5"
TERRAIN_LABEL_LIMIT = 14


def terrain_figure(route: pd.DataFrame, segs: pd.DataFrame, units: Units = METRIC) -> go.Figure:
    """Elevation, pace and heart rate stacked on one distance axis, climbs and
    descents shaded through all three panels.

    Three panels rather than one chart with three y-scales: each measure keeps
    its own honest axis, and the shared x-axis and bands do the lining up.
    """
    has_hr = bool(route["hr"].notna().any())
    rows = 3 if has_hr else 2
    titles = [f"Elevation ({units.elevation_label})", f"Pace ({units.pace_label})"]
    if has_hr:
        titles.append("Heart rate (bpm)")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.28, 0.4, 0.32][:rows], subplot_titles=titles)
    x = units.distance(route["distance_m"] / 1000.0)
    elevation = route["altitude_m"] * units.per_metre

    fig.add_trace(go.Scatter(
        x=x, y=elevation, mode="lines", name="Elevation", showlegend=False,
        line=dict(color=INK_SECONDARY, width=2),
        hovertemplate=f"%{{y:.0f}} {units.elevation_label}<extra>Elevation</extra>",
    ), row=1, col=1)

    pace = route["pace_min_per_km"].astype("float64")
    gap = route["gap_pace_min_per_km"].astype("float64")
    fig.add_trace(go.Scatter(
        x=x, y=units.pace(pace), mode="lines", name="Pace",
        line=dict(color=SERIES["fitness"], width=2),
        customdata=[units.format_pace(v) for v in pace],
        hovertemplate="%{customdata}<extra>Pace</extra>",
    ), row=2, col=1)
    if gap.notna().any():
        fig.add_trace(go.Scatter(
            x=x, y=units.pace(gap), mode="lines", name="Grade-adjusted pace",
            line=dict(color=INK_SECONDARY, width=2, dash="dot"),
            customdata=[units.format_pace(v) for v in gap],
            hovertemplate="%{customdata}<extra>GAP</extra>",
        ), row=2, col=1)
    if has_hr:
        fig.add_trace(go.Scatter(
            x=x, y=route["hr"], mode="lines", name="Heart rate",
            line=dict(color=SERIES["fatigue"], width=2),
            hovertemplate="%{y:.0f} bpm<extra>Heart rate</extra>",
        ), row=3, col=1)

    # Bands go in after the traces: plotly skips shapes on subplots still empty.
    for _, seg in segs[segs["kind"] != "flat"].iterrows():
        fig.add_vrect(x0=units.distance(seg["start_m"] / 1000.0),
                      x1=units.distance(seg["end_m"] / 1000.0),
                      fillcolor=CLIMB_FILL if seg["kind"] == "climb" else DESCENT_FILL,
                      opacity=1.0, layer="below", line_width=0, row="all", col=1)

    # Legend swatches for the bands.
    for name, fill in (("Climb", CLIMB_FILL), ("Descent", DESCENT_FILL)):
        if (segs["kind"] == name.lower()).any():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker=dict(symbol="square", size=14, color=fill, line=dict(width=1, color=AXIS)),
            ), row=1, col=1)

    # "+12 m" / "-12 m" over each climb and descent (the biggest ones on a hilly run).
    hills = segs[segs["kind"] != "flat"]
    if len(hills) > TERRAIN_LABEL_LIMIT:
        hills = hills.loc[hills["change_m"].abs().nlargest(TERRAIN_LABEL_LIMIT).index]
    top = float(np.nanmax(elevation)) if elevation.notna().any() else 0.0
    for _, seg in hills.iterrows():
        fig.add_annotation(
            x=units.distance((seg["start_m"] + seg["end_m"]) / 2000.0), y=top,
            text=units.format_elevation(seg["change_m"]).replace("-", "−"),
            showarrow=False, yanchor="bottom", font=dict(size=11, color=INK_SECONDARY),
            row=1, col=1)

    _base(fig, height=620 if has_hr else 460)
    fig.update_layout(title="Elevation, pace and heart rate along the run",
                      hovermode="x unified", margin=dict(t=110))
    for title in titles:
        fig.update_annotations(selector=dict(text=title), font=dict(size=12, color=INK_SECONDARY),
                               xanchor="left", x=0)
    fig.update_xaxes(title_text="", showline=True, linecolor=AXIS)
    fig.update_xaxes(title_text=f"Distance ({units.distance_label})", row=rows, col=1)
    span = float(np.nanmax(elevation) - np.nanmin(elevation)) if elevation.notna().any() else 0
    pad = max(span * 0.25, 5.0 * units.per_metre)
    fig.update_yaxes(range=[float(np.nanmin(elevation)) - pad * 0.3,
                            float(np.nanmax(elevation)) + pad], row=1, col=1)
    if has_hr:  # the warm-up climb in heart rate should not squash the rest
        hr = route["hr"].dropna()
        low, high = hr.quantile(0.02), hr.quantile(0.98)
        fig.update_yaxes(range=[low - 5, high + 5], row=3, col=1)
    # GPS pace spikes at turns and stops: scale the pace axis to the middle 96%.
    shown = units.pace(pd.concat([pace, gap]).dropna())
    if not shown.empty:
        low, high = shown.quantile(0.02), shown.quantile(0.98)
        _pace_axis(fig, pd.Series([low, high]), row=2, col=1)
        fig.update_yaxes(range=[np.ceil(high * 4) / 4, np.floor(low * 4) / 4],
                         autorange=False, row=2, col=1)
    return fig
