"""Strava Run Analytics -- Streamlit dashboard.

Run with ``streamlit run app.py``.  The app defaults to the bundled synthetic
export so it is never empty; drop your own ``activities.csv`` into ``data/raw/``
and the sidebar offers it as a second source.

Every number on screen is computed by the modules in ``src/`` -- see the README's
Methodology section for the formulas behind Fitness, Form, ACWR and Riegel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import plots
from src.consistency import (
    SWEET_SPOT, acwr_status, compute_acwr, monthly_volume, summary_kpis,
    weekly_streaks, weekly_volume,
)
from src.generate_sample import generate_sample
from src.heart_rate import hr_coverage, monthly_zone_hours, zone_distribution
from src.loader import SAMPLE_CSV, find_real_export, load_activities
from src.performance import (
    RIEGEL_EXPONENT, best_effort_frontier, best_recent_effort, format_duration, format_pace,
    personal_records, predict_races, recent_vs_all_time, rolling_hr, rolling_pace,
)
from src.race_calculator import (
    CALCULATOR_DISTANCES, EXPONENT_RANGE, calculator_predictions, parse_duration,
)
from src.training_load import (
    DEFAULT_HR_MAX, DEFAULT_HR_REST, compute_activity_load, compute_fitness_freshness,
    daily_load_series, estimate_threshold_pace, form_label,
)

st.set_page_config(page_title="Strava Run Analytics", page_icon=":running_shirt_with_sash:",
                   layout="wide")


@st.cache_data(show_spinner="Crunching activities...")
def build_analytics(path: str, _mtime: float, hr_rest: float, hr_max: float) -> dict:
    """Run the whole pipeline once per file and HR setting; ``_mtime`` busts the cache on edit."""
    activities = load_activities(path)
    with_load = compute_activity_load(activities, hr_rest=hr_rest, hr_max=hr_max)
    daily = daily_load_series(with_load)
    return {
        "activities": activities,
        "daily": daily,
        "fitness": compute_fitness_freshness(daily),
        "acwr": compute_acwr(daily),
        "weekly": weekly_volume(activities),
        "monthly": monthly_volume(activities),
        "frontier": best_effort_frontier(activities),
        "records": personal_records(activities),
        "threshold_pace": estimate_threshold_pace(activities),
        "zones": zone_distribution(activities, hr_rest, hr_max),
        "monthly_zones": monthly_zone_hours(activities, hr_rest, hr_max),
        "hr_rest": hr_rest,
        "hr_max": hr_max,
    }


def choose_source() -> Path:
    """Sidebar data-source picker; falls back to the synthetic sample."""
    st.sidebar.title("Strava Run Analytics")
    if not SAMPLE_CSV.exists():
        with st.spinner("Generating sample data..."):
            generate_sample(SAMPLE_CSV)

    sources = {"Sample data (synthetic)": SAMPLE_CSV}
    real_export = find_real_export()
    if real_export is not None:
        sources[f"My export ({real_export.name})"] = real_export

    label = st.sidebar.radio("Data source", list(sources), index=len(sources) - 1)
    if real_export is None:
        st.sidebar.info(
            "Using synthetic data. To analyse your own runs, request your archive "
            "from Strava (Settings -> My Account -> Download or Delete Your Account) "
            "and drop `activities.csv` into `data/raw/`."
        )
    return sources[label]


def choose_heart_rate() -> tuple[float, float]:
    """Sidebar inputs for resting and max HR, used by the load model and zones."""
    st.sidebar.subheader("Heart rate")
    hr_rest = st.sidebar.number_input("Resting HR (bpm)", 30, 100, int(DEFAULT_HR_REST))
    hr_max = st.sidebar.number_input("Max HR (bpm)", 120, 230, int(DEFAULT_HR_MAX))
    st.sidebar.caption("Defaults are assumptions; your own values sharpen training "
                       "load and heart-rate zones.")
    if hr_max <= hr_rest + 20:
        st.sidebar.warning("Max HR should be well above resting HR; using the defaults.")
        return DEFAULT_HR_REST, DEFAULT_HR_MAX
    return float(hr_rest), float(hr_max)


def overview_tab(data: dict) -> None:
    activities, fitness, acwr = data["activities"], data["fitness"], data["acwr"]
    kpis = summary_kpis(activities)
    latest = fitness.iloc[-1]
    latest_acwr = acwr["acwr"].dropna()
    acwr_value = float(latest_acwr.iloc[-1]) if not latest_acwr.empty else float("nan")

    top = st.columns(4)
    top[0].metric("Total distance", f"{kpis['total_distance_km']:,.0f} km")
    top[1].metric("Runs recorded", f"{kpis['total_runs']:,}")
    top[2].metric("Time on feet", f"{kpis['total_hours']:,.0f} h")
    top[3].metric("Weeks tracked", f"{kpis['weeks']:,}")

    week_ago = fitness.iloc[-8] if len(fitness) > 8 else fitness.iloc[0]
    bottom = st.columns(3)
    bottom[0].metric("Fitness (CTL)", f"{latest['ctl']:.0f}",
                     delta=f"{latest['ctl'] - week_ago['ctl']:+.1f} vs last week")
    bottom[1].metric("Form (TSB)", f"{latest['tsb']:+.0f}",
                     delta=f"{latest['tsb'] - week_ago['tsb']:+.1f} vs last week")
    bottom[2].metric("ACWR", f"{acwr_value:.2f}" if acwr_value == acwr_value else "--",
                     delta=acwr_status(acwr_value), delta_color="off")

    st.caption(
        f"**Form right now: {form_label(float(latest['tsb']))}.** "
        f"**Workload: {acwr_status(acwr_value)}.** "
        f"Fitness is your 42-day average training load, Form is how rested you are "
        f"relative to it, and ACWR compares this week's load with the last month's. "
        f"Threshold pace is estimated at {format_pace(data['threshold_pace'])} from your own runs."
    )


def fitness_tab(data: dict) -> None:
    fitness = data["fitness"]
    st.plotly_chart(plots.fitness_freshness_figure(fitness), width="stretch")
    st.caption(
        "Fitness (blue) is the slow 42-day average of your training load -- how much "
        "work your body is used to. Fatigue (orange) is the fast 7-day average -- how "
        "much you did lately. Form (green) is the gap between them: positive means "
        "rested and race-ready, negative means you are carrying fatigue from a hard block."
    )
    latest = fitness.iloc[-1]
    st.info(f"**Form today: {latest['tsb']:+.0f} -- {form_label(float(latest['tsb']))}**")

    with st.expander("How the training load behind this chart is computed"):
        st.markdown(
            f"""
            Each run scores `duration_hours x intensity_factor^2 x 100`, where
            `intensity_factor = threshold_pace / run_pace`. One hour at threshold pace
            therefore scores 100 points. Your threshold pace is estimated as the fastest
            decile of runs over 5 km: **{format_pace(data['threshold_pace'])}**.

            When a run has heart rate, a Banister TRIMP score (rescaled to the same
            100-points-per-threshold-hour axis) is blended in 50/50, because HR catches
            effort that pace misses -- heat, hills and fatigue. Runs without HR use the
            pace score alone. Fitness and Fatigue are then exponentially weighted moving
            averages of daily load with 42-day and 7-day time constants, and Form is
            yesterday's Fitness minus yesterday's Fatigue.
            """
        )


def performance_tab(data: dict) -> None:
    activities, records = data["activities"], data["records"]
    st.plotly_chart(plots.pace_distance_figure(activities, data["frontier"]), width="stretch")
    st.caption(
        "Every run plotted as distance against pace (note the log scale). The blue line "
        "is your best-effort frontier: the fastest pace you have ever held at each distance."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Personal records")
        table = records.rename(columns={"distance": "Distance", "time": "Time", "pace": "Pace"})
        table["Set on"] = table["date"].dt.strftime("%d %b %Y")
        table["In run"] = table["activity_distance_km"].map(lambda km: f"{km:.1f} km")
        st.dataframe(table[["Distance", "Time", "Pace", "Set on", "In run"]],
                     hide_index=True, width="stretch")
        st.caption(
            "Fastest average pace ever held over a run of at least each distance, "
            "and the run it came from."
        )
    with right:
        st.subheader("Race predictions")
        effort = best_recent_effort(activities)
        predictions = predict_races(effort)
        st.dataframe(
            predictions.rename(columns={"distance": "Distance", "predicted_time": "Predicted",
                                        "predicted_pace": "Pace"})[["Distance", "Predicted", "Pace"]],
            hide_index=True, width="stretch",
        )
        st.caption(
            f"Riegel's model (`T2 = T1 x (D2/D1)^1.06`) applied to your best recent effort: "
            f"{effort.distance_km:.2f} km in {format_duration(effort.time_s)}. "
            f"It is reliable from 5K to the half and optimistic for the marathon."
        )

    race_calculator(effort)

    st.subheader("Recent form vs all-time best")
    comparison = recent_vs_all_time(activities)
    comparison["Gap"] = comparison["gap_pct"].map(
        lambda pct: "--" if pd.isna(pct) else f"{pct:+.1f}%")
    st.dataframe(
        comparison.rename(columns={"distance": "Distance", "all_time_best": "All-time best",
                                   "recent_best": "Best in last 90 days", "status": "Status"})
        [["Distance", "All-time best", "Best in last 90 days", "Gap", "Status"]],
        hide_index=True, width="stretch",
    )
    st.caption("How close your last 90 days are to your best ever; a negative gap is a new PR.")

    trend_left, trend_right = st.columns(2)
    with trend_left:
        st.plotly_chart(plots.rolling_pace_figure(rolling_pace(activities)), width="stretch")
        st.caption("Average pace over a trailing 30 days, weighted by distance. Higher is faster.")
    with trend_right:
        st.plotly_chart(plots.heart_rate_figure(rolling_hr(activities)), width="stretch")
        st.caption("Average heart rate per run. A falling line at a steady pace means fitness gains.")


def race_calculator(effort) -> None:
    """Type in any result and get Riegel predictions at every standard distance."""
    st.subheader("Race-time calculator")
    names = list(CALCULATOR_DISTANCES)
    default_time = format_duration(effort.time_s * (5.0 / effort.distance_km) ** RIEGEL_EXPONENT)
    cols = st.columns([1, 1, 2])
    distance = cols[0].selectbox("Distance", names, index=names.index("5K"))
    entered = cols[1].text_input("Your time (H:MM:SS or MM:SS)", value=default_time)
    exponent = cols[2].slider("Riegel exponent", *EXPONENT_RANGE, value=RIEGEL_EXPONENT,
                              step=0.01)
    try:
        time_s = parse_duration(entered)
    except ValueError as err:
        st.warning(str(err))
        return
    table = calculator_predictions(CALCULATOR_DISTANCES[distance], time_s, exponent)
    st.dataframe(
        table.rename(columns={"distance": "Distance", "predicted_time": "Predicted",
                              "predicted_pace": "Pace"})[["Distance", "Predicted", "Pace"]],
        hide_index=True, width="stretch",
    )
    st.caption(
        "Enter a recent race or time trial to see what it implies at other distances. "
        "The exponent is how much pace fades as distance grows: 1.06 is Riegel's "
        "population fit, lower suits endurance-strong runners, higher suits speed-strong ones."
    )


def heart_rate_tab(data: dict) -> None:
    activities, zones = data["activities"], data["zones"]
    coverage = hr_coverage(activities)
    if coverage == 0:
        st.info("None of these runs recorded heart rate, so there are no zones to show.")
        return

    easy_share = float(zones.loc[zones["zone"] <= 2, "share"].sum())
    cards = st.columns(3)
    cards[0].metric("Runs with heart rate", f"{coverage:.0%}")
    cards[1].metric("Most time in", zones.loc[zones["hours"].idxmax(), "label"])
    cards[2].metric("Easy share (Z1-Z2)", f"{easy_share:.0%}")
    st.caption(
        f"Zones use heart-rate reserve between your resting ({data['hr_rest']:.0f}) and "
        f"max ({data['hr_max']:.0f}) heart rate, set in the sidebar."
    )

    st.plotly_chart(plots.zone_distribution_figure(zones), width="stretch")
    st.caption(
        "How your running time splits across five effort zones, from very easy (Z1) to "
        "all-out (Z5). Many coaches aim for roughly 80% of time in the easy zones."
    )
    st.plotly_chart(plots.monthly_zone_figure(data["monthly_zones"]), width="stretch")
    st.caption("The same split month by month, so you can see hard blocks and easy phases.")

    st.dataframe(
        zones.assign(share=zones["share"].map(lambda s: f"{s:.0%}"),
                     hours=zones["hours"].map(lambda h: f"{h:.1f}"))
        .rename(columns={"label": "Zone", "bpm": "Heart rate (bpm)", "runs": "Runs",
                         "hours": "Hours", "share": "Share"})
        [["Zone", "Heart rate (bpm)", "Runs", "Hours", "Share"]],
        hide_index=True, width="stretch",
    )
    st.caption(
        "Approximation: Strava's activity summary only has each run's average heart rate, "
        "so a whole run counts toward the zone of its average. Interval sessions are "
        "credited to the middle zones even though they swing between easy and hard."
    )


def consistency_tab(data: dict) -> None:
    weekly, acwr = data["weekly"], data["acwr"]
    streaks = weekly_streaks(weekly)
    active = weekly[weekly["runs"] > 0]

    cards = st.columns(4)
    cards[0].metric("Current streak", f"{streaks.current} weeks")
    cards[1].metric("Longest streak", f"{streaks.longest} weeks")
    cards[2].metric("Typical week", f"{active['distance_km'].median():.0f} km"
                    if not active.empty else "--")
    cards[3].metric("Biggest week", f"{weekly['distance_km'].max():.0f} km"
                    if not weekly.empty else "--")
    st.caption("A streak is consecutive calendar weeks containing at least one run.")

    st.plotly_chart(plots.weekly_volume_figure(weekly), width="stretch")
    st.caption("Distance run each calendar week. Empty weeks are shown as gaps, not skipped.")

    st.plotly_chart(plots.monthly_volume_figure(data["monthly"]), width="stretch")
    st.caption("The same volume rolled up by month, which smooths out single-week noise.")

    st.plotly_chart(plots.acwr_figure(acwr), width="stretch")
    low, high = SWEET_SPOT
    st.caption(
        f"Acute:chronic workload ratio -- this week's average daily load divided by the "
        f"last four weeks'. Inside the shaded {low:g}-{high:g} band you are building "
        f"sustainably; well above it you are ramping faster than your body is adapting, "
        f"which is the classic injury-risk pattern."
    )


def main() -> None:
    path = choose_source()
    hr_rest, hr_max = choose_heart_rate()
    data = build_analytics(str(path), path.stat().st_mtime, hr_rest, hr_max)
    activities = data["activities"]

    st.title("Strava Run Analytics")
    st.markdown(
        f"Fitness & Freshness, best efforts and workload management for "
        f"**{len(activities):,} runs** between "
        f"**{activities['date'].min():%d %b %Y}** and **{activities['date'].max():%d %b %Y}**."
    )

    overview, fitness, performance, heart_rate, consistency = st.tabs(
        ["Overview", "Fitness & Freshness", "Performance", "Heart rate", "Consistency"]
    )
    with overview:
        overview_tab(data)
    with fitness:
        fitness_tab(data)
    with performance:
        performance_tab(data)
    with heart_rate:
        heart_rate_tab(data)
    with consistency:
        consistency_tab(data)


if __name__ == "__main__":
    main()
