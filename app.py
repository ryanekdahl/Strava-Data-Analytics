'''Strava Run Analytics -- Streamlit dashboard.

Run with ``streamlit run app.py``.  The app defaults to the bundled synthetic
export so it is never empty.  Visitors upload their own Strava archive in the
sidebar (unpacked into a private per-session temp folder, see ``src/upload.py``);
when running locally you can instead drop ``activities.csv`` into ``data/raw/``
and the sidebar offers it as another source.

Six tabs: the stats Strava itself shows (Overview, Activities, Best efforts,
Heart rate, Gear) plus Fitness & Freshness with the acute:chronic workload
ratio for runners who manage their training load.

If the export's ``activities/`` folder of GPS recordings sits next to the CSV,
the app reads those too (once, then from a cache) for true in-run best efforts,
second-by-second heart-rate zones and per-kilometre splits.  Rides, swims and
other sessions count toward Fitness and Fatigue on the same load scale as runs.

The sidebar narrows every tab to a period (Fitness and ACWR still count the
runs before it) and switches the display between kilometres and miles; all
computation stays metric underneath.

Every number on screen is computed by the modules in ``src/`` -- see the README's
Methodology section for the formulas behind Fitness, Form, ACWR and Riegel.
'''

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import pace_heat, plots, replay, streams, terrain, upload
from src.consistency import (
    SWEET_SPOT, acwr_status, compute_acwr, monthly_volume, summary_kpis,
    weekly_streaks, weekly_volume,
)
from src.gear import DEFAULT_RETIRE_KM, INACTIVE_DAYS, gear_summary, unassigned_share
from src.generate_sample import generate_sample, sample_is_current
from src.heart_rate import (
    hr_coverage, monthly_zone_hours, stream_monthly_zone_hours, stream_zone_distribution,
    zone_distribution,
)
from src.near_pr import (
    GETTING_CLOSE, NEW_PR, STATUS_ORDER, WITHIN_REACH, chances, near_pr_table, whole_run_efforts,
)
from src.loader import SAMPLE_CSV, find_real_export, load_activities, load_other_activities
from src.performance import (
    RECENT_WINDOW_DAYS, RIEGEL_EXPONENT, Effort, best_recent_effort, format_duration, format_pace,
    personal_records, predict_races,
)
from src.sports import compute_other_load
from src.run_search import filter_runs
from src.race_calculator import (
    CALCULATOR_DISTANCES, calculator_predictions, parse_duration,
)
from src.training_load import (
    CTL_DAYS, ATL_DAYS, DEFAULT_HR_MAX, DEFAULT_HR_REST, compute_activity_load,
    compute_fitness_freshness, daily_load_series, estimate_threshold_pace, form_label,
)
from src.units import IMPERIAL, UNIT_SYSTEMS, Units

# Relative periods count back from the latest run, not from today, so an old
# export still opens on its own last months rather than on an empty range.
PERIODS = {"All time": None, "Last 12 months": 365, "Last 6 months": 182,
           "Last 3 months": 91, "Last 4 weeks": 28}
CUSTOM_PERIOD = "Custom range"
# Every visitor's upload is its own cache entry, so a hosted app bounds the caches.
CACHE_TTL = "6h"
CACHE_ENTRIES = 24

st.set_page_config(page_title="Strava Run Analytics", page_icon=":running_shirt_with_sash:",
                   layout="wide")


@st.cache_data(show_spinner="Crunching activities...", ttl=CACHE_TTL, max_entries=CACHE_ENTRIES)
def build_analytics(path: str, _mtime: float, hr_rest: float, hr_max: float) -> dict:
    """Run the whole pipeline once per file and setting; ``_mtime`` busts the cache on edit.

    Non-run sessions' load joins the runs' in the daily series, as Strava's own
    Fitness & Freshness does, so Fitness, Fatigue and ACWR see all training.
    """
    activities = load_activities(path)
    with_load = compute_activity_load(activities, hr_rest=hr_rest, hr_max=hr_max)
    other = compute_other_load(load_other_activities(path), hr_rest=hr_rest, hr_max=hr_max)
    counted = (pd.concat([with_load[["date", "load"]], other[["date", "load"]]])
               if not other.empty else with_load)
    daily = daily_load_series(counted)
    return {
        "activities": activities,
        "other": other,
        "fitness": compute_fitness_freshness(daily),
        "acwr": compute_acwr(daily),
        "weekly": weekly_volume(activities),
        "monthly": monthly_volume(activities),
        "records": personal_records(activities),
        "threshold_pace": estimate_threshold_pace(activities),
        "zones": zone_distribution(activities, hr_rest, hr_max),
        "monthly_zones": monthly_zone_hours(activities, hr_rest, hr_max),
        "hr_rest": hr_rest,
        "hr_max": hr_max,
    }


@st.cache_data(show_spinner="Reading GPS summaries...", ttl=CACHE_TTL, max_entries=CACHE_ENTRIES)
def read_streams(path: str, _mtime: float, _cache_token: int) -> streams.StreamData:
    """Gather the cached per-run summaries; the token changes whenever the cache does."""
    return streams.load_stream_data(load_activities(path), path)


def prepare_streams(path: Path, activities: pd.DataFrame) -> streams.StreamData | None:
    """Parse any GPS files not cached yet (with a progress bar), then load the summaries.

    Returns ``None`` when the export names no GPS files at all.
    """
    if not streams.stream_files(activities, path):
        return None
    pending = streams.pending_files(activities, path)
    if pending:
        cache_dir = streams.cache_dir_for(path)
        bar = st.progress(0.0, text=f"Reading {len(pending):,} GPS recordings (first time only)...")
        for done, (activity_id, file) in enumerate(pending, start=1):
            streams.cache_summary(activity_id, file, cache_dir)
            bar.progress(done / len(pending),
                         text=f"Reading GPS recordings: {done:,} of {len(pending):,}")
        bar.empty()
    cache_dir = streams.cache_dir_for(path)
    token = cache_dir.stat().st_mtime_ns if cache_dir.is_dir() else 0
    return read_streams(str(path), path.stat().st_mtime, token)


def choose_source() -> Path:
    """Sidebar data-source picker; falls back to the synthetic sample."""
    st.sidebar.title("Strava Run Analytics")
    if not sample_is_current() or not (SAMPLE_CSV.parent / "activities").is_dir():
        with st.spinner("Generating sample data..."):
            generate_sample(SAMPLE_CSV)

    uploaded = upload_export()
    sources = {"Sample data (synthetic)": SAMPLE_CSV}
    real_export = find_real_export()
    if real_export is not None:
        sources[f"My export ({real_export.name})"] = real_export
    if uploaded is not None:
        sources[UPLOADED_SOURCE] = uploaded

    # A fresh upload switches the view to it; otherwise the visitor's pick stands.
    if st.session_state.pop("switch_to_upload", False):
        st.session_state["data_source"] = UPLOADED_SOURCE
    if st.session_state.get("data_source") not in sources:
        st.session_state["data_source"] = list(sources)[-1]
    label = st.sidebar.radio("Data source", list(sources), key="data_source")
    return sources[label]


UPLOADED_SOURCE = "My uploaded export"
EXPORT_STEPS = (
    "1. On strava.com, open **Settings -> My Account**.\n"
    "2. Under **Download or Delete Your Account**, click **Get Started**, then "
    "**Request Your Archive**.\n"
    "3. Strava emails you a link, usually within a few hours. Download the `.zip`.\n"
    "4. Upload that zip in the sidebar, as it is -- no need to unzip it."
)
PRIVACY_NOTE = (
    "Your export is unpacked into a private temporary folder for your browser session "
    "only. It is never shared with other visitors, and it is deleted when you click "
    "**Remove my data** or a few hours after you leave. Only `activities.csv` and the "
    "GPS files are read; the photos and messages in the archive are skipped."
)


def session_workspace() -> Path:
    """This browser session's private folder for uploads, created on first use."""
    workspace = st.session_state.get("workspace")
    if workspace is None or not Path(workspace).is_dir():
        upload.purge_old_workspaces()
        workspace = upload.new_workspace()
        st.session_state["workspace"] = workspace
    return Path(workspace)


def forget_upload() -> None:
    """Delete this session's uploaded export and clear the widgets holding it."""
    workspace = st.session_state.pop("workspace", None)
    if workspace is not None:
        upload.remove_workspace(Path(workspace))
    for key in ("uploaded_export", "upload_id"):
        st.session_state.pop(key, None)


def upload_export() -> Path | None:
    """Sidebar uploader for the visitor's own Strava archive; returns its ``activities.csv``.

    The archive is unpacked once per distinct file; reruns reuse it.
    """
    st.sidebar.subheader("Your data")
    # Streamlit cannot clear a file uploader from code, so "Remove my data" swaps in a new one.
    generation = st.session_state.setdefault("uploader_generation", 0)
    file = st.sidebar.file_uploader(
        "Upload your Strava export", type=["zip", "csv"], key=f"export_upload_{generation}",
        help="The .zip Strava emails you, unchanged -- or just its activities.csv "
             "(without GPS files, a few views are less detailed).")
    with st.sidebar.expander("How to get your export"):
        st.markdown(EXPORT_STEPS)
        st.caption(PRIVACY_NOTE)

    if file is None:
        if st.session_state.get("uploaded_export") is not None:
            forget_upload()  # the visitor cleared the uploader
        st.session_state.pop("upload_error", None)
        return None
    upload_id = f"{file.file_id}:{file.name}:{file.size}"
    failed = st.session_state.get("upload_error")
    if failed is not None and failed[0] == upload_id:
        st.sidebar.error(failed[1])
        return None
    if st.session_state.get("upload_id") != upload_id:
        return unpack_upload(file, upload_id)

    csv_path = Path(st.session_state["uploaded_export"])
    if not csv_path.is_file():  # the temporary folder expired while the tab sat open
        forget_upload()
        st.session_state["uploader_generation"] += 1
        st.sidebar.warning("Your uploaded data expired after a long idle spell; "
                           "please upload it again.")
        return None
    os.utime(csv_path.parents[1])  # keep an active session's folder from being purged
    if st.sidebar.button("Remove my data", key="remove_upload",
                         help="Deletes your uploaded export and any friends you added, now."):
        forget_upload()
        st.session_state["uploader_generation"] += 1
        st.rerun()
    return csv_path


def unpack_upload(file, upload_id: str) -> None:
    """Unpack and check a newly uploaded export, then rerun onto it (or record why it failed)."""
    forget_upload()  # a replaced upload takes its predecessor's friends with it
    try:
        with st.spinner(f"Unpacking {file.name}..."):
            csv_path = upload.extract_export(file, file.name, session_workspace())
            runs = load_activities(csv_path)
        problem = None if not runs.empty else (
            "That export has no runs in it, so there is nothing to analyse yet.")
    except (upload.ExportError, ValueError, pd.errors.ParserError,
            UnicodeDecodeError, OSError) as err:
        problem = f"That file could not be read as a Strava export: {err}"
    if problem is not None:
        forget_upload()
        st.session_state["upload_error"] = (upload_id, problem)
        st.sidebar.error(problem)
        return None
    st.session_state.update(uploaded_export=csv_path, upload_id=upload_id,
                            switch_to_upload=True)
    st.session_state.pop("upload_error", None)
    st.rerun()


def choose_units() -> Units:
    """Sidebar switch between kilometres and miles; display only."""
    name = st.sidebar.radio("Units", list(UNIT_SYSTEMS), horizontal=True, key="units")
    return UNIT_SYSTEMS[name]


def choose_period(activities: pd.DataFrame, box) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Period picker drawn into ``box``; returns ``(start, end)`` with ``end`` exclusive."""
    box.subheader("Period")
    first = activities["date"].min().normalize()
    last = activities["date"].max().normalize()
    end = last + pd.Timedelta(days=1)
    years = [str(year) for year in sorted(activities["date"].dt.year.unique(), reverse=True)]
    choice = box.selectbox(
        "Show runs from", list(PERIODS) + years + [CUSTOM_PERIOD], key="period",
        help="\"Last ...\" periods count back from your latest run.")
    box.caption("Fitness, Form and ACWR still count the runs before the period; shoe "
                "mileage always uses your whole history.")
    if choice == CUSTOM_PERIOD:
        picked = box.date_input("Between", value=(first.date(), last.date()),
                                min_value=first.date(), max_value=last.date(),
                                key="period_range")
        if len(picked) != 2:  # mid-selection: only the start date has been clicked
            return first, end
        return pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + pd.Timedelta(days=1)
    if choice in years:
        return pd.Timestamp(int(choice), 1, 1), pd.Timestamp(int(choice) + 1, 1, 1)
    days = PERIODS[choice]
    return (first if days is None else end - pd.Timedelta(days=days)), end


def restrict_period(data: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """The analytics for runs from ``start`` up to (not including) ``end``.

    Fitness, Fatigue, Form and ACWR are computed on the whole history and only
    then cut to the period, so a period opens on the fitness actually carried
    into it rather than on zero.  Everything built from the runs themselves is
    recomputed on the period's runs, and ``all_activities`` keeps every run for
    the shoe-mileage tab.  ``previous`` holds the runs of the equally long window
    just before the period, for the Overview's comparison (``None`` when there is
    no such window).
    """
    data = {**data, "all_activities": data["activities"], "previous": None}
    activities = data["activities"]
    if activities.empty or (activities["date"].min() >= start
                            and activities["date"].max() < end):
        return data

    def cut(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["date"] >= start) & (frame["date"] < end)].reset_index(drop=True)

    runs = cut(activities)
    hr_rest, hr_max = data["hr_rest"], data["hr_max"]
    return {
        **data,
        "previous": previous_period_runs(activities, start, end),
        "activities": runs,
        "other": cut(data["other"]),
        "fitness": cut(data["fitness"]),
        "acwr": cut(data["acwr"]),
        "weekly": weekly_volume(runs),
        "monthly": monthly_volume(runs),
        "records": personal_records(runs),
        "zones": zone_distribution(runs, hr_rest, hr_max),
        "monthly_zones": monthly_zone_hours(runs, hr_rest, hr_max),
    }


def previous_period_runs(activities: pd.DataFrame, start: pd.Timestamp,
                         end: pd.Timestamp) -> pd.DataFrame | None:
    """The runs in the window of the same length just before ``start``.

    A calendar year compares with the year before it.  ``None`` when that window
    starts before the export does, since a half-recorded window would make any
    period look like a big step up.
    """
    if start.is_year_start and end == start + pd.DateOffset(years=1):
        prev_start = start - pd.DateOffset(years=1)
    else:
        prev_start = start - (end - start)
    if prev_start < activities["date"].min().normalize():
        return None
    dates = activities["date"]
    return activities[(dates >= prev_start) & (dates < start)].reset_index(drop=True)


def period_delta(now: float, before: float, unit: str = "") -> str:
    """A metric delta like ``+12 km vs previous period``."""
    return f"{now - before:+,.0f}{unit} vs previous period"


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


def max_hr_hint(activities: pd.DataFrame, hr_max: float) -> None:
    """Point out a recorded heart rate above the Max HR setting."""
    recorded = activities["max_hr"].dropna()
    if recorded.empty:
        return
    top = float(recorded.max())
    if top > hr_max:
        st.sidebar.warning(
            f"Your export has a heart rate of {top:.0f} bpm, above the Max HR of "
            f"{hr_max:.0f} set above. If that reading is real (not a strap glitch), "
            f"raise Max HR to match.")
    else:
        st.sidebar.caption(f"Highest heart rate in your export: {top:.0f} bpm.")


def total_elevation(runs: pd.DataFrame, units: Units) -> str:
    """Summed elevation gain, unsigned, in the display units."""
    gain = runs["elevation_gain_m"].sum(min_count=1)
    if pd.isna(gain):
        return "--"
    return f"{gain * units.per_metre:,.0f} {units.elevation_label}"


def overview_tab(data: dict) -> None:
    """Strava's profile totals and training log: volume, streaks, week by week."""
    activities, weekly, units = data["activities"], data["weekly"], data["units"]
    kpis = summary_kpis(activities)

    previous = data.get("previous")
    deltas = {"distance": None, "runs": None, "hours": None}
    if previous is not None:
        before = summary_kpis(previous)
        deltas = {
            "distance": period_delta(units.distance(kpis["total_distance_km"]),
                                     units.distance(before["total_distance_km"]),
                                     f" {units.distance_label}"),
            "runs": period_delta(kpis["total_runs"], before["total_runs"]),
            "hours": period_delta(kpis["total_hours"], before["total_hours"], " h"),
        }

    top = st.columns(4)
    top[0].metric("Total distance", units.format_distance(kpis["total_distance_km"], 0),
                  delta=deltas["distance"])
    top[1].metric("Runs", f"{kpis['total_runs']:,}", delta=deltas["runs"])
    top[2].metric("Moving time", f"{kpis['total_hours']:,.0f} h", delta=deltas["hours"])
    top[3].metric("Elevation gain", total_elevation(activities, units))

    streaks = weekly_streaks(weekly)
    active = weekly[weekly["runs"] > 0]
    bottom = st.columns(4)
    bottom[0].metric("Current streak", f"{streaks.current} weeks",
                     help="Consecutive calendar weeks with at least one run.")
    bottom[1].metric("Longest streak", f"{streaks.longest} weeks")
    bottom[2].metric("Typical week", units.format_distance(active["distance_km"].median(), 0)
                     if not active.empty else "--",
                     help="Median distance over weeks with at least one run.")
    bottom[3].metric("Biggest week", units.format_distance(weekly["distance_km"].max(), 0)
                     if not weekly.empty else "--")

    pr_teaser(data["near_pr"])
    st.plotly_chart(plots.weekly_volume_figure(weekly, units), width="stretch")
    st.caption("Distance run each calendar week. Empty weeks are shown as gaps, not skipped.")
    st.plotly_chart(plots.monthly_volume_figure(data["monthly"], units), width="stretch")
    st.caption("The same volume rolled up by month.")


def fitness_tab(data: dict) -> None:
    """Strava's Fitness & Freshness, plus the acute:chronic workload ratio."""
    fitness, acwr, units = data["fitness"], data["acwr"], data["units"]
    latest = fitness.iloc[-1]
    week_ago = fitness.iloc[-8] if len(fitness) > 8 else fitness.iloc[0]
    latest_acwr = acwr["acwr"].dropna()
    acwr_value = float(latest_acwr.iloc[-1]) if not latest_acwr.empty else float("nan")

    cards = st.columns(4)
    cards[0].metric("Fitness (CTL)", f"{latest['ctl']:.0f}",
                    delta=f"{latest['ctl'] - week_ago['ctl']:+.1f} vs last week")
    cards[1].metric("Fatigue (ATL)", f"{latest['atl']:.0f}",
                    delta=f"{latest['atl'] - week_ago['atl']:+.1f} vs last week",
                    delta_color="inverse")
    cards[2].metric("Form (TSB)", f"{latest['tsb']:+.0f}",
                    delta=f"{latest['tsb'] - week_ago['tsb']:+.1f} vs last week")
    cards[3].metric("ACWR", f"{acwr_value:.2f}" if acwr_value == acwr_value else "--",
                    delta=acwr_status(acwr_value), delta_color="off")
    st.info(f"**Form today: {latest['tsb']:+.0f} -- {form_label(float(latest['tsb']))}**")

    st.plotly_chart(plots.fitness_freshness_figure(fitness, CTL_DAYS, ATL_DAYS), width="stretch")
    st.caption(
        f"Fitness (blue) is the slow {CTL_DAYS}-day average of your training load -- how much "
        f"work your body is used to. Fatigue (orange) is the fast {ATL_DAYS}-day average -- how "
        "much you did lately. Form (green) is the gap between them: positive means "
        "rested and race-ready, negative means you are carrying fatigue from a hard block."
        + other_sports_note(data)
    )

    st.subheader("Workload ratio (ACWR)")
    st.plotly_chart(plots.acwr_figure(acwr), width="stretch")
    low, high = SWEET_SPOT
    st.caption(
        f"Acute:chronic workload ratio -- this week's average daily load divided by the "
        f"last four weeks'. Inside the shaded {low:g}-{high:g} band you are building "
        f"sustainably; well above it you are ramping faster than your body is adapting, "
        f"which is the classic injury-risk pattern."
    )

    gap_share = data["activities"]["gap_pace_min_per_km"].notna().mean()
    with st.expander("How training load is computed"):
        st.markdown(
            f"""
            Each run scores `duration_hours x intensity_factor^2 x 100`, where
            `intensity_factor = threshold_pace / run_pace`. One hour at threshold pace
            therefore scores 100 points. Your threshold pace is estimated as the fastest
            decile of runs over 5 km: **{units.format_pace(data['threshold_pace'])}**.
            Where the export has Strava's grade-adjusted pace ({gap_share:.0%} of runs here),
            that is the pace used, so a slow climb is scored as the hard work it is.

            When a run has heart rate, a Banister TRIMP score (rescaled to the same
            100-points-per-threshold-hour axis) is blended in 50/50, because HR catches
            effort that pace misses -- heat, hills and fatigue. Runs without HR use the
            pace score alone. Fitness and Fatigue are then exponentially weighted moving
            averages of daily load with {CTL_DAYS}-day and {ATL_DAYS}-day time constants, and
            Form is yesterday's Fitness minus yesterday's Fatigue. Rides, swims and other
            sessions are scored on the same axis from heart rate, or a default intensity
            per sport when they have none.
            """
        )


def other_sports_note(data: dict) -> str:
    """A sentence saying how many non-run sessions count toward Fitness."""
    other = data["other"]
    if other.empty:
        return ""
    hours = other["moving_time_s"].sum() / 3600.0
    return (f" Fitness, Fatigue and ACWR also count {len(other):,} rides, swims and other "
            f"sessions ({hours:,.0f} h) in this period.")


def reference_effort(activities: pd.DataFrame, stream_data, units: Units) -> tuple[Effort, str]:
    """The effort race predictions start from, and a sentence saying where it came from.

    With GPS recordings that is the fastest 5K *inside* any run of the last 90
    days; without them, the fastest whole run of at least 5 km.
    """
    if stream_data is not None and not stream_data.best.empty:
        since = activities["date"].max() - pd.Timedelta(days=RECENT_WINDOW_DAYS)
        table = streams.best_effort_table(stream_data.best, activities, since)
        five = table[table["distance"] == "5K"]
        if not five.empty:
            row = five.iloc[0]
            return (Effort(5.0, float(row["time_s"])),
                    f"your fastest 5K inside any run of the last {RECENT_WINDOW_DAYS} days, "
                    f"{format_duration(row['time_s'])} ({row['date']:%d %b})")
    effort = best_recent_effort(activities)
    return effort, (f"your best recent run, {units.format_distance(effort.distance_km, 2)} in "
                    f"{format_duration(effort.time_s)}")


def format_gap(seconds: float) -> str:
    """A time gap as ``4 s`` under a minute, ``1:05`` above."""
    return f"{seconds:.0f} s" if seconds < 59.5 else format_duration(seconds)


def pr_teaser(near: pd.DataFrame) -> None:
    """One line on the Overview pointing at the PRs within reach."""
    within = chances(near)
    if within.empty:
        return
    listed = ", ".join(f"**{r.distance}** ({format_gap(r.gap_s)} off)"
                       for r in within.head(3).itertuples())
    count = len(within)
    st.info(f"{count} PR{'s' if count > 1 else ''} within reach: {listed}. "
            "See **Best efforts** for the pace it would take.", icon=":material/target:")


def pr_watch(near: pd.DataFrame, units: Units) -> None:
    """How close recent form is to each PR, and what it would take to break the close ones."""
    st.subheader("PR watch")
    if near.empty:
        st.info("No best efforts yet to compare against.")
        return
    counts = near["status"].value_counts()
    cards = st.columns(3)
    cards[0].metric(f"New PRs (last {RECENT_WINDOW_DAYS} days)", f"{counts.get(NEW_PR, 0)}")
    cards[1].metric("Within reach", f"{counts.get(WITHIN_REACH, 0)}",
                    help="Recent best no more than 3% slower than the PR.")
    cards[2].metric("Getting close", f"{counts.get(GETTING_CLOSE, 0)}",
                    help="Recent best 3-8% slower than the PR.")

    for r in chances(near).head(3).itertuples():
        st.success(
            f"**{r.distance}: {format_gap(r.gap_s)} off your PR.** Your best in the last "
            f"{RECENT_WINDOW_DAYS} days is {format_duration(r.recent_time_s)} "
            f"({r.recent_date:%d %b}); the record is {format_duration(r.pr_time_s)} from "
            f"{r.pr_date:%d %b %Y}. Hold **{units.format_pace(r.target_pace_min_per_km)}** "
            f"to break it.", icon=":material/target:")

    ordered = near.assign(order=near["status"].map(STATUS_ORDER.index))
    ordered = ordered.sort_values(["order", "gap_pct", "distance_km"])
    st.dataframe(pd.DataFrame({
        "Distance": ordered["distance"],
        "Status": ordered["status"],
        "PR": ordered["pr_time_s"].map(format_duration),
        "Set on": ordered["pr_date"].dt.strftime("%d %b %Y"),
        f"Best, last {RECENT_WINDOW_DAYS} days": ordered["recent_time_s"].map(
            lambda t: format_duration(t) if pd.notna(t) else "--"),
        "Gap": [("--" if pd.isna(g) else "PR" if s == NEW_PR else f"+{format_gap(g)} ({p:.1f}%)")
                for g, p, s in zip(ordered["gap_s"], ordered["gap_pct"], ordered["status"])],
        "Pace to beat": ordered["target_pace_min_per_km"].map(units.format_pace),
    }), hide_index=True, width="stretch")
    st.caption(
        f"Your fastest effort of the last {RECENT_WINDOW_DAYS} days at each distance against "
        "your all-time record, whatever period the sidebar shows. Strava only marks a best "
        "effort once it falls; this flags the records you are close to *before* you break "
        "them. Pace to beat is the even pace that takes one second off the PR."
    )


def best_efforts_tab(data: dict, stream_data) -> None:
    """Strava's best efforts and PRs, near-PR alerts, and race predictions from current form."""
    activities, records, units = data["activities"], data["records"], data["units"]
    pr_watch(data["near_pr"], units)
    if stream_data is not None and not stream_data.best.empty:
        gps_best_efforts(activities, stream_data, units)
    else:
        st.subheader("Personal records")
        table = records.rename(columns={"distance": "Distance", "time": "Time"})
        table["Pace"] = table["pace_min_per_km"].map(units.format_pace)
        table["Set on"] = table["date"].dt.strftime("%d %b %Y")
        table["In run"] = table["activity_distance_km"].map(units.format_distance)
        st.dataframe(table[["Distance", "Time", "Pace", "Set on", "In run"]],
                     hide_index=True, width="stretch")
        st.caption("Fastest average pace held over a whole run of at least each distance. "
                   "Add the export's GPS files to find best efforts inside longer runs.")

    left, right = st.columns(2)
    with left:
        st.subheader("Race predictions")
        effort, source = reference_effort(activities, stream_data, units)
        st.dataframe(prediction_table(predict_races(effort), units),
                     hide_index=True, width="stretch")
        st.caption(
            f"Riegel's model (`T2 = T1 x (D2/D1)^{RIEGEL_EXPONENT}`) applied to {source}. "
            "Reliable from 5K to the half; usually optimistic for the marathon."
        )
    with right:
        race_calculator(effort, units)


def gps_best_efforts(activities: pd.DataFrame, stream_data, units: Units) -> None:
    """True in-run best efforts from the GPS recordings, as Strava lists them."""
    st.subheader("Best efforts")
    best = streams.best_effort_table(stream_data.best, activities)
    st.dataframe(pd.DataFrame({
        "Distance": best["distance"],
        "Time": best["time_s"].map(format_duration),
        "Pace": best["pace_min_per_km"].map(units.format_pace),
        "Set on": best["date"].dt.strftime("%d %b %Y"),
        "Inside": best["name"],
    }), hide_index=True, width="stretch")
    st.caption(
        "The fastest stretch of each distance found *anywhere inside* a run, on moving time "
        f"-- the way Strava finds best efforts. From {stream_data.status['parsed']:,} "
        "GPS recordings."
    )


def prediction_table(predictions: pd.DataFrame, units: Units) -> pd.DataFrame:
    """Distance, predicted time and pace (in the chosen units) for display."""
    pace = predictions["predicted_time_s"] / 60.0 / predictions["distance_km"]
    return pd.DataFrame({"Distance": predictions["distance"],
                         "Predicted": predictions["predicted_time"],
                         "Pace": pace.map(units.format_pace)})


def race_calculator(effort, units: Units) -> None:
    """Type in any result and get Riegel predictions at every standard distance."""
    st.subheader("Race-time calculator")
    names = list(CALCULATOR_DISTANCES)
    default_time = format_duration(effort.time_s * (5.0 / effort.distance_km) ** RIEGEL_EXPONENT)
    cols = st.columns(2)
    distance = cols[0].selectbox("Distance", names, index=names.index("5K"))
    entered = cols[1].text_input("Your time (H:MM:SS or MM:SS)", value=default_time,
                                key="calc_time")
    try:
        time_s = parse_duration(entered)
    except ValueError as err:
        st.warning(str(err))
        return
    table = calculator_predictions(CALCULATOR_DISTANCES[distance], time_s, RIEGEL_EXPONENT)
    st.dataframe(prediction_table(table, units), hide_index=True, width="stretch")
    st.caption("Enter a recent race or time trial to see what it implies at other distances.")


def heart_rate_tab(data: dict, stream_data) -> None:
    activities = data["activities"]
    hr_rest, hr_max = data["hr_rest"], data["hr_max"]
    coverage = hr_coverage(activities)
    if coverage == 0:
        st.info("None of these runs recorded heart rate, so there are no zones to show.")
        return

    hr_seconds = stream_data.hr_seconds if stream_data is not None else pd.DataFrame()
    from_streams = not hr_seconds.empty
    if from_streams:
        zones = stream_zone_distribution(hr_seconds, hr_rest, hr_max)
        monthly = stream_monthly_zone_hours(hr_seconds, hr_rest, hr_max)
    else:
        zones, monthly = data["zones"], data["monthly_zones"]

    easy_share = float(zones.loc[zones["zone"] <= 2, "share"].sum())
    cards = st.columns(3)
    cards[0].metric("Runs with heart rate", f"{coverage:.0%}")
    cards[1].metric("Most time in", zones.loc[zones["hours"].idxmax(), "label"])
    cards[2].metric("Easy share (Z1-Z2)", f"{easy_share:.0%}")
    st.caption(
        f"Zones use heart-rate reserve between your resting ({hr_rest:.0f}) and "
        f"max ({hr_max:.0f}) heart rate, set in the sidebar."
        + (f" Counted second by second from {hr_seconds['activity_id'].nunique():,} "
           "GPS recordings with heart rate." if from_streams else "")
    )

    st.plotly_chart(plots.zone_distribution_figure(zones), width="stretch")
    st.caption(
        "How your running time splits across five effort zones, from very easy (Z1) to "
        "all-out (Z5). Many coaches aim for roughly 80% of time in the easy zones."
    )
    st.plotly_chart(plots.monthly_zone_figure(monthly), width="stretch")
    st.caption("The same split month by month, so you can see hard blocks and easy phases.")

    st.dataframe(
        zones.assign(share=zones["share"].map(lambda s: f"{s:.0%}"),
                     hours=zones["hours"].map(lambda h: f"{h:.1f}"))
        .rename(columns={"label": "Zone", "bpm": "Heart rate (bpm)", "runs": "Runs",
                         "hours": "Hours", "share": "Share"})
        [["Zone", "Heart rate (bpm)", "Runs", "Hours", "Share"]],
        hide_index=True, width="stretch",
    )

    if not from_streams:
        st.caption(
            "Approximation: without the GPS recordings each run counts toward the zone of its "
            "average heart rate. Add the export's `activities/` folder for second-by-second zones."
        )


NO_STREAMS_HELP = (
    "This needs the GPS recordings. In your Strava archive they are in the "
    "`activities/` folder next to `activities.csv`; copy that folder into `data/raw/` "
    "too and reload."
)


def pick_run(runs: pd.DataFrame, units: Units, key: str) -> int | None:
    """A run picker narrowed by words in the title and a distance range.

    ``runs`` should already be in display order.  Returns the chosen
    activity id, or ``None`` (after saying so) when nothing matches.
    """
    cols = st.columns([3, 2])
    query = cols[0].text_input("Search titles", key=f"{key}_query",
                               placeholder="e.g. long run, tempo, parkrun")
    shown = units.distance(runs["distance_km"])
    lo, hi = float(np.floor(shown.min())), float(np.ceil(shown.max()))
    min_km = max_km = None
    if hi > lo:
        # Bounds in the key so a new period or unit system starts a fresh slider.
        low, high = cols[1].slider(f"Distance ({units.distance_label})", lo, hi, (lo, hi),
                                   step=0.5, key=f"{key}_distance_{units.distance_label}_{lo}_{hi}")
        if (low, high) != (lo, hi):
            min_km, max_km = low / units.per_km - 1e-9, high / units.per_km + 1e-9
    found = filter_runs(runs, query, min_km, max_km)
    if found.empty:
        st.info("No runs match that search. Try fewer words or a wider distance range.")
        return None
    labels = {int(row.activity_id): f"{row.date:%a %d %b %Y} -- {row.name} -- "
                                    f"{units.format_distance(row.distance_km)}"
              for row in found.itertuples()}
    count = (f"{len(found)} of {len(runs)} runs" if len(found) < len(runs)
             else f"{len(runs)} runs")
    return st.selectbox(f"Run ({count})", list(labels), format_func=labels.get, key=key)


def run_detail_tab(data: dict, stream_data, path: Path) -> None:
    activities, units = data["activities"], data["units"]
    if stream_data is None or stream_data.splits.empty:
        st.info(NO_STREAMS_HELP if stream_data is None or stream_data.status["parsed"] == 0
                else "None of the runs in this period has a readable GPS recording.")
        return
    all_splits = stream_data.mile_splits if units is IMPERIAL else stream_data.splits
    with_splits = activities[activities["activity_id"].isin(stream_data.splits["activity_id"])]
    runs = with_splits.sort_values("date", ascending=False)
    activity_id = pick_run(runs, units, "detail_run")
    if activity_id is None:
        return
    run = runs.set_index("activity_id").loc[activity_id]
    splits = all_splits[all_splits["activity_id"] == activity_id]
    detail = stream_data.runs.set_index("activity_id").loc[activity_id]

    gap, gap_source = run_gap(run, detail)
    cards = st.columns(6)
    cards[0].metric("Distance", units.format_distance(run["distance_km"], 2))
    cards[1].metric("Moving time", format_duration(run["moving_time_s"]))
    cards[2].metric("Pace", units.format_pace(run["pace_min_per_km"]))
    cards[3].metric("Grade-adjusted pace", units.format_pace(gap) if pd.notna(gap) else "--",
                    help=gap_source)
    cards[4].metric("Average HR", f"{run['avg_hr']:.0f} bpm" if pd.notna(run["avg_hr"]) else "--")
    cards[5].metric("Cadence", f"{run['avg_cadence_spm']:.0f} spm"
                    if pd.notna(run["avg_cadence_spm"]) else "--")

    route = streams.route_for(activity_id, data["all_activities"], path)
    route_map(route, units)
    route_replay(route, units)
    terrain_section(route, units)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plots.splits_figure(splits, units), width="stretch")
    with right:
        if splits["avg_hr"].notna().any():
            st.plotly_chart(plots.split_hr_figure(splits, units), width="stretch")
        else:
            st.info("This run was recorded without heart rate.")
    st.caption(
        "Grade-adjusted pace (GAP) is what each split would have been on the flat: slower than "
        "your pace on a descent, faster on a climb. Per split it comes from the recording's "
        "altitude and a stated slope model (see `src/streams.py`), so it can differ a little "
        "from Strava's own."
    )
    st.dataframe(pd.DataFrame({
        units.split_label: splits["split"],
        "Distance": (splits["distance_m"] / 1000.0).map(lambda km: units.format_distance(km, 2)),
        "Time": splits["time_s"].map(format_duration),
        "Pace": splits["pace_min_per_km"].map(units.format_pace),
        "GAP": splits["gap_pace_min_per_km"].map(
            lambda g: units.format_pace(g) if pd.notna(g) else "--"),
        "Heart rate": splits["avg_hr"].map(lambda h: f"{h:.0f}" if pd.notna(h) else "--"),
        "Cadence": splits["cadence_spm"].map(lambda c: f"{c:.0f}" if pd.notna(c) else "--"),
        "Elevation": splits["elevation_change_m"].map(units.format_elevation),
    }), hide_index=True, width="stretch")


def run_gap(run: pd.Series, detail: pd.Series) -> tuple[float, str]:
    """A run's whole-run grade-adjusted pace and a note saying where it came from.

    Strava's own figure from ``activities.csv`` is preferred; failing that, the
    equivalent flat distance computed from the GPS recording's altitude.
    """
    if pd.notna(run.get("gap_pace_min_per_km")):
        return float(run["gap_pace_min_per_km"]), "Strava's grade-adjusted pace from the export."
    flat_m = detail.get("gap_distance_m")
    if pd.notna(flat_m) and flat_m > 0:
        return (float(run["moving_time_s"] / 60.0 / (flat_m / 1000.0)),
                "Estimated from the recording's altitude (the export has no GAP for this run).")
    return float("nan"), "Neither the export nor the recording has the elevation to adjust for."


def terrain_section(route: pd.DataFrame, units: Units) -> None:
    """Elevation, pace and heart rate lined up, and what the climbs and descents cost."""
    if route.empty:
        return
    st.subheader("Hills: elevation, pace and heart rate")
    if route["altitude_m"].notna().sum() < 2:
        st.caption("This recording has no altitude, so there are no hills to line up.")
        return
    segs = terrain.segments(route)
    hills = segs[segs["kind"] != "flat"]
    if hills.empty:
        st.caption(f"This run never rose or fell more than "
                   f"{units.format_elevation(terrain.MIN_CHANGE_M).lstrip('+')} in one go, so "
                   "it counts as flat.")
        return
    st.plotly_chart(plots.terrain_figure(route, segs, units), width="stretch",
                    key="terrain_chart")
    st.caption(
        "All three panels share the distance axis. Orange bands are climbs and blue bands "
        "descents, shaded through every panel so you can read straight down from a hill to "
        "what your pace and heart rate did. The dotted line is grade-adjusted pace: if it "
        "stays level while your pace drops on a climb, you kept your effort even. Heart rate "
        f"lags effort by about {terrain.HR_LAG_S:.0f} seconds, so it often peaks just after "
        f"the top of a climb. A climb or descent must rise or fall at least "
        f"{units.format_elevation(terrain.MIN_CHANGE_M).lstrip('+')}; smaller bumps count as flat."
    )

    summary = terrain.terrain_summary(segs).set_index("kind")
    st.markdown(terrain_reading(summary, units))


def terrain_reading(summary: pd.DataFrame, units: Units) -> str:
    """One plain-English sentence on climbs vs descents (or vs flat when there are no descents)."""
    if "climb" not in summary.index:
        return ""
    other = "descent" if "descent" in summary.index else "flat" if "flat" in summary.index \
        else None
    if other is None:
        return ""
    climb, rest = summary.loc["climb"], summary.loc[other]
    word = "descents" if other == "descent" else "flat stretches"
    pace_gap = (climb["pace_min_per_km"] - rest["pace_min_per_km"]) * 60.0 / units.per_km
    effort_gap = (climb["gap_pace_min_per_km"] - rest["gap_pace_min_per_km"]) * 60.0 / units.per_km
    parts = [f"On the climbs you ran **{abs(pace_gap):.0f} s/{units.distance_label} "
             f"{'slower' if pace_gap > 0 else 'faster'}** than on the {word}"]
    if pd.notna(effort_gap):
        parts.append(f"but by grade-adjusted pace your effort was "
                     + ("**about the same**" if abs(effort_gap) < 5 else
                        f"**{abs(effort_gap):.0f} s/{units.distance_label} "
                        f"{'easier' if effort_gap > 0 else 'harder'}**"))
    sentence = ", ".join(parts)
    if pd.notna(climb["avg_hr"]) and pd.notna(rest["avg_hr"]):
        diff = climb["avg_hr"] - rest["avg_hr"]
        sentence += f"; heart rate averaged **{abs(diff):.0f} bpm {'higher' if diff >= 0 else 'lower'}**"
    return sentence + "."


def route_map(route: pd.DataFrame, units: Units) -> None:
    """The run's route, coloured by pace or heart rate, on plain axes or street tiles."""
    if route.empty:
        st.caption("This recording has no GPS positions (a treadmill run, perhaps), so there "
                   "is no route to draw.")
        return
    cols = st.columns([1, 1])
    options = {"Pace": "pace", "Heart rate": "hr"} if route["hr"].notna().any() \
        else {"Pace": "pace"}
    color_by = cols[0].radio("Colour route by", list(options), horizontal=True, key="route_color")
    basemap = cols[1].toggle(
        "Street map", key="route_basemap",
        help="Draws the route on CARTO / OpenStreetMap tiles. Your browser downloads the "
             "tiles, which tells CARTO roughly where this run was -- the only thing in the "
             "app that goes over the network. Off, the route is drawn with no map behind it.")
    if options[color_by] == "pace":
        pace_heat_map(route, units, basemap)
        return
    st.plotly_chart(plots.route_map_figure(route, options[color_by], units, basemap),
                    width="stretch")
    st.caption(
        "Darker means faster (or a higher heart rate), with the colour scale set by the run's "
        "own 5th-95th percentile so one GPS spike cannot flatten the rest. Pace is smoothed "
        "over a few points either side, so short surges are softened."
    )


def pace_heat_map(route: pd.DataFrame, units: Units, basemap: bool) -> None:
    """The route painted hot (faster) to cool (slower) around the run's median pace."""
    heat = pace_heat.pace_heat(route, units)
    if heat.empty:
        st.caption("This recording has no usable pace along the route.")
        return
    extremes = pace_heat.split_extremes(route, units)
    st.plotly_chart(plots.pace_heatmap_figure(heat, extremes, units, basemap),
                    width="stretch", key="pace_heat_map")
    st.markdown(pace_heat_reading(heat, extremes, units))
    st.caption(
        f"Each stretch of the route is coloured by its pace against this run's own median: "
        f"red is faster, blue slower, grey within about {heat.step * 30:.0f} s of the median. "
        f"Bands are {heat.step * 60:.0f} s wide and the outer two are open-ended, so a stop or "
        f"a GPS spike lands in \"or slower\" instead of stretching the scale. The bar chart "
        f"is both the legend and how much of the distance fell in each band; the strip below "
        f"lays the same colours along the distance, so you can see when each one came."
    )


def pace_heat_reading(heat: pace_heat.PaceHeat, extremes: pd.DataFrame, units: Units) -> str:
    """One sentence that says what the heat map shows, so nobody has to scrub for it."""
    bands = heat.bands
    middle = len(bands) // 2
    faster = float(bands["share"].iloc[:middle].sum())
    slower = float(bands["share"].iloc[middle + 1:].sum())
    near = pace_heat.near_median_share(heat)
    unit = units.distance_label
    sentence = (f"Median pace **{format_pace(heat.median, suffix=' /' + unit)}**: "
                f"{near:.0%} of the distance close to it, {faster:.0%} faster and "
                f"{slower:.0%} slower")
    if not extremes.empty:
        fast, slow = extremes.set_index("kind").loc["fastest"], \
            extremes.set_index("kind").loc["slowest"]
        label = units.split_label.lower()
        sentence += (f". Fastest {label}: **{int(fast['split'])}** "
                     f"({format_pace(fast['pace'], suffix=' /' + unit)}); slowest: "
                     f"**{int(slow['split'])}** ({format_pace(slow['pace'], suffix=' /' + unit)})")
    return sentence + "."


REPLAY_LENGTHS = {"10 s": 10.0, "20 s": 20.0, "40 s": 40.0, "1 min": 60.0}


def route_replay(route: pd.DataFrame, units: Units) -> None:
    """Play the run back: a dot crossing the route on the run's own clock, pace drawn below."""
    frames = replay.replay_frames(route)
    if frames.empty:
        return
    with st.expander("▶ Replay this run", expanded=False):
        length = st.segmented_control(
            "Replay length", list(REPLAY_LENGTHS), default="20 s", key="replay_length",
            help="How long the whole run takes to play back. The dot keeps the run's own "
                 "rhythm at any length: fast stretches still pass quicker than slow ones.")
        marks = replay.milestones(route, units.split_m)
        st.plotly_chart(
            plots.route_replay_figure(route, frames, marks, units,
                                      basemap=st.session_state.get("route_basemap", False),
                                      duration_s=REPLAY_LENGTHS[length or "20 s"]),
            width="stretch", key="route_replay")
        st.caption(
            f"Press Play, or drag the slider to scrub. The dot moves on your moving time, so "
            f"it slows where you slowed and stops are skipped. Numbered markers are each "
            f"{units.split_label.lower()}; the line above the map gives your distance, "
            f"elapsed time, current pace and your last {units.split_label.lower()} split. The "
            f"street map follows the toggle above."
        )


def gear_tab(data: dict) -> None:
    activities, units = data["all_activities"], data["units"]
    low, high, step = (200, 750, 25) if units is IMPERIAL else (300, 1200, 50)
    default = int(round(units.distance(DEFAULT_RETIRE_KM) / step) * step)
    shown = st.slider("Retire shoes at", low, high, default, step=step,
                      format=f"%d {units.distance_label}", key=f"retire_{units.distance_label}")
    retire_km = shown / units.per_km
    gear = gear_summary(activities, retire_km)
    if gear.empty:
        st.info("This export does not record which shoes each run used.")
        return
    active = gear[gear["active"]]
    cards = st.columns(3)
    cards[0].metric("Shoes in use", f"{len(active)}")
    due = active[active["status"] != "Fine"]
    cards[1].metric("Due for replacement", f"{len(due)}")
    cards[2].metric("Runs with no shoe set", f"{unassigned_share(activities):.0%}")
    for row in due.itertuples():
        st.warning(f"**{row.gear}** has {units.format_distance(row.distance_km, 0)} on it -- "
                   + ("past" if row.status == "Past limit" else "close to")
                   + f" the {units.format_distance(retire_km, 0)} limit.")
    st.plotly_chart(plots.gear_figure(gear, retire_km, units), width="stretch")
    st.caption(
        f"Blue shoes were worn in the last {INACTIVE_DAYS} days of your data; grey ones "
        "have not been, and count as retired. Most running shoes are good for 500-800 km "
        "(300-500 miles). Shoe mileage counts every run, whatever the sidebar's period."
    )
    st.dataframe(pd.DataFrame({
        "Shoe": gear["gear"], "Status": gear["status"],
        "Distance": gear["distance_km"].map(lambda km: units.format_distance(km, 0)),
        "Runs": gear["runs"],
        "First run": gear["first_used"].dt.strftime("%d %b %Y"),
        "Last run": gear["last_used"].dt.strftime("%d %b %Y"),
    }), hide_index=True, width="stretch")


def main() -> None:
    path = choose_source()
    period_box = st.sidebar.container()  # filled once the runs are loaded
    units = choose_units()
    hr_rest, hr_max = choose_heart_rate()
    data = build_analytics(str(path), path.stat().st_mtime, hr_rest, hr_max)
    all_activities = data["activities"]
    max_hr_hint(all_activities, hr_max)
    start, end = choose_period(all_activities, period_box)
    data = {**restrict_period(data, start, end), "units": units}
    activities = data["activities"]

    st.title("Strava Run Analytics")
    if activities.empty:
        st.warning(f"No runs between {start:%d %b %Y} and "
                   f"{end - pd.Timedelta(days=1):%d %b %Y}. Pick another period in the sidebar.")
        return
    st.markdown(
        f"Training totals, best efforts and Fitness & Freshness for "
        f"**{len(activities):,} runs** between "
        f"**{activities['date'].min():%d %b %Y}** and **{activities['date'].max():%d %b %Y}**"
        + (f" (of {len(all_activities):,} in the export)."
           if len(activities) < len(all_activities) else ".")
    )
    if path == SAMPLE_CSV:
        st.info(
            "**You are looking at demo data** -- 16 months of a made-up runner's training. "
            "To see your own, upload your Strava export in the sidebar; it takes about a "
            "minute to process. Don't have it yet? Open **How to get your export** in the "
            "sidebar -- Strava usually emails it within a few hours.",
            icon=":material/waving_hand:")
    all_streams = prepare_streams(path, all_activities)
    if all_streams is not None and all_streams.status["failed"]:
        st.caption(f"{all_streams.status['failed']:,} GPS files could not be read and are "
                   "left out of the GPS-based views.")
    stream_data = (streams.restrict(all_streams, activities["activity_id"])
                   if all_streams is not None else None)
    has_gps = all_streams is not None and not all_streams.best.empty
    data["near_pr"] = near_pr_table(all_streams.best if has_gps
                                    else whole_run_efforts(all_activities))

    overview, run_detail, efforts, fitness, heart_rate, gear = st.tabs(
        ["Overview", "Activities", "Best efforts", "Fitness & Freshness", "Heart rate", "Gear"]
    )
    with overview:
        overview_tab(data)
    with run_detail:
        run_detail_tab(data, stream_data, path)
    with efforts:
        best_efforts_tab(data, stream_data)
    with fitness:
        fitness_tab(data)
    with heart_rate:
        heart_rate_tab(data, stream_data)
    with gear:
        gear_tab(data)


if __name__ == "__main__":
    main()
