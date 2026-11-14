"""Strava Run Analytics -- Streamlit dashboard.

Run with ``streamlit run app.py``.  The app defaults to the bundled synthetic
export so it is never empty.  Visitors upload their own Strava archive in the
sidebar (unpacked into a private per-session temp folder, see ``src/upload.py``);
when running locally you can instead drop ``activities.csv`` into ``data/raw/``
and the sidebar offers it as another source.

If the export's ``activities/`` folder of GPS recordings sits next to the CSV,
the app reads those too (once, then from a cache) for true in-run best efforts,
second-by-second heart-rate zones and per-kilometre splits.

Rides, swims, walks and gym sessions in the export are scored on the same
load axis as runs (``src/sports.py``); by default they count toward Fitness,
Fatigue and ACWR, and the All sports tab breaks the load down by sport.

The sidebar narrows every tab to a period (Fitness and ACWR still count the
runs before it) and switches the display between kilometres and miles; all
computation stays metric underneath.

Every number on screen is computed by the modules in ``src/`` -- see the README's
Methodology section for the formulas behind Fitness, Form, ACWR and Riegel.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import pace_zones, plots, streams, terrain, upload
from src.consistency import (
    SWEET_SPOT, acwr_status, compute_acwr, monthly_volume, summary_kpis,
    weekly_streaks, weekly_volume,
)
from src.gear import DEFAULT_RETIRE_KM, INACTIVE_DAYS, gear_summary, unassigned_share
from src.generate_sample import (
    friends_are_current, generate_friends, generate_sample, sample_is_current,
)
from src.head_to_head import (
    FRIENDS_DIR, METRICS, RACE_MEASURES, SAMPLE_FRIENDS_DIR, SHARED_DISTANCE_TOLERANCE,
    SHARED_WINDOW_MIN, TIE, athlete_profile, cumulative_totals, cut_period, find_friend_exports,
    fitness_comparison, head_to_head_table, prediction_comparison, record_comparison,
    remove_friend_export, save_friend_export, scoreboard, shared_runs, weekly_battles,
)
from src.heart_rate import (
    hr_coverage, monthly_zone_hours, run_zone_breakdown, stream_monthly_zone_hours,
    stream_run_zone_breakdown, stream_zone_distribution, zone_distribution,
)
from src.loader import RAW_DIR, SAMPLE_CSV, find_real_export, load_activities, load_other_activities
from src.performance import (
    RECENT_WINDOW_DAYS, RIEGEL_EXPONENT, Effort, best_effort_frontier, best_recent_effort,
    format_duration, personal_records, predict_races, recent_vs_all_time,
    rolling_hr, rolling_pace,
)
from src.personal_models import (
    MIN_PERFORMANCE_RUNS, TAU1_GRID, TAU2_GRID, fit_banister, fit_riegel, modelled_performance,
    performance_index, riegel_efforts,
)
from src.race_plan import (
    MAX_SAFE_RAMP, RACE_READY_FORM, TAPER_SHAPES, plan_weeks, project_fitness,
    race_day_advice, race_day_summary, recent_weekly_load, template_plan,
)
from src.recovery import (
    RECOVERY_CSV, RECOVERY_METRICS, TEMPLATE_CSV, TRAINING_METRICS, correlation_table,
    join_recovery, load_recovery, rolling_baseline, synthetic_recovery,
)
from src.sports import (
    DEFAULT_INTENSITY, METHOD_DEFAULT, all_sessions, compute_other_load, sport_summary,
    weekly_sport_hours,
)
from src.sticker import INKS, png_size, render_sticker
from src.race_calculator import (
    CALCULATOR_DISTANCES, EXPONENT_RANGE, calculator_predictions, parse_duration, split_times,
)
from src.training_load import (
    ATL_DAYS, CTL_DAYS, DEFAULT_HR_MAX, DEFAULT_HR_REST, compute_activity_load,
    compute_fitness_freshness, daily_load_series, estimate_threshold_pace, form_label,
)
from src.units import IMPERIAL, UNIT_SYSTEMS, Units

FITTED_CONSTANTS = "Fitted to my data"
STANDARD_CONSTANTS = f"Standard ({CTL_DAYS} / {ATL_DAYS} days)"
RIEGEL_WINDOWS = {"Last 6 months": 182, "Last 12 months": 365, "All time": None}
MIN_RELATIVE_EFFORT_RUNS = 20
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
def build_analytics(path: str, _mtime: float, hr_rest: float, hr_max: float,
                    include_other: bool = True, ftp: float | None = None) -> dict:
    """Run the whole pipeline once per file and setting; ``_mtime`` busts the cache on edit.

    With ``include_other`` the non-run sessions' load joins the runs' in the
    daily series, so Fitness, Fatigue, ACWR and the Banister fit all see it.
    """
    activities = load_activities(path)
    with_load = compute_activity_load(activities, hr_rest=hr_rest, hr_max=hr_max)
    other = compute_other_load(load_other_activities(path), ftp=ftp, hr_rest=hr_rest,
                               hr_max=hr_max)
    counted = (pd.concat([with_load[["date", "load"]], other[["date", "load"]]])
               if include_other and not other.empty else with_load)
    daily = daily_load_series(counted)
    performance = performance_index(activities)
    return {
        "activities": activities,
        "scored": with_load,
        "other": other,
        "sessions": all_sessions(with_load, other),
        "include_other": include_other and not other.empty,
        "ftp": ftp,
        "daily": daily,
        "fitness": compute_fitness_freshness(daily),
        "ctl_days": CTL_DAYS,
        "atl_days": ATL_DAYS,
        "performance": performance,
        "banister": fit_banister(daily, performance),
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


@st.cache_data(show_spinner=False)
def other_sports_info(path: str, _mtime: float) -> tuple[int, bool]:
    """How many non-run sessions the export has, and whether any ride carries power."""
    other = load_other_activities(path)
    return len(other), bool(other[["avg_watts", "weighted_watts"]].notna().any().any())


@st.cache_data(show_spinner="Crunching a friend's activities...", ttl=CACHE_TTL,
               max_entries=4 * CACHE_ENTRIES)
def build_friend(path: str, _mtime: float, include_other: bool = True) -> dict:
    """A friend's runs and Fitness, scored like yours but with the default heart rates.

    Only what Head to head needs is computed -- no GPS files, no personal fits.
    """
    activities = load_activities(path)
    scored = compute_activity_load(activities)
    other = compute_other_load(load_other_activities(path))
    counted = (pd.concat([scored[["date", "load"]], other[["date", "load"]]])
               if include_other and not other.empty else scored)
    return {"activities": activities,
            "fitness": compute_fitness_freshness(daily_load_series(counted))}


@st.cache_data(show_spinner=False)
def read_recovery(content: bytes) -> pd.DataFrame:
    """Parse a sleep/HRV CSV once per distinct file contents."""
    return load_recovery(content)


def choose_source() -> Path:
    """Sidebar data-source picker; falls back to the synthetic sample."""
    st.sidebar.title("Strava Run Analytics")
    if not sample_is_current() or not (SAMPLE_CSV.parent / "activities").is_dir():
        with st.spinner("Generating sample data..."):
            generate_sample(SAMPLE_CSV)
    if not friends_are_current(SAMPLE_CSV):
        with st.spinner("Generating sample friends..."):
            generate_friends(SAMPLE_CSV)

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
    """Delete this session's uploaded export and friends, and clear the widgets holding them."""
    workspace = st.session_state.pop("workspace", None)
    if workspace is not None:
        upload.remove_workspace(Path(workspace))
    for key in ("uploaded_export", "upload_id", "h2h_friends", "h2h_shared_friend"):
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
    box.caption("Fitness, Form and ACWR still count the runs before the period. Personal "
                "models and shoe mileage always use your whole history.")
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
    recomputed on the period's runs.  ``daily``, ``performance`` and the fits
    keep the whole history (the personal models need every run they can get),
    the non-run ``other`` sessions and the combined ``sessions`` are cut too,
    and ``all_activities`` keeps every run for the shoe-mileage tab.
    ``previous`` holds the runs of the equally long window just before the
    period, for the Overview's comparison (``None`` when there is no such window).
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
        "scored": cut(data["scored"]),
        "other": cut(data["other"]),
        "sessions": cut(data["sessions"]),
        "fitness": cut(data["fitness"]),
        "acwr": cut(data["acwr"]),
        "weekly": weekly_volume(runs),
        "monthly": monthly_volume(runs),
        "frontier": best_effort_frontier(runs),
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


def choose_other_sports(path: Path) -> tuple[bool, float | None]:
    """Sidebar controls for non-run sessions: count them toward Fitness, and an FTP for rides.

    Hidden entirely for an export with nothing but runs.
    """
    count, has_power = other_sports_info(str(path), path.stat().st_mtime)
    if not count:
        return False, None
    st.sidebar.subheader("Other sports")
    include = st.sidebar.toggle(
        "Count them in Fitness & Form", value=True, key="include_other",
        help=f"{count:,} rides, swims, walks and other sessions in this export. On, their "
             "load is added to the runs' for Fitness, Fatigue, Form and ACWR.")
    ftp = None
    if has_power:
        ftp = st.sidebar.number_input(
            "Cycling FTP (W)", 0, 600, 0, step=5, key="ftp",
            help="Your functional threshold power. Set it to score rides with a power meter "
                 "by TSS; leave it at 0 to score them by heart rate alone.") or None
    return include, None if ftp is None else float(ftp)


def choose_time_constants() -> bool:
    """Sidebar switch between textbook and personally fitted Fitness/Fatigue constants."""
    st.sidebar.subheader("Fitness model")
    choice = st.sidebar.radio(
        "Time constants", [STANDARD_CONSTANTS, FITTED_CONSTANTS], key="time_constants",
        help="The fitted constants come from the Personal models tab.")
    return choice == FITTED_CONSTANTS


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


def apply_time_constants(data: dict, fitted: bool) -> dict:
    """Swap in Fitness/Fatigue computed with the fitted time constants when asked."""
    fit = data["banister"]
    if not fitted:
        return data
    if fit is None:
        st.sidebar.warning("No personal fit is available for this data, so the standard "
                           "constants are in use. See the Personal models tab.")
        return data
    return {**data, "fitness": compute_fitness_freshness(data["daily"], fit.tau1, fit.tau2),
            "ctl_days": fit.tau1, "atl_days": fit.tau2}


def overview_tab(data: dict) -> None:
    activities, fitness, acwr = data["activities"], data["fitness"], data["acwr"]
    units = data["units"]
    kpis = summary_kpis(activities)
    latest = fitness.iloc[-1]
    latest_acwr = acwr["acwr"].dropna()
    acwr_value = float(latest_acwr.iloc[-1]) if not latest_acwr.empty else float("nan")

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
    top[1].metric("Runs recorded", f"{kpis['total_runs']:,}", delta=deltas["runs"])
    top[2].metric("Time on feet", f"{kpis['total_hours']:,.0f} h", delta=deltas["hours"])
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
        f"Fitness is your {data['ctl_days']}-day average training load, Form is how rested you are "
        f"relative to it, and ACWR compares this week's load with the last month's. "
        f"Threshold pace is estimated at {units.format_pace(data['threshold_pace'])} from your "
        "own runs."
        + other_sports_note(data)
    )


def other_sports_note(data: dict) -> str:
    """A sentence on whether the period's non-run sessions count toward Fitness."""
    other = data["other"]
    if other.empty:
        return ""
    hours = other["moving_time_s"].sum() / 3600.0
    return (f" Fitness, Form and ACWR include {len(other):,} other sessions ({hours:,.0f} h) "
            "-- see All sports." if data["include_other"] else
            f" {len(other):,} other sessions ({hours:,.0f} h) are left out of Fitness, Form "
            "and ACWR (sidebar).")


def fitness_tab(data: dict) -> None:
    fitness, ctl_days, atl_days = data["fitness"], data["ctl_days"], data["atl_days"]
    st.plotly_chart(plots.fitness_freshness_figure(fitness, ctl_days, atl_days), width="stretch")
    st.caption(
        f"Fitness (blue) is the slow {ctl_days}-day average of your training load -- how much "
        f"work your body is used to. Fatigue (orange) is the fast {atl_days}-day average -- how "
        "much you did lately. Form (green) is the gap between them: positive means "
        "rested and race-ready, negative means you are carrying fatigue from a hard block."
        + ("" if (ctl_days, atl_days) == (CTL_DAYS, ATL_DAYS) else
           " These time constants were fitted to your own runs (see Personal models).")
    )
    latest = fitness.iloc[-1]
    st.info(f"**Form today: {latest['tsb']:+.0f} -- {form_label(float(latest['tsb']))}**")

    gap_share = data["activities"]["gap_pace_min_per_km"].notna().mean()
    with st.expander("How the training load behind this chart is computed"):
        st.markdown(
            f"""
            Each run scores `duration_hours x intensity_factor^2 x 100`, where
            `intensity_factor = threshold_pace / run_pace`. One hour at threshold pace
            therefore scores 100 points. Your threshold pace is estimated as the fastest
            decile of runs over 5 km: **{data['units'].format_pace(data['threshold_pace'])}**.
            Where the export has Strava's grade-adjusted pace ({gap_share:.0%} of runs here),
            that is the pace used, so a slow climb is scored as the hard work it is.

            When a run has heart rate, a Banister TRIMP score (rescaled to the same
            100-points-per-threshold-hour axis) is blended in 50/50, because HR catches
            effort that pace misses -- heat, hills and fatigue. Runs without HR use the
            pace score alone. Fitness and Fatigue are then exponentially weighted moving
            averages of daily load with {ctl_days}-day and {atl_days}-day time constants, and
            Form is yesterday's Fitness minus yesterday's Fatigue.
            """
        )
        if data["include_other"]:
            st.markdown(
                "Rides, swims, walks and gym sessions are on the same axis: heart-rate "
                "TRIMP where they have heart rate, cycling TSS from power when an FTP is set, "
                "and a stated default intensity per sport otherwise. Their load is added to "
                "the runs' each day -- the All sports tab shows how."
            )
    relative_effort_section(data)


def form_word(tsb: float) -> str:
    """The first words of :func:`form_label` -- "Fresh", "Very fatigued" -- for a metric card."""
    return form_label(tsb).split(" -- ")[0].split(" (")[0]


def race_plan_tab(data: dict) -> None:
    """Project Fitness, Fatigue and Form from the whole history to a race date."""
    daily, ctl_days, atl_days = data["daily"], data["ctl_days"], data["atl_days"]
    last = daily.index.max()
    plan_start = last + pd.Timedelta(days=1)
    recent = recent_weekly_load(daily)
    st.markdown(
        f"Pick a race and plan your weekly training load up to it; the chart runs the "
        f"Fitness & Freshness model forward to show the Form you would start with. The plan "
        f"begins the day after your last recorded session ({last:%d %b %Y}).")

    cols = st.columns(4)
    race = pd.Timestamp(cols[0].date_input(
        "Race date", value=(last + pd.Timedelta(weeks=12)).date(),
        min_value=(plan_start + pd.Timedelta(days=1)).date(),
        max_value=(last + pd.Timedelta(weeks=52)).date(), key="race_date"))
    weekly = cols[1].number_input(
        "Starting weekly load", 0, 3000, int(round(recent / 10) * 10), step=10, key="plan_load",
        help=f"Your last four weeks averaged {recent:.0f} points a week. One hour at threshold "
             "pace scores 100; an easy hour about 50-60.")
    increase = cols[2].slider("Weekly increase", 0, 15, 5, format="%d%%", key="plan_increase",
                              help="How much each build week adds to the one before.")
    taper = cols[3].selectbox(
        "Taper", list(TAPER_SHAPES), index=2, key="plan_taper",
        format_func=lambda n: "None" if n == 0 else f"{n} week{'s' if n > 1 else ''}",
        help="The final weeks keep "
             + "; ".join(f"{n}: {', '.join(f'{s:.0%}' for s in shape)}"
                         for n, shape in TAPER_SHAPES.items() if n)
             + " of the peak build week.")

    template = template_plan(plan_weeks(plan_start, race), weekly, increase / 100, taper)
    chart_box, summary_box = st.container(), st.container()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Week by week** -- edit any week's load to override the template.")
        edited = st.data_editor(
            template.assign(week_start=template["week_start"].dt.date),
            column_order=["weeks_out", "week_start", "days", "load"],
            column_config={
                "weeks_out": st.column_config.NumberColumn("Weeks to race", disabled=True),
                "week_start": st.column_config.DateColumn("From", disabled=True,
                                                          format="ddd D MMM"),
                "days": st.column_config.NumberColumn("Days", disabled=True),
                "load": st.column_config.NumberColumn("Planned load", min_value=0, step=10,
                                                      format="%d"),
            },
            hide_index=True, width="stretch",
            # A new template replaces any hand edits made to the old one.
            key=f"plan_editor_{race:%Y%m%d}_{weekly}_{increase}_{taper}",
        )
    plan = template.assign(load=pd.to_numeric(edited["load"], errors="coerce")
                           .fillna(0.0).to_numpy())
    projection = project_fitness(daily, plan, race, ctl_days, atl_days)
    summary = race_day_summary(projection)

    chart_box.plotly_chart(plots.race_plan_figure(projection, RACE_READY_FORM, ctl_days,
                                                  atl_days), width="stretch")
    chart_box.caption(
        "Solid lines are your recorded history, dotted lines the plan. The green band is "
        f"Form of {RACE_READY_FORM[0]:+.0f} to {RACE_READY_FORM[1]:+.0f}: rested enough to race "
        "well without having lost much Fitness to rest."
        + ("" if (ctl_days, atl_days) == (CTL_DAYS, ATL_DAYS) else
           f" The projection uses your fitted {ctl_days}- and {atl_days}-day time constants."))
    cards = summary_box.columns(4)
    cards[0].metric("Race-day Fitness", f"{summary['ctl']:.0f}",
                    delta=f"{summary['ctl_change']:+.1f} vs now")
    cards[1].metric("Race-day Form", f"{summary['tsb']:+.0f}",
                    delta=form_word(summary["tsb"]),
                    delta_color="off")
    cards[2].metric("Lowest Form on the way", f"{summary['min_tsb']:+.0f}",
                    delta=form_word(summary["min_tsb"]),
                    delta_color="off")
    cards[3].metric("Steepest week", f"{summary['max_ramp']:+.1f}",
                    delta="Fitness points in 7 days", delta_color="off")
    in_band = RACE_READY_FORM[0] <= summary["tsb"] <= RACE_READY_FORM[1]
    (summary_box.success if in_band else summary_box.warning)(race_day_advice(summary))
    if summary["max_ramp"] > MAX_SAFE_RAMP:
        summary_box.warning(
            f"Fitness climbs {summary['max_ramp']:.1f} points in one week of this plan. Rises "
            f"above about {MAX_SAFE_RAMP:.0f} a week are a common rule of thumb for a build that "
            "is hard to absorb -- consider a smaller weekly increase.")

    with right:
        st.markdown(
            f"""
            **How the projection works.** Each week's planned load is spread evenly over its
            days and appended to your recorded daily load, and the same {ctl_days}-day and
            {atl_days}-day averages behind the Fitness & Freshness chart are run forward, so
            the plan starts from the Fitness and Fatigue you actually have today.

            The template starts at your chosen weekly load, adds the weekly increase through
            the build, then tapers -- cutting volume by roughly 40-50% over the last
            fortnight, the range tapering research finds works best. Keep some faster running
            in the taper: it is volume, not intensity, that comes down.

            Race-day Form is the balance at the end of the day before the race. Where in a
            week the hard sessions fall moves Form by a few points on any given day, which an
            even spread does not show.
            """
        )


def relative_effort_section(data: dict) -> None:
    """Check this app's load score against Strava's own Relative Effort."""
    scored = data["scored"].dropna(subset=["relative_effort"])
    if len(scored) < MIN_RELATIVE_EFFORT_RUNS:
        return
    st.subheader("Checked against Strava's Relative Effort")
    pearson = float(scored["load"].corr(scored["relative_effort"]))
    # Spearman's rank correlation is Pearson's on the ranks (and needs no scipy).
    spearman = float(scored["load"].rank().corr(scored["relative_effort"].rank()))
    st.plotly_chart(plots.relative_effort_figure(scored, pearson), width="stretch")
    st.caption(
        f"Each dot is one of {len(scored):,} runs with both scores. Pearson r = {pearson:+.2f} "
        f"measures a straight-line match; Spearman's rank correlation = {spearman:+.2f} asks "
        "only whether the two scores put runs in the same order. The scales differ by design "
        "-- Strava's formula is proprietary and heart-rate only -- so agreement in *order* "
        "is the meaningful test: a high rank correlation says this transparent model ranks "
        "your sessions the way Strava does."
    )


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


def performance_tab(data: dict, stream_data, riegel) -> None:
    activities, records, units = data["activities"], data["records"], data["units"]
    has_streams = stream_data is not None and not stream_data.best.empty
    st.plotly_chart(plots.pace_distance_figure(activities, data["frontier"], units),
                    width="stretch")
    st.caption(
        "Every run plotted as distance against pace (note the log scale). The blue line "
        "is your best-effort frontier: the fastest pace you have ever held at each distance."
    )

    if has_streams:
        gps_best_efforts(activities, stream_data, records, units)

    left, right = st.columns(2)
    with left:
        st.subheader("Personal records" + (" (whole runs)" if has_streams else ""))
        table = records.rename(columns={"distance": "Distance", "time": "Time"})
        table["Pace"] = table["pace_min_per_km"].map(units.format_pace)
        table["Set on"] = table["date"].dt.strftime("%d %b %Y")
        table["In run"] = table["activity_distance_km"].map(units.format_distance)
        st.dataframe(table[["Distance", "Time", "Pace", "Set on", "In run"]],
                     hide_index=True, width="stretch")
        st.caption(
            "Fastest average pace ever held over a run of at least each distance, "
            "and the run it came from."
        )
    with right:
        st.subheader("Race predictions")
        choice = st.radio("Riegel exponent", ["Standard", "Fitted to you"], horizontal=True,
                          key="riegel_choice", disabled=riegel is None,
                          help=f"Standard is Riegel's {RIEGEL_EXPONENT}; yours is "
                               + (f"{riegel.exponent:.3f}" if riegel is not None
                                  else "not available") + " (Personal models tab).")
        exponent = riegel.exponent if riegel is not None and choice == "Fitted to you" \
            else RIEGEL_EXPONENT
        effort, source = reference_effort(activities, stream_data, units)
        st.dataframe(prediction_table(predict_races(effort, exponent=exponent), units),
                     hide_index=True, width="stretch")
        st.caption(
            f"Riegel's model (`T2 = T1 x (D2/D1)^{exponent:.3f}`) applied to {source}. "
            f"The standard exponent is reliable from 5K to the half and optimistic for the "
            f"marathon; the fitted one comes from your own best efforts (Personal models tab)."
        )

    race_calculator(effort, units, exponent)

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
        st.plotly_chart(plots.rolling_pace_figure(rolling_pace(activities), units=units),
                        width="stretch")
        st.caption("Average pace over a trailing 30 days, weighted by distance. Higher is "
                   "faster. Grade-adjusted pace (GAP) is what the pace would have been on the "
                   "flat; a GAP line well above the pace line means hilly running.")
    with trend_right:
        st.plotly_chart(plots.heart_rate_figure(rolling_hr(activities)), width="stretch")
        st.caption("Average heart rate per run. A falling line at a steady pace means fitness gains.")


def gps_best_efforts(activities: pd.DataFrame, stream_data, records: pd.DataFrame,
                     units: Units) -> None:
    """True in-run best efforts and the pace curve, from the GPS recordings."""
    st.subheader("Best efforts from your GPS recordings")
    best = streams.best_effort_table(stream_data.best, activities)
    whole_run = records.set_index("distance")["implied_time_s"] if not records.empty \
        else pd.Series(dtype="float64")
    table = pd.DataFrame({
        "Distance": best["distance"],
        "Time": best["time_s"].map(format_duration),
        "Pace": best["pace_min_per_km"].map(units.format_pace),
        "Set on": best["date"].dt.strftime("%d %b %Y"),
        "Inside": best["name"],
        "Whole-run estimate": best["distance"].map(
            lambda d: format_duration(whole_run[d]) if d in whole_run.index else "--"),
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "The fastest stretch of each distance found *anywhere inside* a run, by sliding a "
        "window along the GPS track, on moving time. The last column is what the whole-run "
        "method below can see; the gap is the 5K you ran inside a long run, or the fast "
        f"finish of a tempo session. From {stream_data.status['parsed']:,} recordings."
    )
    since = activities["date"].max() - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    st.plotly_chart(plots.pace_curve_figure(
        streams.pace_curve(stream_data.best), streams.pace_curve(stream_data.best, since),
        RECENT_WINDOW_DAYS, units), width="stretch")
    st.caption(
        "The running equivalent of a cyclist's power curve: your best pace at every "
        "distance from 400 m up. Where the orange line sits on the blue one, recent form "
        "matches your best; a gap shows how far off it you are at that distance."
    )


def prediction_table(predictions: pd.DataFrame, units: Units) -> pd.DataFrame:
    """Distance, predicted time and pace (in the chosen units) for display."""
    pace = predictions["predicted_time_s"] / 60.0 / predictions["distance_km"]
    return pd.DataFrame({"Distance": predictions["distance"],
                         "Predicted": predictions["predicted_time"],
                         "Pace": pace.map(units.format_pace)})


def race_calculator(effort, units: Units, exponent: float = RIEGEL_EXPONENT) -> None:
    """Type in any result and get Riegel predictions at every standard distance."""
    st.subheader("Race-time calculator")
    names = list(CALCULATOR_DISTANCES)
    default_time = format_duration(effort.time_s * (5.0 / effort.distance_km) ** exponent)
    start = round(min(max(exponent, EXPONENT_RANGE[0]), EXPONENT_RANGE[1]), 2)
    cols = st.columns([1, 1, 2])
    distance = cols[0].selectbox("Distance", names, index=names.index("5K"))
    entered = cols[1].text_input("Your time (H:MM:SS or MM:SS)", value=default_time)
    exponent = cols[2].slider("Riegel exponent", *EXPONENT_RANGE, value=start, step=0.01)
    try:
        time_s = parse_duration(entered)
    except ValueError as err:
        st.warning(str(err))
        return
    table = calculator_predictions(CALCULATOR_DISTANCES[distance], time_s, exponent)
    st.dataframe(prediction_table(table, units), hide_index=True, width="stretch")
    st.caption(
        "Enter a recent race or time trial to see what it implies at other distances. "
        "The exponent is how much pace fades as distance grows: 1.06 is Riegel's "
        "population fit, lower suits endurance-strong runners, higher suits speed-strong ones."
    )
    split_sheet(table, units)


def split_sheet(table: pd.DataFrame, units: Units) -> None:
    """Even-pace clock times at every kilometre or mile of one predicted race."""
    with st.expander(f"Even-pace split sheet (per {units.split_label.lower()})"):
        names = table["distance"].tolist()
        default = names.index("Half Marathon") if "Half Marathon" in names else len(names) - 1
        race = st.selectbox("Race", names, index=default, key="split_sheet_race")
        row = table.set_index("distance").loc[race]
        sheet = split_times(row["distance_km"], row["predicted_time_s"], units.split_m / 1000.0)
        st.dataframe(pd.DataFrame({
            units.split_label: sheet["at_km"].map(
                lambda km: f"{units.distance(km):.2f}".rstrip("0").rstrip(".")),
            "Split": sheet["split_time_s"].map(format_duration),
            "Clock": sheet["elapsed"],
        }), hide_index=True, width="stretch", height=min(36 * (len(sheet) + 1) + 2, 420))
        st.caption(f"Goal {row['predicted_time']} for the {race}, run at an even "
                   f"{units.format_pace(row['predicted_time_s'] / 60.0 / row['distance_km'])}.")


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

    breakdown = (stream_run_zone_breakdown(hr_seconds, hr_rest, hr_max) if from_streams
                 else run_zone_breakdown(activities, hr_rest, hr_max))
    run_zone_section(breakdown, activities, data["units"], from_streams)

    st.dataframe(
        zones.assign(share=zones["share"].map(lambda s: f"{s:.0%}"),
                     hours=zones["hours"].map(lambda h: f"{h:.1f}"))
        .rename(columns={"label": "Zone", "bpm": "Heart rate (bpm)", "runs": "Runs",
                         "hours": "Hours", "share": "Share"})
        [["Zone", "Heart rate (bpm)", "Runs", "Hours", "Share"]],
        hide_index=True, width="stretch",
    )

    if from_streams:
        same_runs = activities[activities["activity_id"].isin(hr_seconds["activity_id"])]
        average_method = zone_distribution(same_runs, hr_rest, hr_max)
        st.plotly_chart(plots.zone_method_figure(average_method, zones), width="stretch")
        st.caption(
            "The same runs counted two ways. Crediting a whole run to the zone of its "
            "average heart rate (orange) piles time into the middle zones; counting each "
            "second (blue) shows the easy minutes and hard intervals an average hides. "
            "\"Runs\" in the table above counts every run that touched a zone, so it adds "
            "up to more than the number of runs."
        )
    else:
        st.caption(
            "Approximation: without the GPS recordings only each run's average heart rate "
            "is known, so a whole run counts toward the zone of its average. Interval "
            "sessions are credited to the middle zones even though they swing between easy "
            "and hard. Add the export's `activities/` folder for second-by-second zones."
        )


def run_zone_section(breakdown: pd.DataFrame, activities: pd.DataFrame, units: Units,
                     from_streams: bool) -> None:
    """Each logged run's split across the zones: a chart of recent runs and a full table."""
    if breakdown.empty:
        return
    st.subheader("Run by run")
    runs = (activities[activities["activity_id"].isin(breakdown["activity_id"])]
            .sort_values("date", ascending=False).set_index("activity_id"))
    labels = pd.Series({a: f"{row.date:%d %b %Y} · {row['name']}" if row["name"]
                        else f"{row.date:%d %b %Y}" for a, row in runs.iterrows()})
    shown = (st.slider("Recent runs to chart", 5, min(60, len(runs)), min(20, len(runs)))
             if len(runs) > 5 else len(runs))
    recent = breakdown[breakdown["activity_id"].isin(runs.index[:shown])]
    st.plotly_chart(plots.run_zone_figure(recent, labels), width="stretch")
    st.caption(
        "How each run's time split across the zones, newest at the top. Easy runs should be "
        "almost all light blue; hard sessions show up as dark bars."
        if from_streams else
        "Without the GPS recordings each run is credited wholly to the zone of its average "
        "heart rate, so every bar is a single colour."
    )

    shares = (breakdown.pivot(index="activity_id", columns="zone", values="share")
              .reindex(runs.index))
    table = pd.DataFrame({
        "Date": runs["date"].dt.strftime("%Y-%m-%d"),
        "Run": runs["name"],
        "Distance": runs["distance_km"].map(units.format_distance),
        "Time": runs["moving_time_s"].map(format_duration),
        "Avg HR": runs["avg_hr"].map(lambda h: f"{h:.0f}" if pd.notna(h) else "--"),
    })
    for zone in shares.columns:
        table[f"Z{zone}"] = shares[zone].map(lambda s: f"{s:.0%}" if s > 0 else "")
    table["Easy (Z1-Z2)"] = shares[[1, 2]].sum(axis=1).map(lambda s: f"{s:.0%}")
    with st.expander(f"All {len(runs):,} runs with heart rate"):
        st.dataframe(table, hide_index=True, width="stretch")


def pace_zones_tab(data: dict, stream_data) -> None:
    activities, units = data["activities"], data["units"]
    threshold = data["threshold_pace"]
    gap_seconds = stream_data.gap_seconds if stream_data is not None else pd.DataFrame()
    from_streams = not gap_seconds.empty
    if from_streams:
        zones = pace_zones.stream_zone_distribution(gap_seconds, threshold, units)
        monthly = pace_zones.stream_monthly_zone_hours(gap_seconds, threshold)
        breakdown = pace_zones.stream_run_zone_breakdown(gap_seconds, threshold)
    else:
        zones = pace_zones.zone_distribution(activities, threshold, units)
        monthly = pace_zones.monthly_zone_hours(activities, threshold)
        breakdown = pace_zones.run_zone_breakdown(activities, threshold)
    if zones["hours"].sum() == 0:
        st.info("None of these runs has a pace to put in a zone.")
        return

    gap_share = float(activities["gap_pace_min_per_km"].notna().mean())
    easy_share = float(zones.loc[zones["zone"] <= 2, "share"].sum())
    cards = st.columns(3)
    cards[0].metric("Threshold pace", units.format_pace(threshold),
                    help="The fastest tenth of your runs of 5 km or more, on grade-adjusted "
                         "pace where the export has it -- the same threshold that scores "
                         "training load.")
    cards[1].metric("Most time in", zones.loc[zones["hours"].idxmax(), "label"])
    cards[2].metric("Easy share (Z1-Z2)", f"{easy_share:.0%}")
    st.caption(
        "Zones are fractions of your threshold speed, so they move with your fitness. "
        + (f"Counted second by second from {gap_seconds['activity_id'].nunique():,} GPS "
           "recordings, on pace corrected for the slope where the recording has altitude."
           if from_streams else
           f"Each run counts at its average pace, grade-adjusted for {gap_share:.0%} of runs "
           "(the rest have no Grade Adjusted Distance in the export).")
    )

    st.plotly_chart(plots.zone_distribution_figure(zones, "Time in pace zone", "pace", ""),
                    width="stretch", key="pace_zone_chart")
    st.caption(
        "How your running time splits across five pace zones. Compare it with the Heart rate "
        "tab: if your easy runs land in Z2 by pace but Z3 by heart rate, they are costing more "
        "than their pace suggests (heat, hills, fatigue), or your zones need new settings."
    )
    st.plotly_chart(plots.monthly_zone_figure(monthly), width="stretch",
                    key="pace_zone_monthly")
    st.caption("The same split month by month.")

    if not breakdown.empty:
        st.subheader("Run by run")
        runs = (activities[activities["activity_id"].isin(breakdown["activity_id"])]
                .sort_values("date", ascending=False).set_index("activity_id"))
        labels = pd.Series({a: f"{row.date:%d %b %Y} · {row['name']}" if row["name"]
                            else f"{row.date:%d %b %Y}" for a, row in runs.iterrows()})
        shown = (st.slider("Recent runs to chart", 5, min(60, len(runs)), min(20, len(runs)),
                           key="pace_zone_runs") if len(runs) > 5 else len(runs))
        recent = breakdown[breakdown["activity_id"].isin(runs.index[:shown])]
        st.plotly_chart(plots.run_zone_figure(recent, labels), width="stretch",
                        key="pace_zone_run_chart")
        st.caption("How each run's time split across the pace zones, newest at the top."
                   if from_streams else
                   "Without the GPS recordings each run is credited wholly to the zone of its "
                   "average pace, so every bar is a single colour.")

    st.dataframe(
        zones.assign(share=zones["share"].map(lambda v: f"{v:.0%}"),
                     hours=zones["hours"].map(lambda h: f"{h:.1f}"))
        .rename(columns={"label": "Zone", "pace": f"Pace ({units.pace_label})", "runs": "Runs",
                         "hours": "Hours", "share": "Share"})
        [["Zone", f"Pace ({units.pace_label})", "Runs", "Hours", "Share"]],
        hide_index=True, width="stretch",
    )

    if from_streams:
        same_runs = activities[activities["activity_id"].isin(gap_seconds["activity_id"])]
        average_method = pace_zones.zone_distribution(same_runs, threshold, units)
        st.plotly_chart(plots.zone_method_figure(average_method, zones,
                                                 "Whole run at its average pace"),
                        width="stretch", key="pace_zone_methods")
        st.caption(
            "The same runs counted two ways. An interval session averages out to a middle "
            "zone; counting each second shows its recovery jogs and its fast reps. \"Runs\" "
            "in the table counts every run that touched a zone, so it adds up to more than "
            "the number of runs."
        )


def consistency_tab(data: dict) -> None:
    weekly, acwr, units = data["weekly"], data["acwr"], data["units"]
    streaks = weekly_streaks(weekly)
    active = weekly[weekly["runs"] > 0]

    cards = st.columns(4)
    cards[0].metric("Current streak", f"{streaks.current} weeks")
    cards[1].metric("Longest streak", f"{streaks.longest} weeks")
    cards[2].metric("Typical week", units.format_distance(active["distance_km"].median(), 0)
                    if not active.empty else "--")
    cards[3].metric("Biggest week", units.format_distance(weekly["distance_km"].max(), 0)
                    if not weekly.empty else "--")
    st.caption("A streak is consecutive calendar weeks containing at least one run.")

    time_cards = st.columns(4)
    time_cards[0].metric("Typical week (time)", f"{active['moving_time_h'].median():.1f} h"
                         if not active.empty else "--")
    time_cards[1].metric("Typical week (runs)", f"{active['runs'].median():.0f} runs"
                         if not active.empty else "--")
    time_cards[2].metric("Average per week", f"{weekly['moving_time_h'].mean():.1f} h"
                         if not weekly.empty else "--")
    time_cards[3].metric("Average runs per week", f"{weekly['runs'].mean():.1f}"
                         if not weekly.empty else "--")
    st.caption("Typical is the median over weeks with at least one run; averages include "
               "empty weeks, so breaks pull them down.")

    st.plotly_chart(plots.weekly_volume_figure(weekly, units), width="stretch")
    st.caption("Distance run each calendar week. Empty weeks are shown as gaps, not skipped.")

    hours_col, runs_col = st.columns(2)
    with hours_col:
        st.plotly_chart(plots.weekly_hours_figure(weekly), width="stretch")
        st.caption("Moving time per calendar week.")
    with runs_col:
        st.plotly_chart(plots.weekly_runs_figure(weekly), width="stretch")
        st.caption("Number of runs per calendar week.")

    st.plotly_chart(plots.monthly_volume_figure(data["monthly"], units), width="stretch")
    st.caption("The same volume rolled up by month, which smooths out single-week noise.")

    st.plotly_chart(plots.acwr_figure(acwr), width="stretch")
    low, high = SWEET_SPOT
    st.caption(
        f"Acute:chronic workload ratio -- this week's average daily load divided by the "
        f"last four weeks'. Inside the shaded {low:g}-{high:g} band you are building "
        f"sustainably; well above it you are ramping faster than your body is adapting, "
        f"which is the classic injury-risk pattern."
    )


def sports_tab(data: dict) -> None:
    sessions, other, units = data["sessions"], data["other"], data["units"]
    if data["other"].empty:
        st.info("Every activity in this period is a run. Rides, swims, walks and gym "
                "sessions in your export show up here, and count toward Fitness and Form.")
        return
    summary = sport_summary(sessions)
    hours = sessions["moving_time_s"].sum() / 3600.0
    other_hours = other["moving_time_s"].sum() / 3600.0
    total_load = sessions["load"].sum()
    cards = st.columns(4)
    cards[0].metric("Sessions", f"{len(sessions):,}",
                    delta=f"{len(other):,} not runs", delta_color="off")
    cards[1].metric("Hours trained", f"{hours:,.0f} h")
    cards[2].metric("Cross-training share of time", f"{other_hours / hours:.0%}"
                    if hours else "--")
    cards[3].metric("Cross-training share of load", f"{other['load'].sum() / total_load:.0%}"
                    if total_load else "--")
    if not data["include_other"]:
        st.warning("These sessions are shown here but left out of Fitness, Form and ACWR. "
                   "Switch **Count them in Fitness & Form** on in the sidebar to include them.")

    value = st.radio("Stack by", ["Hours", "Training load"], horizontal=True,
                     key="sport_stack")
    weekly = weekly_sport_hours(sessions)
    st.plotly_chart(plots.sport_weekly_figure(weekly, "hours" if value == "Hours" else "load"),
                    width="stretch")
    st.caption("Every session per calendar week, stacked by sport with runs at the bottom. "
               "Weeks where the runs disappear but the bar does not are cross-training "
               "carrying your fitness -- through an injury, say.")

    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(plots.sport_load_figure(summary), width="stretch")
    with right:
        st.dataframe(pd.DataFrame({
            "Sport": summary["sport"],
            "Sessions": summary["sessions"],
            "Hours": summary["hours"].map(lambda h: f"{h:,.1f}"),
            "Distance": summary["distance_km"].map(
                lambda km: units.format_distance(km, 0) if km > 0 else "--"),
            "Load": summary["load"].map(lambda x: f"{x:,.0f}"),
            "Load per hour": (summary["load"] / summary["hours"]).map(lambda x: f"{x:.0f}"),
            "Share of load": summary["load_share"].map(lambda x: f"{x:.0%}"),
        }), hide_index=True, width="stretch")
        st.caption("Load per hour is how hard each sport is for you on average: 100 is an "
                   "hour at threshold.")

    defaulted = other[other["load_method"] == METHOD_DEFAULT]
    with st.expander("How sessions other than runs are scored"):
        st.markdown(
            "Each session is scored on the runs' scale, 100 points per hour at threshold, "
            "with the best evidence it carries. **Heart rate**: the same Banister TRIMP the "
            "runs use. **Power** (rides, with an FTP in the sidebar): "
            "`hours x (power / FTP)^2 x 100`, from Strava's weighted average power, blended "
            "50/50 with TRIMP when both exist. **Neither**: "
            "`hours x IF^2 x 100` with an assumed intensity factor for the sport -- "
            + ", ".join(f"{sport} {factor:.2f}" for sport, factor in DEFAULT_INTENSITY.items())
            + f". {len(defaulted):,} of {len(other):,} sessions here fall back to that "
            "assumption; recording heart rate replaces it with your own effort."
        )
        methods = (other.groupby(["sport", "load_method"]).size().unstack(fill_value=0))
        st.dataframe(methods, width="stretch")
        if data["ftp"] is None and other[["avg_watts", "weighted_watts"]].notna().any().any():
            st.caption("Some rides carry power. Set your FTP in the sidebar to score them by it.")

    with st.expander(f"All {len(other):,} sessions other than runs"):
        recent = other.sort_values("date", ascending=False)
        st.dataframe(pd.DataFrame({
            "Date": recent["date"].dt.strftime("%Y-%m-%d"),
            "Name": recent["name"],
            "Type": recent["activity_type"],
            "Time": recent["moving_time_s"].map(format_duration),
            "Distance": recent["distance_km"].map(
                lambda km: units.format_distance(km) if km > 0 else "--"),
            "Avg HR": recent["avg_hr"].map(lambda h: f"{h:.0f}" if pd.notna(h) else "--"),
            "Load": recent["load"].map(lambda x: f"{x:.0f}"),
            "Scored by": recent["load_method"],
        }), hide_index=True, width="stretch")


def recovery_tab(data: dict, is_sample: bool) -> None:
    uploaded = None
    has_saved = RECOVERY_CSV.is_file()
    with st.expander("Import sleep & HRV data", expanded=not has_saved):
        uploaded = st.file_uploader("Sleep score / HRV CSV", type="csv")
        st.markdown(
            "Any CSV with a **date** column and a **sleep score** and/or **HRV (ms)** "
            "column works -- names are matched loosely, so exports from Whoop, Oura or "
            "Garmin usually load as-is. Each date is the morning the reading describes. "
            "Running the app on your own computer? Save the file as `data/raw/recovery.csv` "
            "to skip uploading it every session."
        )
        st.download_button("Download a template", TEMPLATE_CSV, "recovery_template.csv",
                           mime="text/csv")

    if uploaded is not None:
        content, origin = uploaded.getvalue(), f"uploaded file `{uploaded.name}`"
    elif has_saved:
        content, origin = RECOVERY_CSV.read_bytes(), "`data/raw/recovery.csv`"
    else:
        content, origin = None, None

    if content is not None:
        try:
            recovery = read_recovery(content)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as err:
            st.error(f"Could not read {origin}: {err}")
            return
    elif is_sample:
        recovery = synthetic_recovery(data["fitness"])
        origin = "synthetic sleep and HRV data generated to match the sample runs"
    else:
        st.info("Upload a sleep score / HRV file above to see how it lines up with "
                "your training.")
        return

    joined = join_recovery(recovery, data["fitness"])
    if joined.empty:
        st.warning(
            f"None of the {len(recovery):,} days in the recovery file fall inside your "
            f"training history ({data['activities']['date'].min():%d %b %Y} -- "
            f"{data['activities']['date'].max():%d %b %Y})."
        )
        return
    available = [key for key in RECOVERY_METRICS if joined[key].notna().any()]
    table = correlation_table(joined)
    st.caption(f"Using {origin}: {len(joined):,} days overlap with your runs.")

    cards = st.columns(4)
    cards[0].metric("Days matched", f"{len(joined):,}")
    cards[1].metric("Average sleep score", f"{joined['sleep_score'].mean():.0f}"
                    if "sleep_score" in available else "--")
    cards[2].metric("Average HRV", f"{joined['hrv_ms'].mean():.0f} ms"
                    if "hrv_ms" in available else "--")
    ranked = table.dropna(subset=["r"])
    if not ranked.empty:
        top = ranked.loc[ranked["r"].abs().idxmax()]
        cards[3].metric("Strongest link", f"r = {top['r']:+.2f}",
                        delta=f"{top['recovery_label']} vs {top['training_label'].lower()}",
                        delta_color="off")
    else:
        cards[3].metric("Strongest link", "--")

    st.subheader("How training lines up with recovery")
    st.dataframe(
        table.rename(columns={"recovery_label": "Recovery", "training_label": "Training",
                              "n": "Days", "strength": "Reading"})
        [["Recovery", "Training", "r", "Days", "Reading"]],
        hide_index=True, width="stretch",
        column_config={"r": st.column_config.NumberColumn("r", format="%+.2f")},
    )
    st.caption(
        "Pearson correlation: -1 means one always falls as the other rises, 0 means no "
        "straight-line link, +1 means they rise together. A negative r between yesterday's "
        "load and HRV is the classic pattern -- hard days suppress the next morning's HRV. "
        "Correlation is not cause, and with fewer than a few months of data, r swings a lot."
    )

    if available:
        left, right = st.columns(2)
        recovery_key = left.selectbox("Recovery metric", available,
                                      format_func=RECOVERY_METRICS.get)
        training_key = right.selectbox("Training metric", list(TRAINING_METRICS),
                                       format_func=TRAINING_METRICS.get)
        r = table.set_index(["recovery", "training"]).loc[(recovery_key, training_key), "r"]
        st.plotly_chart(plots.recovery_scatter_figure(
            joined, training_key, recovery_key, TRAINING_METRICS[training_key],
            RECOVERY_METRICS[recovery_key], float(r)), width="stretch")
        st.caption("Each dot is one morning's reading against the training metric for "
                   "that day; the line is the least-squares trend.")

        columns = st.columns(len(available))
        titles = {"sleep_score": ("Sleep score", "Score"), "hrv_ms": ("Overnight HRV", "HRV (ms)")}
        for column, key in zip(columns, available):
            with column:
                title, y_title = titles[key]
                st.plotly_chart(plots.recovery_timeline_figure(
                    joined, key, rolling_baseline(joined, key), title, y_title),
                    width="stretch")
        st.caption("Nightly readings with a trailing 7-day average. Trends in the average "
                   "matter more than any single night.")


NO_STREAMS_HELP = (
    "This needs the GPS recordings. In your Strava archive they are in the "
    "`activities/` folder next to `activities.csv`; copy that folder into `data/raw/` "
    "too and reload."
)


def run_detail_tab(data: dict, stream_data, path: Path) -> None:
    activities, units = data["activities"], data["units"]
    if stream_data is None or stream_data.splits.empty:
        st.info(NO_STREAMS_HELP if stream_data is None or stream_data.status["parsed"] == 0
                else "None of the runs in this period has a readable GPS recording.")
        return
    all_splits = stream_data.mile_splits if units is IMPERIAL else stream_data.splits
    with_splits = activities[activities["activity_id"].isin(stream_data.splits["activity_id"])]
    runs = with_splits.sort_values("date", ascending=False)
    labels = {int(row.activity_id): f"{row.date:%a %d %b %Y} -- {row.name} -- "
                                    f"{units.format_distance(row.distance_km)}"
              for row in runs.itertuples()}
    activity_id = st.selectbox("Run", list(labels), format_func=labels.get)
    run = runs.set_index("activity_id").loc[activity_id]
    splits = all_splits[all_splits["activity_id"] == activity_id]
    detail = stream_data.runs.set_index("activity_id").loc[activity_id]

    gap, gap_source = run_gap(run, detail)
    cards = st.columns(7)
    cards[0].metric("Distance", units.format_distance(run["distance_km"], 2))
    cards[1].metric("Moving time", format_duration(run["moving_time_s"]))
    cards[2].metric("Pace", units.format_pace(run["pace_min_per_km"]))
    cards[3].metric("Grade-adjusted pace", units.format_pace(gap) if pd.notna(gap) else "--",
                    help=gap_source)
    cards[4].metric("Average HR", f"{run['avg_hr']:.0f} bpm" if pd.notna(run["avg_hr"]) else "--")
    cards[5].metric("Cadence", f"{run['avg_cadence_spm']:.0f} spm"
                    if pd.notna(run["avg_cadence_spm"]) else "--")
    decoupling = detail["decoupling_pct"]
    cards[6].metric("Aerobic decoupling", f"{decoupling:+.1f}%" if pd.notna(decoupling) else "--")

    route = streams.route_for(activity_id, data["all_activities"], path)
    route_map(route, units)
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
        "from Strava's own. "
        "Aerobic decoupling compares speed per heartbeat in the first and second half of the "
        "run. Under about 5% on a steady run means your aerobic base held up; a large value "
        "means your heart worked harder for the same speed as the run went on. Interval "
        "sessions and fast finishes decouple by design, so read it on steady runs."
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


TERRAIN_NAMES = {"climb": "Climbs", "flat": "Flat", "descent": "Descents"}


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
    st.dataframe(pd.DataFrame({
        "Terrain": [TERRAIN_NAMES[k] for k in summary.index],
        "Stretches": summary["count"].to_numpy(),
        "Distance": (summary["distance_m"] / 1000.0).map(units.format_distance).to_numpy(),
        "Share of time": summary["time_share"].map(lambda v: f"{v:.0%}").to_numpy(),
        "Elevation": summary["change_m"].map(units.format_elevation).to_numpy(),
        "Pace": summary["pace_min_per_km"].map(units.format_pace).to_numpy(),
        "GAP": summary["gap_pace_min_per_km"].map(units.format_pace).to_numpy(),
        "Avg HR": summary["avg_hr"].map(lambda h: f"{h:.0f}" if pd.notna(h) else "--").to_numpy(),
    }), hide_index=True, width="stretch")

    with st.expander(f"Each climb and descent ({len(hills)})"):
        st.dataframe(pd.DataFrame({
            "": hills["kind"].map({"climb": "Climb", "descent": "Descent"}),
            "From": (hills["start_m"] / 1000.0).map(lambda km: units.format_distance(km, 2)),
            "Length": (hills["distance_m"] / 1000.0).map(lambda km: units.format_distance(km, 2)),
            "Elevation": hills["change_m"].map(units.format_elevation),
            "Grade": hills["grade"].map(lambda g: f"{g:+.1%}"),
            "Pace": hills["pace_min_per_km"].map(units.format_pace),
            "GAP": hills["gap_pace_min_per_km"].map(units.format_pace),
            "Avg HR": hills["avg_hr"].map(lambda h: f"{h:.0f}" if pd.notna(h) else "--"),
            "HR change": (hills["hr_end"] - hills["hr_start"]).map(
                lambda h: f"{h:+.0f} bpm" if pd.notna(h) else "--"),
        }), hide_index=True, width="stretch")
        st.caption(f"HR change is heart rate {terrain.HR_LAG_S:.0f} s after the end of the "
                   f"stretch minus {terrain.HR_LAG_S:.0f} s after its start, to allow for the lag.")


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
    st.plotly_chart(plots.route_map_figure(route, options[color_by], units, basemap),
                    width="stretch")
    st.caption(
        "Darker means faster (or a higher heart rate), with the colour scale set by the run's "
        "own 5th-95th percentile so one GPS spike cannot flatten the rest. Pace is smoothed "
        "over a few points either side, so short surges are softened."
    )


def current_riegel(activities: pd.DataFrame, stream_data):
    """The personal Riegel fit for the window and distances chosen on Personal models."""
    days = RIEGEL_WINDOWS[st.session_state.get("riegel_window", "Last 12 months")]
    since = activities["date"].max() - pd.Timedelta(days=days) if days else None
    low, high = st.session_state.get("riegel_range", ("1K", "Marathon"))
    efforts = riegel_efforts(
        activities, stream_data.best if stream_data is not None else None, since,
        streams.NAMED_DISTANCES_M[low] / 1000.0, streams.NAMED_DISTANCES_M[high] / 1000.0)
    return fit_riegel(efforts)


def personal_models_tab(data: dict, riegel) -> None:
    st.markdown(
        "Two numbers on this dashboard are textbook defaults: Riegel's exponent of "
        f"{RIEGEL_EXPONENT} and the {CTL_DAYS}- and {ATL_DAYS}-day time constants behind "
        "Fitness and Fatigue. Here each is estimated from your own runs, with an honest "
        "measure of how sure the estimate is. Both use your whole history, whatever period "
        "the sidebar shows."
    )
    st.subheader("Your Riegel exponent")
    cols = st.columns(2)
    cols[0].radio("Efforts from", list(RIEGEL_WINDOWS), index=1, horizontal=True,
                  key="riegel_window")
    names = [name for name in streams.NAMED_DISTANCES_M if name != "400 m"]
    cols[1].select_slider("Distances to fit", names, value=("1K", "Marathon"),
                          key="riegel_range")
    if riegel is None:
        st.info("At least three distances are needed to fit an exponent. Widen the window "
                "or the distance range.")
    else:
        cards = st.columns(3)
        cards[0].metric("Your exponent", f"{riegel.exponent:.3f}",
                        delta=f"{riegel.exponent - RIEGEL_EXPONENT:+.3f} vs {RIEGEL_EXPONENT}",
                        delta_color="off")
        cards[1].metric("95% interval", f"{riegel.ci_low:.3f} - {riegel.ci_high:.3f}")
        cards[2].metric("Distances fitted", f"{riegel.n}")
        st.plotly_chart(plots.riegel_fit_figure(riegel, RIEGEL_EXPONENT, data["units"]),
                        width="stretch")
        verdict = (
            f"The interval excludes {RIEGEL_EXPONENT}, so your data say your pace fades "
            + ("more" if riegel.exponent > RIEGEL_EXPONENT else "less")
            + " with distance than the textbook runner's."
            if riegel.differs_from_default else
            f"The interval includes {RIEGEL_EXPONENT}, so your data cannot tell you apart "
            "from the textbook runner -- the standard exponent is a fine choice."
        )
        st.caption(
            "Riegel's `T = a x D^b` is a straight line on log-log axes, so b is the slope of a "
            "least-squares fit of log time on log distance, with a t-based 95% interval. "
            f"{verdict} Two cautions: best efforts at several distances often come from one "
            "run, so the points are not independent and the interval is optimistic; and a "
            "best half marathon from an easy long run is not a race, which tilts the line "
            "steeper. Fit only the distances you have actually raced for the cleanest "
            "number. Choose \"Fitted to you\" on the Performance tab to predict with it."
        )

    st.subheader("Your Fitness and Fatigue time constants")
    fit = data["banister"]
    if fit is None:
        st.info(
            f"This needs at least {MIN_PERFORMANCE_RUNS} runs with heart rate, and a fit in "
            "which fitness helps and fatigue hurts. Neither held for this data, so the "
            "dashboard keeps the standard constants."
        )
        return
    cards = st.columns(4)
    cards[0].metric("Fitness time constant", f"{fit.tau1} days",
                    delta=f"90% range {fit.tau1_interval[0]:.0f}-{fit.tau1_interval[1]:.0f}",
                    delta_color="off")
    cards[1].metric("Fatigue time constant", f"{fit.tau2} days",
                    delta=f"90% range {fit.tau2_interval[0]:.0f}-{fit.tau2_interval[1]:.0f}",
                    delta_color="off")
    cards[2].metric("Variance explained", f"{fit.r2:.0%}",
                    delta=f"{fit.r2_default:.0%} at {CTL_DAYS} / {ATL_DAYS} days",
                    delta_color="off")
    cards[3].metric("A hard session pays off",
                    f"day {fit.peak_day}" if fit.peak_day is not None else "--",
                    delta=(f"net positive from day {fit.positive_day}"
                           if fit.positive_day is not None else None), delta_color="off")

    modelled = modelled_performance(data["daily"], fit)
    st.plotly_chart(plots.performance_model_figure(data["performance"], modelled),
                    width="stretch")
    st.caption(
        "Each dot is one run's speed relative to what its heart rate and distance predict "
        "(from a regression across all your runs): above zero, you ran faster than usual for "
        "that effort. The line is Banister's model -- a slow fitness average of your training "
        "load minus a fast fatigue average, plus whatever fitness you already had on the "
        "export's first day -- with both time constants and gains fitted to the dots."
    )
    edge = []
    if fit.tau1 in (TAU1_GRID[0], TAU1_GRID[-1]):
        edge.append("fitness")
    if fit.tau2 in (TAU2_GRID[0], TAU2_GRID[-1]):
        edge.append("fatigue")
    if edge:
        st.warning(
            f"The best {' and '.join(edge)} time constant sits at the edge of the range "
            "searched, so the true value may lie beyond it -- read it as \"at least\" or "
            "\"at most\", not as an estimate."
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plots.time_constant_surface_figure(fit, (CTL_DAYS, ATL_DAYS)),
                        width="stretch")
        st.caption(
            "R² for every pair of time constants searched (blank where the fit would have "
            "fitness hurting or fatigue helping). A long flat ridge is the usual picture: "
            "many pairs fit almost equally well, which is why the 90% ranges above, from "
            "refitting on 200 bootstrap resamples of your runs, matter more than the single "
            "best pair."
        )
    with right:
        st.plotly_chart(plots.impulse_figure(fit), width="stretch")
        st.caption(
            "What the fitted model says one training session does: fatigue costs "
            "performance at first, fitness wins out as fatigue clears, and the net effect "
            "peaks before slowly fading. The peak day is a rough guide to how long before a "
            "race the last hard sessions still pay off."
        )
    better = fit.r2 - fit.r2_default
    st.info(
        f"Your fitted constants explain {fit.r2:.0%} of the run-to-run variation against "
        f"{fit.r2_default:.0%} for the standard {CTL_DAYS} / {ATL_DAYS} days"
        + (f" -- a gain of {better:.0%}. " if better >= 0.01 else
           " -- no real gain, so the standard constants are as good a description of you. ")
        + "Switch the Fitness & Freshness chart to your constants under "
          "**Fitness model** in the sidebar."
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


STICKER_PREVIEW_PX = 320
# Preview backdrops that keep each ink readable (the PNG itself stays transparent).
STICKER_BACKDROPS = {"White": "linear-gradient(160deg, #3a4a5c, #1d2731)",
                     "Black": "linear-gradient(160deg, #f4efe6, #d9d2c5)",
                     "Orange": "linear-gradient(160deg, #3a4a5c, #1d2731)"}


def period_sticker_stats(activities: pd.DataFrame, fitness: pd.DataFrame,
                         units: Units) -> dict[str, str]:
    """Every stat a period sticker can show, formatted for display."""
    kpis = summary_kpis(activities)
    moving_s = activities["moving_time_s"].sum()
    elevation = activities["elevation_gain_m"].sum()
    return {
        "Distance": units.format_distance(kpis["total_distance_km"], 0),
        "Runs": f"{kpis['total_runs']:,}",
        "Time": f"{kpis['total_hours']:,.0f} h",
        "Avg pace": units.format_pace(moving_s / 60.0 / kpis["total_distance_km"]),
        "Elevation gain": f"{elevation * units.per_metre:,.0f} {units.elevation_label}",
        "Longest run": units.format_distance(activities["distance_km"].max()),
        "Fitness": f"{fitness['ctl'].iloc[-1]:.0f}",
    }


def run_sticker_stats(run: pd.Series, units: Units) -> dict[str, str]:
    """Every stat a run sticker can show, formatted for display; missing ones left out."""
    stats = {
        "Distance": units.format_distance(run["distance_km"], 2),
        "Pace": units.format_pace(run["pace_min_per_km"]),
        "Time": format_duration(run["moving_time_s"]),
    }
    if pd.notna(run["elevation_gain_m"]):
        stats["Elev gain"] = f"{run['elevation_gain_m'] * units.per_metre:,.0f} " \
                             f"{units.elevation_label}"
    if pd.notna(run["avg_hr"]):
        stats["Avg HR"] = f"{run['avg_hr']:.0f} bpm"
    return stats


def sticker_preview(png: bytes, ink: str) -> None:
    """The sticker on a backdrop, with a button that copies the PNG to the clipboard."""
    width, height = png_size(png)
    shown = min(STICKER_PREVIEW_PX, width) * height / width
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    components.html(f"""
<div style="font-family: sans-serif;">
  <div style="background: {STICKER_BACKDROPS[ink]}; border-radius: 12px; padding: 20px;
              display: flex; justify-content: center;">
    <img src="{data_url}" style="width: {STICKER_PREVIEW_PX}px; max-width: 100%;">
  </div>
  <button id="copy" style="margin-top: 12px; padding: 8px 16px; border-radius: 8px;
          border: 1px solid #2a78d6; background: #2a78d6; color: white; font-size: 14px;
          cursor: pointer;">Copy sticker</button>
  <span id="status" style="margin-left: 10px; font-size: 13px; color: #555;"></span>
</div>
<script>
  const status = document.getElementById("status");
  document.getElementById("copy").addEventListener("click", async () => {{
    try {{
      // A promise, not a blob, keeps Safari's user-gesture requirement satisfied.
      const blob = fetch("{data_url}").then(r => r.blob());
      await navigator.clipboard.write([new ClipboardItem({{"image/png": blob}})]);
      status.textContent = "Copied -- paste it into your story.";
    }} catch (err) {{
      status.textContent = "This browser blocked the copy; use Download instead.";
    }}
  }});
</script>
""", height=int(shown) + 100)


def share_tab(data: dict, stream_data, path: Path) -> None:
    activities, units = data["activities"], data["units"]
    st.markdown("A transparent sticker of your stats, like Strava's, to paste onto an "
                "Instagram story.")
    kind = st.radio("Sticker for", ["This period", "One run"], horizontal=True,
                    key="sticker_kind")
    route = None
    if kind == "One run":
        runs = activities.sort_values("date", ascending=False)
        labels = {int(row.activity_id): f"{row.date:%a %d %b %Y} -- {row.name} -- "
                                        f"{units.format_distance(row.distance_km)}"
                  for row in runs.itertuples()}
        activity_id = st.selectbox("Run", list(labels), format_func=labels.get,
                                   key="sticker_run")
        run = runs.set_index("activity_id").loc[activity_id]
        available = run_sticker_stats(run, units)
        default = ["Distance", "Pace", "Time"]
        caption = f"{run['name']} · {run['date']:%d %b %Y}"
        if stream_data is not None:
            route = streams.route_for(activity_id, data["all_activities"], path)
    else:
        available = period_sticker_stats(activities, data["fitness"], units)
        default = ["Distance", "Runs", "Time"]
        first, last = activities["date"].min(), activities["date"].max()
        caption = (f"{first:%b %Y}" if f"{first:%b %Y}" == f"{last:%b %Y}"
                   else f"{first:%b %Y} – {last:%b %Y}")

    left, right = st.columns([1, 1])
    with left:
        chosen = st.multiselect("Stats", list(available), default=default, max_selections=4,
                                key=f"sticker_stats_{kind}")
        ink = st.radio("Colour", list(INKS), horizontal=True, key="sticker_ink")
        caption = st.text_input("Caption", caption, key=f"sticker_caption_{kind}")
        show_route = route is not None and not route.empty and st.toggle(
            "Show route", value=True, key="sticker_route")
        if not chosen:
            st.info("Pick at least one stat.")
            return
        png = render_sticker([(name, available[name]) for name in chosen], caption,
                             route if show_route else None, ink)
        st.download_button("Download PNG", png, file_name="run-sticker.png", mime="image/png")
        st.caption(
            "**To post it:** copy the sticker, then on your iPhone open an Instagram story "
            "and tap Paste (or the sticker Instagram offers from your clipboard). Copying on a "
            "Mac reaches the phone through Universal Clipboard; otherwise download the PNG and "
            "AirDrop it, then add it from Photos. The background is transparent, so your "
            "photo shows through -- the backdrop here is only for the preview."
        )
    with right:
        sticker_preview(png, ink)


MAX_FRIENDS = 5


def format_metric(key: str, value: float, units: Units) -> str:
    """One :data:`src.head_to_head.METRICS` value, formatted for the table."""
    if pd.isna(value):
        return "--"
    if key in ("distance_km", "weekly_km", "longest_km"):
        return units.format_distance(value, 0 if key == "distance_km" else 1)
    if key == "elevation_m":
        return f"{value * units.per_metre:,.0f} {units.elevation_label}"
    if key == "hours":
        return f"{value:,.1f} h"
    if key == "active_share":
        return f"{value:.0%}"
    if key == "longest_streak":
        return f"{value:.0f} weeks"
    if key == "avg_pace":
        return units.format_pace(value)
    return f"{value:,.0f}"


def friends_folder(path: Path) -> tuple[Path, bool]:
    """Where the friends for the export at ``path`` live, and whether that folder is kept.

    An export in ``data/raw/`` keeps its friends in ``data/raw/friends/`` on this
    computer; an uploaded export's friends sit in the same private session folder
    as the upload and go when it does.
    """
    if path == SAMPLE_CSV:
        return SAMPLE_FRIENDS_DIR, True
    if path.parent == RAW_DIR:
        return FRIENDS_DIR, True
    return path.parents[1] / "friends", False


def add_friend_form(folder: Path, kept: bool) -> None:
    """Upload a friend's export (zip or ``activities.csv``), check it reads, and save it to ``folder``."""
    with st.expander("Add a friend"):
        where = ("It is saved as `data/raw/friends/<name>.csv` and stays on this computer; "
                 "you can also drop files there yourself." if kept else
                 "It is kept only for this browser session, alongside your own upload, and "
                 "deleted with it.")
        st.markdown(
            "Ask a friend to request their archive from Strava (Settings -> My Account -> "
            f"Download or Delete Your Account) and send you the zip or its `activities.csv`. {where}")
        name = st.text_input("Friend's name", key="friend_name")
        uploaded = st.file_uploader("Their export (.zip or activities.csv)", type=["zip", "csv"],
                                    key="friend_upload")
        if not st.button("Add friend", disabled=not (name.strip() and uploaded)):
            return
        try:
            content = upload.read_export_csv(uploaded, uploaded.name)
            path = save_friend_export(content, name.strip(), folder)
        except ValueError as err:
            st.error(str(err))
            return
        try:
            runs = load_activities(path)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as err:
            path.unlink()
            st.error(f"That file could not be read as a Strava export: {err}")
            return
        if runs.empty:
            path.unlink()
            st.error("That export has no runs in it.")
            return
        st.rerun()


def remove_friend_form(exports: dict[str, Path], folder: Path) -> None:
    """Delete a friend's export from ``folder``."""
    with st.expander("Remove a friend"):
        name = st.selectbox("Friend", list(exports), key="friend_remove")
        if not st.button(f"Remove {name}", key="friend_remove_button"):
            return
        try:
            remove_friend_export(exports[name], folder)
        except (ValueError, OSError) as err:
            st.error(f"Could not remove {name}: {err}")
            return
        # Widgets still holding the name would no longer match their options.
        chosen = st.session_state.get("h2h_friends")
        if chosen and name in chosen:
            st.session_state["h2h_friends"] = [n for n in chosen if n != name]
        if st.session_state.get("h2h_shared_friend") == name:
            del st.session_state["h2h_shared_friend"]
        st.rerun()


def race_measure_format(measure: str, units: Units) -> tuple[str, str, str, float]:
    """Axis title, number format, hover suffix and display scale for a :data:`RACE_MEASURES` key."""
    if measure == "distance_km":
        return (f"Distance ({units.distance_label})", ",.1f", f" {units.distance_label}",
                units.per_km)
    if measure == "elevation_m":
        return (f"Elevation gain ({units.elevation_label})", ",.0f", f" {units.elevation_label}",
                units.per_metre)
    if measure == "hours":
        return "Time on feet (h)", ",.1f", " h", 1.0
    return "Runs", ",.0f", "", 1.0


def head_to_head_tab(data: dict, path: Path, start: pd.Timestamp, end: pd.Timestamp) -> None:
    units = data["units"]
    is_sample = path == SAMPLE_CSV
    st.markdown(
        "Put your running next to your friends': everyone measured over the period picked "
        "in the sidebar, from each person's own Strava export.")
    if is_sample:
        st.caption("These friends are synthetic, generated alongside the sample data. With "
                   "your own export selected you can add real friends here.")
    folder, kept = friends_folder(path)
    if not is_sample:
        add_friend_form(folder, kept)

    exports = find_friend_exports(folder)
    if not exports:
        st.info("No friends added yet. Add a friend's `activities.csv` above to compare.")
        return
    if not is_sample:
        remove_friend_form(exports, folder)
    cols = st.columns([1, 3])
    me = cols[0].text_input("Your name", "You", key="h2h_me").strip() or "You"
    chosen = cols[1].multiselect("Compare with", list(exports),
                                 default=list(exports)[:MAX_FRIENDS],
                                 max_selections=MAX_FRIENDS, key="h2h_friends")
    if not chosen:
        st.info("Pick at least one friend to compare with.")
        return
    if me in chosen:
        st.warning(f"A friend is also called {me}; pick a different name for yourself.")
        return

    # Yours with the standard constants, as your friends' are, so Fitness compares fairly.
    athletes = {me: {"activities": data["all_activities"],
                     "fitness": compute_fitness_freshness(data["daily"])}}
    for name in chosen:
        friend_path = exports[name]
        athletes[name] = build_friend(str(friend_path), friend_path.stat().st_mtime,
                                      data["include_other"])
    names = list(athletes)
    runs = {name: cut_period(a["activities"], start, end) for name, a in athletes.items()}
    idle = [name for name in chosen if runs[name].empty]
    if idle:
        st.warning(f"{', '.join(idle)} recorded no runs in this period, so they score zero "
                   "on everything below. Try a longer period in the sidebar.")

    labels = {key: label for key, label, _ in METRICS}
    counted = st.multiselect("Scoreboard categories", list(labels), default=list(labels),
                             format_func=labels.get, key="h2h_categories")
    if not counted:
        st.info("Pick at least one scoreboard category.")
        return
    profiles = {name: athlete_profile(a["activities"], a["fitness"], start, end)
                for name, a in athletes.items()}
    table = head_to_head_table(profiles)
    table = table[table["metric"].isin(counted)].reset_index(drop=True)
    wins = scoreboard(table, names)
    st.subheader("Scoreboard")
    cards = st.columns(len(names))
    for card, name in zip(cards, names):
        card.metric(name, f"{wins[name]} of {len(table)}", delta="categories led",
                    delta_color="off")
    display = pd.DataFrame({"Metric": table["label"]})
    for name in names:
        display[name] = [format_metric(key, value, units)
                         for key, value in zip(table["metric"], table[name])]
    display["Leader"] = table["leader"].fillna("--")
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        f"Over {start:%d %b %Y} -- {end - pd.Timedelta(days=1):%d %b %Y} for everyone. "
        "\"Weeks with a run\" and the streak count calendar weeks in the period. A tie leads "
        "for no one. Fitness is scored against each runner's *own* threshold pace, so it "
        "compares how hard each of you is training for yourselves, not who is faster -- the "
        "tables below answer that.")

    measure = st.radio("Race on", list(RACE_MEASURES), format_func=RACE_MEASURES.get,
                       horizontal=True, key="h2h_measure")
    label = RACE_MEASURES[measure]
    y_title, fmt, suffix, scale = race_measure_format(measure, units)
    st.plotly_chart(plots.athlete_lines_figure(
        cumulative_totals(runs, start, end, measure), names, measure,
        f"{label} race through the period", y_title, fmt, scale=scale), width="stretch")
    st.caption(f"Running total of {label.lower()} from the start of the period. A steeper "
               "line is a bigger block of training.")

    battles = weekly_battles(runs, start, end, measure)
    week_wins = battles["leader"].value_counts()
    st.plotly_chart(plots.weekly_battle_figure(
        battles, names, f"Weekly {label.lower()}, head to head", y_title, fmt, suffix, scale),
        width="stretch")
    st.caption(f"Weeks won (most {label.lower()}): " + ", ".join(
        f"**{name}** {int(week_wins.get(name, 0))}" for name in names)
        + (f", tied {int(week_wins[TIE])}" if TIE in week_wins else "")
        + f" of {len(battles)} weeks.")

    st.plotly_chart(plots.athlete_lines_figure(
        fitness_comparison({name: a["fitness"] for name, a in athletes.items()}, start, end),
        names, "ctl", "Fitness (CTL), head to head", "Load points", ".0f"), width="stretch")
    st.caption(
        f"Everyone's Fitness uses the standard {CTL_DAYS}-day constant, and friends' heart-rate "
        f"scores use the default resting {DEFAULT_HR_REST:.0f} / max {DEFAULT_HR_MAX:.0f} bpm, "
        "since their own are not known. Other sports count as set in the sidebar.")

    left, right = st.columns(2)
    with left:
        st.subheader("Best efforts")
        records = record_comparison(runs)
        if records.empty:
            st.info("No runs to compare in this period.")
        else:
            st.dataframe(pd.DataFrame({
                "Distance": records["distance"],
                **{name: records[name].map(format_duration) for name in names},
                "Fastest": records["leader"].fillna("--"),
                "By": records["margin_s"].map(
                    lambda s: format_duration(s) if pd.notna(s) else "--"),
            }), hide_index=True, width="stretch")
            st.caption("Fastest pace held over a whole run of at least each distance, "
                       "carried over exactly that distance. \"By\" is the gap to second place.")
    with right:
        st.subheader("Race predictions")
        predictions = prediction_comparison(runs)
        st.dataframe(pd.DataFrame({
            "Distance": predictions["distance"],
            **{name: predictions[name].map(format_duration) for name in names},
            "Fastest": predictions["leader"].fillna("--"),
        }), hide_index=True, width="stretch")
        st.caption(f"Riegel's model (exponent {RIEGEL_EXPONENT}) from each runner's fastest "
                   "run of 5 km or more in their last 90 days of the period.")

    st.subheader("Runs together")
    friend = (chosen[0] if len(chosen) == 1 else
              st.selectbox("Runs with", chosen, key="h2h_shared_friend"))
    cols = st.columns(2)
    window = cols[0].slider("Started within (minutes)", 1, 60, SHARED_WINDOW_MIN,
                            key="h2h_window")
    tolerance = cols[1].slider("Distances differ by at most", 5, 50,
                               round(SHARED_DISTANCE_TOLERANCE * 100), step=5, format="%d%%",
                               key="h2h_tolerance") / 100
    shared = shared_runs(runs[me], runs[friend], window, tolerance)
    if shared.empty:
        st.info(f"No runs in this period where you and {friend} started within {window} "
                f"minutes of each other and differ in distance by at most {tolerance:.0%}. "
                "Try loosening the matching above.")
        return
    faster = int((shared["pace_gap_s"] > 0).sum())
    cards = st.columns(3)
    cards[0].metric("Runs together", f"{len(shared):,}")
    cards[1].metric(f"{me} faster", f"{faster} of {len(shared)}")
    cards[2].metric("Average gap", f"{abs(units.pace(shared['pace_gap_s'].mean())):.0f} "
                                   f"s/{units.distance_label}",
                    delta=f"{me if shared['pace_gap_s'].mean() >= 0 else friend} faster",
                    delta_color="off")
    st.plotly_chart(plots.shared_runs_figure(shared, me, friend, units), width="stretch")
    st.caption(
        f"Runs that started within {window} minutes of each other and differ in "
        f"distance by at most {tolerance:.0%} -- most likely the same run. Paces still differ: one of you may have warmed up, "
        "waited at a light, or dropped the other at the end.")
    st.dataframe(pd.DataFrame({
        "Date": shared["date"].dt.strftime("%d %b %Y"),
        "Run": shared["name"],
        f"{me} distance": shared["distance_km"].map(units.format_distance),
        f"{friend} distance": shared["other_distance_km"].map(units.format_distance),
        f"{me} pace": shared["pace_min_per_km"].map(units.format_pace),
        f"{friend} pace": shared["other_pace_min_per_km"].map(units.format_pace),
    }).iloc[::-1], hide_index=True, width="stretch")


def main() -> None:
    path = choose_source()
    period_box = st.sidebar.container()  # filled once the runs are loaded
    units = choose_units()
    hr_rest, hr_max = choose_heart_rate()
    include_other, ftp = choose_other_sports(path)
    fitted = choose_time_constants()
    data = build_analytics(str(path), path.stat().st_mtime, hr_rest, hr_max, include_other, ftp)
    data = apply_time_constants(data, fitted)
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
        f"Fitness & Freshness, best efforts and workload management for "
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
    riegel = current_riegel(all_activities, all_streams)
    stream_data = (streams.restrict(all_streams, activities["activity_id"])
                   if all_streams is not None else None)

    (overview, fitness, race_plan, performance, heart_rate, pace, consistency, all_sports,
     run_detail, personal, gear, recovery, friends, share) = st.tabs(
        ["Overview", "Fitness & Freshness", "Race plan", "Performance", "Heart rate",
         "Pace zones", "Consistency", "All sports", "Run detail", "Personal models", "Gear",
         "Sleep & HRV", "Head to head", "Share"]
    )
    with overview:
        overview_tab(data)
    with fitness:
        fitness_tab(data)
    with race_plan:
        race_plan_tab(data)
    with performance:
        performance_tab(data, stream_data, riegel)
    with heart_rate:
        heart_rate_tab(data, stream_data)
    with pace:
        pace_zones_tab(data, stream_data)
    with consistency:
        consistency_tab(data)
    with all_sports:
        sports_tab(data)
    with run_detail:
        run_detail_tab(data, stream_data, path)
    with personal:
        personal_models_tab(data, riegel)
    with gear:
        gear_tab(data)
    with recovery:
        recovery_tab(data, is_sample=path == SAMPLE_CSV)
    with friends:
        head_to_head_tab(data, path, start, end)
    with share:
        share_tab(data, stream_data, path)


if __name__ == "__main__":
    main()
