"""Rides, swims, walks and strength sessions on the same load axis as runs.

Fitness and Fatigue are about the body, not the sport: a two-hour ride the day
before a long run is still fatigue.  This module scores every non-run activity
from :func:`src.loader.load_other_activities` on the scale
:mod:`src.training_load` uses for runs -- **100 points per hour at threshold**
-- so the two can be summed into one daily load.

Each session is scored with the best evidence it carries:

Heart rate -- Banister TRIMP
    The same rescaled TRIMP as runs (:func:`src.training_load.trimp`).  Heart
    rate is sport-agnostic, which is what makes it the backbone here.

Power -- TSS (rides, with an FTP set)
    ``TSS = duration_hours * (power / FTP) ** 2 * 100``, with Strava's weighted
    average power where present (it is a normalized power) and average watts
    otherwise.  It is the cycling twin of the run pace score, so with heart
    rate too the two are blended ``hr_weight`` / ``1 - hr_weight``, as runs are.

Neither -- a stated default intensity
    ``load = duration_hours * IF ** 2 * 100`` with a per-sport intensity factor
    from :data:`DEFAULT_INTENSITY`.  These are assumptions, not measurements:
    an easy ride at 0.65 scores 42 points an hour, a walk at 0.40 scores 16.
    Recording heart rate replaces them with the athlete's own effort.

Strava's many activity types are folded into a few families
(:data:`SPORT_FAMILIES`), so "Virtual Ride" and "Gravel Ride" count as one sport.
"""

from __future__ import annotations

import pandas as pd

from src.training_load import DEFAULT_HR_MAX, DEFAULT_HR_REST, DEFAULT_HR_WEIGHT, trimp

RUN = "Run"
OTHER = "Other"

# Strava activity type (case-folded) -> sport family shown on the dashboard.
SPORT_FAMILIES: dict[str, str] = {
    **dict.fromkeys(("ride", "virtual ride", "mountain bike ride", "gravel ride",
                     "e-bike ride", "e-mountain bike ride", "handcycle", "velomobile"), "Ride"),
    **dict.fromkeys(("swim",), "Swim"),
    **dict.fromkeys(("walk",), "Walk"),
    **dict.fromkeys(("hike",), "Hike"),
    **dict.fromkeys(("weight training", "workout", "crossfit", "hiit"), "Strength"),
    **dict.fromkeys(("rowing", "virtual row", "canoe", "kayaking", "stand up paddling"), "Row"),
    **dict.fromkeys(("alpine ski", "backcountry ski", "nordic ski", "snowboard",
                     "snowshoe"), "Ski"),
    **dict.fromkeys(("yoga", "pilates"), "Yoga"),
    **dict.fromkeys(("elliptical", "stair-stepper"), "Cardio machine"),
}

# Intensity factor assumed when a session has neither heart rate nor power.
DEFAULT_INTENSITY: dict[str, float] = {
    "Ride": 0.65,
    "Swim": 0.70,
    "Walk": 0.40,
    "Hike": 0.55,
    "Strength": 0.60,
    "Row": 0.70,
    "Ski": 0.60,
    "Yoga": 0.35,
    "Cardio machine": 0.60,
    OTHER: 0.55,
}

# How each session's load was worked out, for the dashboard's table.
METHOD_HR_POWER = "Heart rate + power"
METHOD_HR = "Heart rate"
METHOD_POWER = "Power"
METHOD_DEFAULT = "Default intensity"


def sport_family(activity_type: pd.Series) -> pd.Series:
    """Fold Strava's activity types into the families of :data:`SPORT_FAMILIES`."""
    key = activity_type.astype("string").str.strip().str.casefold()
    return key.map(SPORT_FAMILIES).fillna(OTHER).astype(str)


def power_tss(moving_time_s: pd.Series, watts: pd.Series, ftp: float | None) -> pd.Series:
    """Cycling TSS, 100 points per hour at FTP; ``NaN`` without power or an FTP."""
    if not ftp or ftp <= 0:
        return pd.Series(float("nan"), index=moving_time_s.index)
    return (moving_time_s / 3600.0) * (watts / ftp) ** 2 * 100.0


def compute_other_load(
    other: pd.DataFrame,
    ftp: float | None = None,
    hr_weight: float = DEFAULT_HR_WEIGHT,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: float = DEFAULT_HR_MAX,
) -> pd.DataFrame:
    """Return ``other`` with ``sport``, ``load_hr``, ``load_power``,
    ``load_default``, the final ``load`` and the ``load_method`` behind it.

    Power counts only for rides: a power figure on a row or ski erg is on a
    different scale from cycling FTP.
    """
    result = other.copy()
    result["sport"] = sport_family(result["activity_type"]) if not result.empty \
        else pd.Series(dtype=str)
    hours = result["moving_time_s"] / 3600.0
    intensity = result["sport"].map(DEFAULT_INTENSITY).fillna(DEFAULT_INTENSITY[OTHER])
    result["load_default"] = (hours * intensity ** 2 * 100.0).astype("float64")
    result["load_hr"] = trimp(result["moving_time_s"], result["avg_hr"], hr_rest, hr_max)
    watts = result["weighted_watts"].fillna(result["avg_watts"])
    result["load_power"] = power_tss(result["moving_time_s"], watts, ftp).where(
        result["sport"] == "Ride")

    has_hr, has_power = result["load_hr"].notna(), result["load_power"].notna()
    blended = hr_weight * result["load_hr"] + (1.0 - hr_weight) * result["load_power"]
    result["load"] = (blended.where(has_hr & has_power)
                      .fillna(result["load_hr"])
                      .fillna(result["load_power"])
                      .fillna(result["load_default"])
                      .astype("float64"))
    method = pd.Series(METHOD_DEFAULT, index=result.index, dtype=object)
    method[has_power] = METHOD_POWER
    method[has_hr] = METHOD_HR
    method[has_hr & has_power] = METHOD_HR_POWER
    result["load_method"] = method.astype(str)
    return result


def all_sessions(runs_scored: pd.DataFrame, other_scored: pd.DataFrame) -> pd.DataFrame:
    """Runs and everything else in one frame: ``date, name, sport, distance_km,
    moving_time_s, avg_hr, load, load_method``, oldest first."""
    columns = ["activity_id", "date", "name", "sport", "distance_km", "moving_time_s",
               "avg_hr", "load", "load_method"]
    runs = runs_scored.assign(
        sport=RUN,
        load_method=runs_scored["avg_hr"].notna().map(
            {True: "Pace + heart rate", False: "Pace"}).astype(str),
    )
    frames = [frame[columns] for frame in (runs, other_scored) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return (pd.concat(frames, ignore_index=True).sort_values("date", kind="stable")
            .reset_index(drop=True))


def sport_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    """Sessions, hours, distance and load per sport, biggest share of load first."""
    columns = ["sport", "sessions", "hours", "distance_km", "load", "load_share"]
    if sessions.empty:
        return pd.DataFrame(columns=columns)
    summary = sessions.groupby("sport").agg(
        sessions=("date", "count"),
        hours=("moving_time_s", lambda s: s.sum() / 3600.0),
        distance_km=("distance_km", "sum"),
        load=("load", "sum"),
    ).reset_index()
    total = summary["load"].sum()
    summary["load_share"] = summary["load"] / total if total > 0 else 0.0
    return summary.sort_values("load", ascending=False).reset_index(drop=True)[columns]


def weekly_sport_hours(sessions: pd.DataFrame) -> pd.DataFrame:
    """Long-form ``week_start, sport, hours, load``, zero-filled for every week and sport."""
    columns = ["week_start", "sport", "hours", "load"]
    if sessions.empty:
        return pd.DataFrame(columns=columns)
    week = sessions["date"].dt.to_period("W-SUN").dt.start_time
    totals = (sessions.assign(week_start=week, hours=sessions["moving_time_s"] / 3600.0)
              .groupby(["week_start", "sport"])[["hours", "load"]].sum())
    weeks = pd.date_range(week.min(), week.max(), freq="W-MON")
    sports = sessions["sport"].unique()
    full = pd.MultiIndex.from_product([weeks, sports], names=["week_start", "sport"])
    return totals.reindex(full, fill_value=0.0).reset_index()[columns]
