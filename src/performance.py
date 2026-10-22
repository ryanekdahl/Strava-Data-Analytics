"""Best efforts, personal records, pace trends and Riegel race predictions.

Best efforts here are *activity-level*: the fastest average pace achieved over a
run of at least the target distance, a conservative proxy for a true in-run
split (see the README's Limitations section).

Race predictions use Riegel's endurance model ``T2 = T1 * (D2 / D1) ** 1.06``.
The exponent is Riegel's empirical fatigue factor: pace cannot hold as distance
grows, and 1.06 says the time penalty compounds slightly faster than distance.
It fits trained runners well between 5K and the half marathon and is optimistic
for marathon extrapolation from short efforts -- stated plainly, not hidden.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

RIEGEL_EXPONENT = 1.06

# Standard distances used for best efforts and the PR table.
STANDARD_DISTANCES: dict[str, float] = {
    "3K": 3.0,
    "5K": 5.0,
    "10K": 10.0,
    "15K": 15.0,
    "Half Marathon": 21.0975,
    "30K": 30.0,
    "Marathon": 42.195,
}

# Distances predicted on the Performance tab.
RACE_DISTANCES: dict[str, float] = {
    "5K": 5.0,
    "10K": 10.0,
    "Half Marathon": 21.0975,
    "Marathon": 42.195,
}

RECENT_WINDOW_DAYS = 90


class Effort(NamedTuple):
    """A single reference effort: ``time_s`` covering ``distance_km``."""

    distance_km: float
    time_s: float


def format_duration(seconds: float | None) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    if seconds is None or not np.isfinite(seconds):
        return "--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_pace(pace_min_per_km: float | None, suffix: str = " /km") -> str:
    """Format a min/km pace as ``M:SS /km`` (see :mod:`src.units` for miles)."""
    if pace_min_per_km is None or not np.isfinite(pace_min_per_km):
        return "--"
    minutes, seconds = divmod(int(round(pace_min_per_km * 60)), 60)
    return f"{minutes}:{seconds:02d}{suffix}"


def best_efforts(
    activities: pd.DataFrame, distances: dict[str, float] = STANDARD_DISTANCES
) -> pd.DataFrame:
    """Fastest qualifying run for each standard distance.

    For distance ``D``, considers every run of at least ``D`` km and keeps the
    one with the lowest average pace; ``implied_time_s`` is that pace carried
    over exactly ``D`` km.  Distances never run far enough for are omitted.
    """
    rows = []
    for label, distance_km in distances.items():
        qualifying = activities[
            (activities["distance_km"] >= distance_km)
            & activities["pace_min_per_km"].notna()
        ]
        if qualifying.empty:
            continue
        best = qualifying.loc[qualifying["pace_min_per_km"].idxmin()]
        pace = float(best["pace_min_per_km"])
        rows.append({
            "distance": label, "distance_km": distance_km,
            "activity_id": int(best["activity_id"]), "date": best["date"],
            "name": best["name"], "activity_distance_km": float(best["distance_km"]),
            "pace_min_per_km": pace, "implied_time_s": pace * distance_km * 60.0,
        })
    return pd.DataFrame(rows, columns=[
        "distance", "distance_km", "activity_id", "date", "name",
        "activity_distance_km", "pace_min_per_km", "implied_time_s"])


def best_effort_frontier(activities: pd.DataFrame) -> pd.DataFrame:
    """Pareto frontier of the pace-vs-distance cloud.

    A run is on the frontier when no longer run was also faster, i.e. it is the
    fastest thing achieved at or beyond its own distance.  Sorting by distance
    descending and keeping strict running-minimum paces gives that set, which is
    the curve highlighted on the scatter plot.
    """
    ordered = activities.sort_values("distance_km", ascending=False)
    ordered = ordered[ordered["pace_min_per_km"].notna()]
    running_min = ordered["pace_min_per_km"].cummin()
    on_frontier = ordered["pace_min_per_km"] <= running_min
    # cummin() includes the current row, so ties with a longer run must be cut.
    strictly_better = ordered["pace_min_per_km"] < running_min.shift(1).fillna(np.inf)
    frontier = ordered[on_frontier & strictly_better]
    return frontier.sort_values("distance_km").reset_index(drop=True)


def riegel_time(
    time_s: float, from_km: float, to_km: float, exponent: float = RIEGEL_EXPONENT
) -> float:
    """Predict a finishing time at ``to_km`` from a known effort at ``from_km``."""
    if from_km <= 0 or to_km <= 0:
        raise ValueError("distances must be positive")
    return float(time_s * (to_km / from_km) ** exponent)


def best_recent_effort(
    activities: pd.DataFrame,
    days: int = RECENT_WINDOW_DAYS,
    min_distance_km: float = 5.0,
) -> Effort | None:
    """The fastest run of at least ``min_distance_km`` in the last ``days``.

    Predictions should reflect current shape, not a two-year-old peak, so the
    reference effort comes from a recent window; if that window is empty the
    search widens to the whole history rather than returning nothing.
    """
    if activities.empty:
        return None
    cutoff = activities["date"].max() - pd.Timedelta(days=days)
    recent = activities[
        (activities["date"] >= cutoff) & (activities["distance_km"] >= min_distance_km)
    ]
    if recent.empty:
        recent = activities[activities["distance_km"] >= min_distance_km]
    if recent.empty:
        recent = activities
    if recent.empty:
        return None
    best = recent.loc[recent["pace_min_per_km"].idxmin()]
    return Effort(float(best["distance_km"]), float(best["moving_time_s"]))


def predict_races(
    best_effort: Effort | Sequence[float],
    distances: dict[str, float] = RACE_DISTANCES,
    exponent: float = RIEGEL_EXPONENT,
) -> pd.DataFrame:
    """Riegel predictions for each race distance from one reference effort.

    ``best_effort`` is an :class:`Effort` (or any ``(distance_km, time_s)`` pair).
    Returns ``DataFrame[distance, distance_km, predicted_time_s, predicted_time,
    predicted_pace]``.
    """
    effort = Effort(*best_effort)
    rows = []
    for label, distance_km in distances.items():
        predicted_s = riegel_time(effort.time_s, effort.distance_km, distance_km, exponent)
        rows.append({
            "distance": label, "distance_km": distance_km,
            "predicted_time_s": predicted_s,
            "predicted_time": format_duration(predicted_s),
            "predicted_pace": format_pace(predicted_s / 60.0 / distance_km),
        })
    return pd.DataFrame(rows)


def personal_records(activities: pd.DataFrame) -> pd.DataFrame:
    """All-time PR table, formatted for display."""
    records = best_efforts(activities)
    if records.empty:
        return records
    display = records.copy()
    display["time"] = display["implied_time_s"].map(format_duration)
    display["pace"] = display["pace_min_per_km"].map(format_pace)
    return display


def recent_vs_all_time(
    activities: pd.DataFrame, recent_days: int = RECENT_WINDOW_DAYS
) -> pd.DataFrame:
    """Compare recent best efforts against all-time bests, distance by distance.

    ``gap_pct`` is how much slower the recent best is than the all-time best,
    answering "am I near my peak right now?" without eyeballing two tables.
    """
    if activities.empty:
        return pd.DataFrame()
    cutoff = activities["date"].max() - pd.Timedelta(days=recent_days)
    all_time = best_efforts(activities).set_index("distance")
    recent = best_efforts(activities[activities["date"] >= cutoff]).set_index("distance")
    if all_time.empty:
        return pd.DataFrame()

    merged = all_time.join(recent[["pace_min_per_km", "implied_time_s"]], rsuffix="_recent")
    merged["gap_pct"] = (
        (merged["pace_min_per_km_recent"] - merged["pace_min_per_km"])
        / merged["pace_min_per_km"]
        * 100.0
    )
    merged["all_time_best"] = merged["implied_time_s"].map(format_duration)
    merged["recent_best"] = merged["implied_time_s_recent"].map(format_duration)
    merged["status"] = np.where(
        merged["pace_min_per_km_recent"].isna(),
        "no recent effort",
        np.where(merged["gap_pct"] <= 0.5, "at or near PR", "off PR pace"),
    )
    return merged.reset_index()[
        ["distance", "distance_km", "all_time_best", "recent_best", "gap_pct", "status"]
    ]


def rolling_pace(activities: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Distance-weighted rolling average pace, one point per run.

    Weighting by distance stops a 3 km shakeout from moving the average as much
    as a 20 km long run, so the trend tracks fitness, not run selection.
    """
    if activities.empty:
        return pd.DataFrame(columns=["date", "rolling_pace"])
    frame = activities[["date", "distance_km", "moving_time_s"]].dropna().set_index("date")
    window = f"{window_days}D"
    distance = frame["distance_km"].rolling(window).sum()
    minutes = (frame["moving_time_s"] / 60.0).rolling(window).sum()
    return pd.DataFrame({
        "date": frame.index, "rolling_pace": (minutes / distance).to_numpy(),
    }).reset_index(drop=True)


def rolling_hr(activities: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Rolling mean of average heart rate, skipping runs recorded without HR."""
    frame = activities[["date", "avg_hr"]].dropna()
    if frame.empty:
        return pd.DataFrame(columns=["date", "avg_hr", "rolling_hr"])
    indexed = frame.set_index("date")
    rolled = indexed["avg_hr"].rolling(f"{window_days}D").mean()
    return pd.DataFrame({
        "date": indexed.index,
        "avg_hr": indexed["avg_hr"].to_numpy(),
        "rolling_hr": rolled.to_numpy(),
    }).reset_index(drop=True)
