"""Head-to-head comparisons between you and friends who share their Strava exports.

Each friend is one more ``activities.csv`` -- requested from Strava exactly as
your own is -- saved in ``data/raw/friends/`` as ``<name>.csv`` or
``<name>/activities.csv``.  Every friend's file goes through the same
:func:`src.loader.load_activities` path as yours, so a friend's runs are cleaned
and scored by the same rules.

Fairness rules behind every comparison
--------------------------------------
* **Same window.**  Everyone is measured over the period picked in the sidebar,
  so a friend with a longer export does not win on history alone.  Fitness is
  still computed on each athlete's whole history and only then cut, as on the
  Fitness & Freshness tab.
* **Per-week rates as well as totals**, so "distance per week" and "weeks with a
  run" reward consistency rather than whoever happened to start recording first.
* **Fitness is relative to each athlete.**  Load is scored against each person's
  *own* threshold pace (and heart rate), so a Fitness of 50 means "50 points of
  work relative to my ability".  Comparing it says who is training harder for
  themselves, not who is faster -- the pace and best-effort tables answer that.
* **Best efforts are whole-run**, the fastest pace held over a run of at least
  each distance (:func:`src.performance.best_efforts`), since friends' GPS
  recordings are not read.
* **Runs together** are found by start time (within :data:`SHARED_WINDOW_MIN`
  minutes) and distance (within :data:`SHARED_DISTANCE_TOLERANCE`), the
  closest thing an export has to "we ran this together".
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.consistency import weekly_streaks
from src.loader import RAW_DIR, SAMPLE_CSV
from src.performance import (
    RACE_DISTANCES, STANDARD_DISTANCES, best_efforts, best_recent_effort, predict_races,
)

FRIENDS_DIR = RAW_DIR / "friends"
SAMPLE_FRIENDS_DIR = SAMPLE_CSV.parent / "friends"

SHARED_WINDOW_MIN = 15
SHARED_DISTANCE_TOLERANCE = 0.15
TIE = "Tie"

# key, label, whether the higher value wins
METRICS: list[tuple[str, str, bool]] = [
    ("distance_km", "Distance", True),
    ("runs", "Runs", True),
    ("hours", "Time on feet", True),
    ("elevation_m", "Elevation gain", True),
    ("longest_km", "Longest run", True),
    ("weekly_km", "Distance per week", True),
    ("active_share", "Weeks with a run", True),
    ("longest_streak", "Longest weekly streak", True),
    ("avg_pace", "Average pace", False),
    ("fitness", "Fitness (CTL) at period end", True),
]


def friend_name(path: Path) -> str:
    """The display name for a friend's export: the file's stem, or its folder's name."""
    stem = path.parent.name if path.name == "activities.csv" else path.stem
    return re.sub(r"[_\-]+", " ", stem).strip().title()


def find_friend_exports(folder: str | Path = FRIENDS_DIR) -> dict[str, Path]:
    """Every friend's export in ``folder``: ``<name>.csv`` files and ``<name>/activities.csv``."""
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    found = sorted(folder.glob("*.csv")) + sorted(folder.glob("*/activities.csv"))
    return {friend_name(path): path for path in found}


def safe_file_stem(name: str) -> str:
    """A filename stem for a friend's name: letters, digits and underscores only."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    if not stem:
        raise ValueError("A friend's name needs at least one letter or digit.")
    return stem


def save_friend_export(content: bytes, name: str, folder: str | Path = FRIENDS_DIR) -> Path:
    """Save an uploaded export as ``<folder>/<name>.csv`` and return its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe_file_stem(name)}.csv"
    path.write_bytes(content)
    return path


def cut_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows dated from ``start`` up to (not including) ``end``."""
    return frame[(frame["date"] >= start) & (frame["date"] < end)].reset_index(drop=True)


def period_weeks(activities: pd.DataFrame, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    """Distance and runs per Monday-start week across the whole period, zero-filled.

    Unlike :func:`src.consistency.weekly_volume` the weeks span the period, not
    just the athlete's first to last run, so two athletes get the same weeks.
    """
    weeks = pd.date_range(start.normalize() - pd.Timedelta(days=start.weekday()),
                          end - pd.Timedelta(days=1), freq="W-MON")
    if activities.empty:
        return pd.DataFrame({"week_start": weeks, "distance_km": 0.0, "runs": 0})
    week_start = (activities["date"].dt.normalize()
                  - pd.to_timedelta(activities["date"].dt.weekday, unit="D"))
    grouped = activities.groupby(week_start).agg(distance_km=("distance_km", "sum"),
                                                 runs=("activity_id", "count"))
    grouped = grouped.reindex(weeks, fill_value=0)
    return grouped.rename_axis("week_start").reset_index()


def athlete_profile(activities: pd.DataFrame, fitness: pd.DataFrame,
                    start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """Every :data:`METRICS` value for one athlete's runs in ``[start, end)``.

    ``activities`` and ``fitness`` may cover the whole history; both are cut here.
    """
    runs = cut_period(activities, start, end)
    weekly = period_weeks(runs, start, end)
    n_weeks = max(len(weekly), 1)
    distance = float(runs["distance_km"].sum())
    moving_s = float(runs["moving_time_s"].sum())
    before_end = fitness[fitness["date"] < end]
    return {
        "distance_km": distance,
        "runs": float(len(runs)),
        "hours": moving_s / 3600.0,
        "elevation_m": float(runs["elevation_gain_m"].sum()),
        "longest_km": float(runs["distance_km"].max()) if not runs.empty else np.nan,
        "weekly_km": distance / n_weeks,
        "active_share": float((weekly["runs"] > 0).mean()) if not weekly.empty else 0.0,
        "longest_streak": float(weekly_streaks(weekly).longest),
        "avg_pace": moving_s / 60.0 / distance if distance > 0 else np.nan,
        "fitness": float(before_end["ctl"].iloc[-1]) if not before_end.empty else np.nan,
    }


def leader(values: pd.Series, higher_is_better: bool = True) -> str | None:
    """The name (index label) with the best value, :data:`TIE` on a tie, ``None`` if all missing."""
    values = values.dropna()
    if values.empty:
        return None
    best = values.max() if higher_is_better else values.min()
    winners = values.index[np.isclose(values.to_numpy(dtype=float), best)]
    return str(winners[0]) if len(winners) == 1 else TIE


def head_to_head_table(profiles: dict[str, dict[str, float]]) -> pd.DataFrame:
    """One row per metric: each athlete's value and who leads.

    Returns ``DataFrame[metric, label, higher_is_better, <athlete>..., leader]``.
    """
    names = list(profiles)
    rows = []
    for key, label, higher in METRICS:
        values = pd.Series({name: profiles[name][key] for name in names}, dtype="float64")
        rows.append({"metric": key, "label": label, "higher_is_better": higher,
                     **values.to_dict(), "leader": leader(values, higher)})
    return pd.DataFrame(rows, columns=["metric", "label", "higher_is_better", *names, "leader"])


def scoreboard(table: pd.DataFrame, names: list[str]) -> pd.Series:
    """How many rows of a comparison table each athlete leads (ties count for no one)."""
    return table["leader"].value_counts().reindex(names, fill_value=0).astype(int)


def record_comparison(runs_by_name: dict[str, pd.DataFrame],
                      distances: dict[str, float] = STANDARD_DISTANCES) -> pd.DataFrame:
    """Each athlete's whole-run best time at each standard distance, and who is fastest.

    Returns ``DataFrame[distance, <athlete>..., leader, margin_s]`` where each
    athlete column holds seconds (``NaN`` if they never ran that far) and
    ``margin_s`` is the gap from the fastest to the next fastest.
    """
    names = list(runs_by_name)
    times = {name: best_efforts(runs, distances).set_index("distance")["implied_time_s"]
             for name, runs in runs_by_name.items()}
    rows = []
    for label in distances:
        values = pd.Series({name: times[name].get(label, np.nan) for name in names},
                           dtype="float64")
        if values.isna().all():
            continue
        ranked = values.dropna().sort_values()
        rows.append({"distance": label, **values.to_dict(), "leader": leader(values, False),
                     "margin_s": float(ranked.iloc[1] - ranked.iloc[0])
                     if len(ranked) > 1 else np.nan})
    return pd.DataFrame(rows, columns=["distance", *names, "leader", "margin_s"])


def prediction_comparison(runs_by_name: dict[str, pd.DataFrame],
                          exponent: float | None = None) -> pd.DataFrame:
    """Riegel race predictions for each athlete from their best recent run.

    Returns ``DataFrame[distance, <athlete>..., leader]`` in seconds.  Each
    athlete's reference is :func:`src.performance.best_recent_effort` on their
    own runs, so "recent" means their last 90 days in the window.
    """
    names = list(runs_by_name)
    columns = {}
    for name, runs in runs_by_name.items():
        effort = best_recent_effort(runs)
        if effort is None:
            columns[name] = pd.Series(np.nan, index=list(RACE_DISTANCES))
            continue
        kwargs = {} if exponent is None else {"exponent": exponent}
        predicted = predict_races(effort, **kwargs).set_index("distance")["predicted_time_s"]
        columns[name] = predicted
    wide = pd.DataFrame(columns, index=list(RACE_DISTANCES))
    wide["leader"] = [leader(row, False) for _, row in wide[names].iterrows()]
    return wide.rename_axis("distance").reset_index()


def weekly_battles(runs_by_name: dict[str, pd.DataFrame], start: pd.Timestamp,
                   end: pd.Timestamp) -> pd.DataFrame:
    """Distance per week for every athlete side by side, and who ran furthest.

    Returns ``DataFrame[week_start, <athlete>..., leader]``; a week nobody ran
    has no leader.
    """
    names = list(runs_by_name)
    wide = pd.DataFrame({name: period_weeks(runs, start, end).set_index("week_start")
                         ["distance_km"] for name, runs in runs_by_name.items()})
    wide["leader"] = [leader(row.where(row > 0), True) for _, row in wide[names].iterrows()]
    return wide.rename_axis("week_start").reset_index()


def cumulative_distance(runs_by_name: dict[str, pd.DataFrame], start: pd.Timestamp,
                        end: pd.Timestamp) -> pd.DataFrame:
    """Running total of distance per day through the period, one row per athlete-day.

    Returns ``DataFrame[date, athlete, distance_km]``.
    """
    days = pd.date_range(start.normalize(), end - pd.Timedelta(days=1), freq="D")
    frames = []
    for name, runs in runs_by_name.items():
        daily = (runs.groupby(runs["date"].dt.normalize())["distance_km"].sum()
                 .reindex(days, fill_value=0.0).cumsum())
        frames.append(pd.DataFrame({"date": days, "athlete": name,
                                    "distance_km": daily.to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["date", "athlete", "distance_km"])
    return pd.concat(frames, ignore_index=True)


def fitness_comparison(fitness_by_name: dict[str, pd.DataFrame], start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.DataFrame:
    """Each athlete's Fitness (CTL) through the period: ``DataFrame[date, athlete, ctl]``."""
    frames = [cut_period(fitness, start, end)[["date", "ctl"]].assign(athlete=name)
              for name, fitness in fitness_by_name.items()]
    if not frames:
        return pd.DataFrame(columns=["date", "athlete", "ctl"])
    return pd.concat(frames, ignore_index=True)[["date", "athlete", "ctl"]]


def shared_runs(mine: pd.DataFrame, theirs: pd.DataFrame,
                window_min: float = SHARED_WINDOW_MIN,
                tolerance: float = SHARED_DISTANCE_TOLERANCE) -> pd.DataFrame:
    """Runs two athletes most likely did together, with each one's pace.

    A pair matches when the runs start within ``window_min`` minutes of each
    other and their distances differ by at most ``tolerance`` of the longer one.
    Each run is matched at most once.  Returns ``DataFrame[date, name,
    distance_km, other_distance_km, pace_min_per_km, other_pace_min_per_km,
    pace_gap_s]`` where ``pace_gap_s`` is seconds per km by which the first
    athlete was faster (positive) or slower (negative).
    """
    columns = ["date", "name", "distance_km", "other_distance_km", "pace_min_per_km",
               "other_pace_min_per_km", "pace_gap_s"]
    if mine.empty or theirs.empty:
        return pd.DataFrame(columns=columns)
    left = mine[["date", "name", "distance_km", "pace_min_per_km"]].sort_values("date")
    right = (theirs[["date", "activity_id", "distance_km", "pace_min_per_km"]]
             .rename(columns={"date": "other_date", "activity_id": "other_id",
                              "distance_km": "other_distance_km",
                              "pace_min_per_km": "other_pace_min_per_km"})
             .sort_values("other_date"))
    merged = pd.merge_asof(left, right, left_on="date", right_on="other_date",
                           direction="nearest",
                           tolerance=pd.Timedelta(minutes=window_min)).dropna(subset=["other_id"])
    longer = merged[["distance_km", "other_distance_km"]].max(axis=1)
    close = (merged["distance_km"] - merged["other_distance_km"]).abs() <= tolerance * longer
    merged = merged[close].drop_duplicates(subset="other_id")
    merged["pace_gap_s"] = (merged["other_pace_min_per_km"] - merged["pace_min_per_km"]) * 60.0
    return merged[columns].reset_index(drop=True)
