"""Pace zones: how running time splits across Z1-Z5 by *speed*, not heart rate.

Zones are fractions of **threshold speed** -- the speed of the threshold pace
:func:`src.training_load.estimate_threshold_pace` finds in your own runs -- so a
zone is the same intensity factor ``IF = threshold_pace / pace`` that scores
training load:

====  ==========  ==================  ================================
Zone  Name        % threshold speed   Feels like
====  ==========  ==================  ================================
Z1    Recovery    below 78%           shakeouts, warm-ups
Z2    Endurance   78-88%              easy and long runs
Z3    Tempo       88-95%              steady, "comfortably hard"
Z4    Threshold   95-103%             around one-hour race effort
Z5    VO2 max     103% and above      intervals, 5K pace and faster
====  ==========  ==================  ================================

Pace is **grade-adjusted** wherever it can be (Strava's whole-run GAP from
``activities.csv``; the slope model in :mod:`src.streams` for the GPS
recordings), because a slow climb is not an easy run.

Like :mod:`src.heart_rate`, there are two ways to count: crediting each run's
whole moving time to the zone of its average pace (all ``activities.csv``
offers), or crediting every second of a GPS recording to its own zone.  An
interval session averages out to Z3 the first way and shows its Z1 jogs and Z5
reps the second.  Both return the same columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training_load import effort_pace
from src.units import METRIC, Units

# (zone number, name, lower bound as a fraction of threshold speed)
ZONES: list[tuple[int, str, float]] = [
    (1, "Recovery", 0.0),
    (2, "Endurance", 0.78),
    (3, "Tempo", 0.88),
    (4, "Threshold", 0.95),
    (5, "VO2 max", 1.03),
]
ZONE_NUMBERS = [zone for zone, _, _ in ZONES]


def _zone_of_speed_fraction(fraction: np.ndarray) -> np.ndarray:
    floors = [floor for _, _, floor in ZONES[1:]]
    return np.searchsorted(floors, fraction, side="right") + 1


def assign_zone(pace_min_per_km: pd.Series, threshold_pace: float) -> pd.Series:
    """Zone 1-5 for each pace (min/km); missing or non-positive paces get ``<NA>``."""
    if threshold_pace <= 0:
        raise ValueError("threshold_pace must be positive")
    pace = pace_min_per_km.astype("float64")
    valid = pace.notna() & (pace > 0)
    fraction = (threshold_pace / pace.where(valid)).fillna(0.0).to_numpy()
    zones = pd.Series(_zone_of_speed_fraction(fraction), index=pace.index).astype("Int64")
    return zones.mask(~valid)


def zone_table(threshold_pace: float, units: Units = METRIC) -> pd.DataFrame:
    """The five zones with their pace boundaries, in ``units``.

    Returns ``DataFrame[zone, label, pace, slow_min_per_km, fast_min_per_km]``;
    ``slow`` is the zone's slow edge (NaN for Z1) and ``fast`` its fast edge
    (NaN for Z5), both in min/km.
    """
    rows = []
    for i, (zone, name, floor) in enumerate(ZONES):
        slow = threshold_pace / floor if floor > 0 else np.nan
        fast = threshold_pace / ZONES[i + 1][2] if i + 1 < len(ZONES) else np.nan
        if np.isnan(slow):
            text = f"slower than {units.format_pace(fast)}"
        elif np.isnan(fast):
            text = f"faster than {units.format_pace(slow)}"
        else:
            text = f"{units.format_pace(slow)} to {units.format_pace(fast)}"
        rows.append({"zone": zone, "label": f"Z{zone} {name}", "pace": text,
                     "slow_min_per_km": slow, "fast_min_per_km": fast})
    return pd.DataFrame(rows)


def _summarise(seconds_by_zone: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Join per-zone runs and hours onto the zone table and add each zone's share."""
    result = table.join(seconds_by_zone, on="zone")
    result["runs"] = result["runs"].fillna(0).astype(int)
    result["hours"] = result["hours"].fillna(0.0)
    total = result["hours"].sum()
    result["share"] = result["hours"] / total if total > 0 else 0.0
    return result


def _monthly(zoned: pd.DataFrame, seconds: str, table: pd.DataFrame) -> pd.DataFrame:
    """Hours per zone per month over a gap-free month index (long form)."""
    columns = ["month_start", "zone", "label", "hours"]
    if zoned.empty:
        return pd.DataFrame(columns=columns)
    months = zoned["date"].dt.to_period("M").dt.to_timestamp()
    hours = zoned.groupby([months, "zone"])[seconds].sum() / 3600.0
    calendar = pd.date_range(months.min(), months.max(), freq="MS")
    grid = pd.MultiIndex.from_product([calendar, ZONE_NUMBERS], names=["month_start", "zone"])
    frame = hours.reindex(grid, fill_value=0.0).rename("hours").reset_index()
    frame["label"] = frame["zone"].map(table.set_index("zone")["label"])
    return frame[columns]


def _per_run(seconds: pd.Series, dates: pd.Series, table: pd.DataFrame) -> pd.DataFrame:
    """Zero-fill ``seconds`` (indexed by activity_id, zone) to every zone of every run."""
    columns = ["activity_id", "date", "zone", "label", "minutes", "share"]
    if seconds.empty:
        return pd.DataFrame(columns=columns)
    grid = pd.MultiIndex.from_product(
        [seconds.index.get_level_values(0).unique(), ZONE_NUMBERS],
        names=["activity_id", "zone"])
    frame = (seconds.reindex(grid, fill_value=0.0) / 60.0).rename("minutes").reset_index()
    totals = frame.groupby("activity_id")["minutes"].transform("sum")
    frame["share"] = (frame["minutes"] / totals).where(totals > 0, 0.0)
    frame["date"] = frame["activity_id"].map(dates)
    frame["label"] = frame["zone"].map(table.set_index("zone")["label"])
    return frame.sort_values(["date", "activity_id", "zone"]).reset_index(drop=True)[columns]


# --------------------------------------------------------------------------
# Whole runs, from activities.csv
# --------------------------------------------------------------------------

def _zoned_runs(activities: pd.DataFrame, threshold_pace: float) -> pd.DataFrame:
    zones = assign_zone(effort_pace(activities), threshold_pace)
    return activities.assign(zone=zones).dropna(subset=["zone"]).astype({"zone": int})


def zone_distribution(activities: pd.DataFrame, threshold_pace: float,
                      units: Units = METRIC) -> pd.DataFrame:
    """Runs, hours and share of time per zone, each run at its average (GAP) pace.

    Returns ``DataFrame[zone, label, pace, slow_min_per_km, fast_min_per_km,
    runs, hours, share]`` with all five zones present.
    """
    zoned = _zoned_runs(activities, threshold_pace)
    grouped = zoned.groupby("zone").agg(
        runs=("activity_id", "size"), hours=("moving_time_s", lambda s: s.sum() / 3600.0))
    return _summarise(grouped, zone_table(threshold_pace, units))


def monthly_zone_hours(activities: pd.DataFrame, threshold_pace: float) -> pd.DataFrame:
    """Hours per zone per month, each run at its average pace: ``[month_start, zone, label, hours]``."""
    return _monthly(_zoned_runs(activities, threshold_pace), "moving_time_s",
                    zone_table(threshold_pace))


def run_zone_breakdown(activities: pd.DataFrame, threshold_pace: float) -> pd.DataFrame:
    """Each run's minutes and share per zone; under this method one zone gets 100%.

    Long form ``DataFrame[activity_id, date, zone, label, minutes, share]``.
    """
    zoned = _zoned_runs(activities, threshold_pace)
    seconds = zoned.groupby(["activity_id", "zone"])["moving_time_s"].sum().astype(float)
    return _per_run(seconds, zoned.set_index("activity_id")["date"], zone_table(threshold_pace))


# --------------------------------------------------------------------------
# Second by second, from the GPS recordings
# --------------------------------------------------------------------------

def _zoned_seconds(gap_seconds: pd.DataFrame, threshold_pace: float) -> pd.DataFrame:
    """The speed-histogram rows (:attr:`src.streams.StreamData.gap_seconds`) with zones."""
    pace = 1000.0 / gap_seconds["speed_m_s"].astype("float64") / 60.0
    zones = assign_zone(pace, threshold_pace)
    return gap_seconds.assign(zone=zones).dropna(subset=["zone"]).astype({"zone": int})


def stream_zone_distribution(gap_seconds: pd.DataFrame, threshold_pace: float,
                             units: Units = METRIC) -> pd.DataFrame:
    """True time in zone from the GPS recordings; same columns as :func:`zone_distribution`.

    ``runs`` counts runs that touched the zone.
    """
    table = zone_table(threshold_pace, units)
    if gap_seconds.empty:
        return table.assign(runs=0, hours=0.0, share=0.0)
    zoned = _zoned_seconds(gap_seconds, threshold_pace)
    grouped = zoned.groupby("zone").agg(
        runs=("activity_id", "nunique"), hours=("seconds", lambda s: s.sum() / 3600.0))
    return _summarise(grouped, table)


def stream_monthly_zone_hours(gap_seconds: pd.DataFrame, threshold_pace: float) -> pd.DataFrame:
    """Monthly hours per zone from the GPS recordings."""
    if gap_seconds.empty:
        return pd.DataFrame(columns=["month_start", "zone", "label", "hours"])
    return _monthly(_zoned_seconds(gap_seconds, threshold_pace), "seconds",
                    zone_table(threshold_pace))


def stream_run_zone_breakdown(gap_seconds: pd.DataFrame, threshold_pace: float) -> pd.DataFrame:
    """Each run's true minutes and share per zone; same columns as :func:`run_zone_breakdown`."""
    table = zone_table(threshold_pace)
    if gap_seconds.empty:
        return _per_run(pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]"), table)
    zoned = _zoned_seconds(gap_seconds, threshold_pace)
    seconds = zoned.groupby(["activity_id", "zone"])["seconds"].sum().astype(float)
    dates = zoned.drop_duplicates("activity_id").set_index("activity_id")["date"]
    return _per_run(seconds, dates, table)
