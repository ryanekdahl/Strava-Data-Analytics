"""Heart-rate zone distribution: how training time splits across Z1-Z5.

Zones are defined on heart-rate reserve (the Karvonen method), the same
``hrr = (avg_hr - hr_rest) / (hr_max - hr_rest)`` scale the TRIMP term in
:mod:`src.training_load` uses, so one pair of ``hr_rest``/``hr_max`` values
drives both the load model and the zones:

====  =============  ==============
Zone  Name           % HR reserve
====  =============  ==============
Z1    Recovery       below 60%
Z2    Aerobic        60-70%
Z3    Tempo          70-80%
Z4    Threshold      80-90%
Z5    VO2 max        90% and above
====  =============  ==============

**Approximation, stated plainly.**  ``activities.csv`` carries one *average*
heart rate per run, not a second-by-second stream, so each run's whole moving
time is credited to the zone of its average.  A run of intervals averaging Z3
really spent its time split between Z1 and Z5; this distribution therefore
understates both ends and overstates the middle.

**True time in zone** comes from the GPS recordings when they are in the export:
:mod:`src.streams` stores how many seconds each run spent at every whole-number
heart rate, and :func:`stream_zone_distribution` credits each of those seconds
to its own zone.  Both functions return the same columns, so the dashboard can
show either -- or both side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training_load import DEFAULT_HR_MAX, DEFAULT_HR_REST

# (zone number, name, lower bound as a fraction of heart-rate reserve)
ZONES: list[tuple[int, str, float]] = [
    (1, "Recovery", 0.0),
    (2, "Aerobic", 0.60),
    (3, "Tempo", 0.70),
    (4, "Threshold", 0.80),
    (5, "VO2 max", 0.90),
]


def heart_rate_reserve(
    avg_hr: pd.Series, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.Series:
    """Average HR as a fraction of heart-rate reserve (unclipped; NaN stays NaN)."""
    if hr_max <= hr_rest:
        raise ValueError("hr_max must be greater than hr_rest")
    return (avg_hr - hr_rest) / (hr_max - hr_rest)


def assign_zone(
    avg_hr: pd.Series, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.Series:
    """Zone 1-5 for each run's average HR; runs without HR get ``<NA>``.

    Below the Z2 floor is Z1 and anything at or above 90% is Z5, so every run
    with a heart rate lands in exactly one zone.
    """
    hrr = heart_rate_reserve(avg_hr, hr_rest, hr_max)
    floors = [floor for _, _, floor in ZONES[1:]]
    zones = pd.Series(np.searchsorted(floors, hrr.to_numpy(), side="right") + 1,
                      index=avg_hr.index).astype("Int64")
    return zones.mask(hrr.isna())


def zone_table(hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX) -> pd.DataFrame:
    """The five zones with their bpm boundaries for this athlete."""
    reserve = hr_max - hr_rest
    rows = []
    for i, (zone, name, floor) in enumerate(ZONES):
        low = hr_rest + floor * reserve if zone > 1 else None
        high = hr_rest + ZONES[i + 1][2] * reserve if zone < len(ZONES) else None
        if low is None:
            bpm = f"below {high:.0f}"
        elif high is None:
            bpm = f"{low:.0f}+"
        else:
            bpm = f"{low:.0f}-{high:.0f}"
        rows.append({"zone": zone, "label": f"Z{zone} {name}", "bpm": bpm})
    return pd.DataFrame(rows)


def zone_distribution(
    activities: pd.DataFrame, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.DataFrame:
    """Runs, hours and share of HR-recorded time per zone, all five zones present.

    Returns ``DataFrame[zone, label, bpm, runs, hours, share]``; ``share`` is a
    fraction of time on runs *with* heart rate, so missing straps do not count
    as easy running.
    """
    table = zone_table(hr_rest, hr_max)
    zones = assign_zone(activities["avg_hr"], hr_rest, hr_max)
    with_hr = activities.assign(zone=zones).dropna(subset=["zone"]).astype({"zone": int})
    grouped = with_hr.groupby("zone").agg(
        runs=("activity_id", "size"), hours=("moving_time_s", lambda s: s.sum() / 3600.0))
    result = table.join(grouped, on="zone")
    result["runs"] = result["runs"].fillna(0).astype(int)
    result["hours"] = result["hours"].fillna(0.0)
    total = result["hours"].sum()
    result["share"] = result["hours"] / total if total > 0 else 0.0
    return result


def monthly_zone_hours(
    activities: pd.DataFrame, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.DataFrame:
    """Hours per zone per calendar month, zero-filled over a gap-free month index.

    Returns long-form ``DataFrame[month_start, zone, label, hours]``.
    """
    columns = ["month_start", "zone", "label", "hours"]
    zones = assign_zone(activities["avg_hr"], hr_rest, hr_max)
    with_hr = activities.assign(zone=zones).dropna(subset=["zone"]).astype({"zone": int})
    if with_hr.empty:
        return pd.DataFrame(columns=columns)

    months = with_hr["date"].dt.to_period("M").dt.to_timestamp()
    hours = with_hr.groupby([months, "zone"])["moving_time_s"].sum() / 3600.0
    calendar = pd.date_range(months.min(), months.max(), freq="MS")
    grid = pd.MultiIndex.from_product([calendar, [zone for zone, _, _ in ZONES]],
                                      names=["month_start", "zone"])
    frame = hours.reindex(grid, fill_value=0.0).rename("hours").reset_index()
    labels = zone_table(hr_rest, hr_max).set_index("zone")["label"]
    frame["label"] = frame["zone"].map(labels)
    return frame[columns]


def hr_coverage(activities: pd.DataFrame) -> float:
    """Fraction of runs that recorded an average heart rate."""
    if activities.empty:
        return 0.0
    return float(activities["avg_hr"].notna().mean())


def _stream_zones(hr_seconds: pd.DataFrame, hr_rest: float, hr_max: float) -> pd.DataFrame:
    """The per-bpm histogram rows with each bpm's zone attached."""
    zones = assign_zone(hr_seconds["bpm"].astype("float64"), hr_rest, hr_max)
    return hr_seconds.assign(zone=zones.astype(int))


def stream_zone_distribution(
    hr_seconds: pd.DataFrame, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.DataFrame:
    """True time in zone from second-by-second heart rate.

    ``hr_seconds`` is :attr:`src.streams.StreamData.hr_seconds`.  Returns the
    same ``DataFrame[zone, label, bpm, runs, hours, share]`` as
    :func:`zone_distribution`; ``runs`` counts runs that touched the zone.
    """
    table = zone_table(hr_rest, hr_max)
    if hr_seconds.empty:
        return table.assign(runs=0, hours=0.0, share=0.0)
    zoned = _stream_zones(hr_seconds, hr_rest, hr_max)
    grouped = zoned.groupby("zone").agg(
        runs=("activity_id", "nunique"), hours=("seconds", lambda s: s.sum() / 3600.0))
    result = table.join(grouped, on="zone")
    result["runs"] = result["runs"].fillna(0).astype(int)
    result["hours"] = result["hours"].fillna(0.0)
    total = result["hours"].sum()
    result["share"] = result["hours"] / total if total > 0 else 0.0
    return result


def stream_monthly_zone_hours(
    hr_seconds: pd.DataFrame, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.DataFrame:
    """Monthly hours per zone from second-by-second heart rate.

    Same long-form ``DataFrame[month_start, zone, label, hours]`` as
    :func:`monthly_zone_hours`, zero-filled over a gap-free month index.
    """
    columns = ["month_start", "zone", "label", "hours"]
    if hr_seconds.empty:
        return pd.DataFrame(columns=columns)
    zoned = _stream_zones(hr_seconds, hr_rest, hr_max)
    months = zoned["date"].dt.to_period("M").dt.to_timestamp()
    hours = zoned.groupby([months, "zone"])["seconds"].sum() / 3600.0
    calendar = pd.date_range(months.min(), months.max(), freq="MS")
    grid = pd.MultiIndex.from_product([calendar, [zone for zone, _, _ in ZONES]],
                                      names=["month_start", "zone"])
    frame = hours.reindex(grid, fill_value=0.0).rename("hours").reset_index()
    labels = zone_table(hr_rest, hr_max).set_index("zone")["label"]
    frame["label"] = frame["zone"].map(labels)
    return frame[columns]


def _per_run_frame(seconds: pd.Series, dates: pd.Series, hr_rest: float,
                   hr_max: float) -> pd.DataFrame:
    """Zero-fill ``seconds`` (indexed by activity_id, zone) to every zone of every run."""
    columns = ["activity_id", "date", "zone", "label", "minutes", "share"]
    if seconds.empty:
        return pd.DataFrame(columns=columns)
    grid = pd.MultiIndex.from_product(
        [seconds.index.get_level_values(0).unique(), [zone for zone, _, _ in ZONES]],
        names=["activity_id", "zone"])
    frame = (seconds.reindex(grid, fill_value=0.0) / 60.0).rename("minutes").reset_index()
    totals = frame.groupby("activity_id")["minutes"].transform("sum")
    frame["share"] = (frame["minutes"] / totals).where(totals > 0, 0.0)
    frame["date"] = frame["activity_id"].map(dates)
    labels = zone_table(hr_rest, hr_max).set_index("zone")["label"]
    frame["label"] = frame["zone"].map(labels)
    return frame.sort_values(["date", "activity_id", "zone"]).reset_index(drop=True)[columns]


def run_zone_breakdown(
    activities: pd.DataFrame, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.DataFrame:
    """Each run's minutes and share per zone, from its average heart rate.

    Under the average-HR approximation a run sits wholly in one zone, so every
    run has one zone at 100%.  Returns long-form
    ``DataFrame[activity_id, date, zone, label, minutes, share]`` with all five
    zones per run; runs without heart rate are left out.
    """
    zones = assign_zone(activities["avg_hr"], hr_rest, hr_max)
    with_hr = activities.assign(zone=zones).dropna(subset=["zone"]).astype({"zone": int})
    seconds = with_hr.groupby(["activity_id", "zone"])["moving_time_s"].sum().astype(float)
    return _per_run_frame(seconds, with_hr.set_index("activity_id")["date"], hr_rest, hr_max)


def stream_run_zone_breakdown(
    hr_seconds: pd.DataFrame, hr_rest: float = DEFAULT_HR_REST, hr_max: float = DEFAULT_HR_MAX
) -> pd.DataFrame:
    """Each run's true minutes and share per zone from second-by-second heart rate.

    Same long-form columns as :func:`run_zone_breakdown`.
    """
    if hr_seconds.empty:
        return _per_run_frame(pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]"),
                              hr_rest, hr_max)
    zoned = _stream_zones(hr_seconds, hr_rest, hr_max)
    seconds = zoned.groupby(["activity_id", "zone"])["seconds"].sum().astype(float)
    dates = zoned.drop_duplicates("activity_id").set_index("activity_id")["date"]
    return _per_run_frame(seconds, dates, hr_rest, hr_max)
