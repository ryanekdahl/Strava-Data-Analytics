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
understates both ends and overstates the middle.  True time-in-zone needs the
per-activity HR streams.
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
