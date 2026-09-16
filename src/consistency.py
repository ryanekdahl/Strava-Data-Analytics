"""Volume, streaks and workload-ratio metrics -- the "am I training sanely?" view.

The headline metric is **ACWR**, the acute:chronic workload ratio::

    ACWR = mean daily load over the last 7 days
           / mean daily load over the last 28 days

It asks one question: is this week's training in proportion to what the body has
actually been prepared for?  Ratios roughly between 0.8 and 1.3 are the commonly
cited "sweet spot" in the sports-science literature; above ~1.5 the athlete is
ramping faster than adaptation allows (an injury-risk flag), and below 0.8 they
are detraining.  Means rather than sums are used so the ratio is 1.0 for a
perfectly steady block regardless of the window lengths.

The ratio is a *screening* tool, not a diagnosis: it is noisy on low volume, and
the exact risk thresholds are debated.  The dashboard therefore shades the band
rather than issuing verdicts.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

ACUTE_DAYS = 7
CHRONIC_DAYS = 28
SWEET_SPOT = (0.8, 1.3)
HIGH_RISK = 1.5


class Streaks(NamedTuple):
    """Consecutive-week running streaks, in weeks."""

    current: int
    longest: int


def weekly_volume(activities: pd.DataFrame) -> pd.DataFrame:
    """Distance, run count and time per calendar week (Monday-start).

    Weeks with no runs are filled with zeros so the bar chart shows the gaps
    instead of silently closing them up.
    """
    columns = ["week_start", "distance_km", "runs", "moving_time_h"]
    if activities.empty:
        return pd.DataFrame(columns=columns)

    indexed = activities.set_index("date")
    grouped = indexed.resample("W-MON", label="left", closed="left").agg(
        distance_km=("distance_km", "sum"),
        runs=("activity_id", "count"),
        moving_time_h=("moving_time_s", "sum"),
    )
    grouped["moving_time_h"] = grouped["moving_time_h"] / 3600.0
    grouped = grouped.asfreq("W-MON", fill_value=0.0)
    return grouped.reset_index(names="week_start")[columns]


def monthly_volume(activities: pd.DataFrame) -> pd.DataFrame:
    """Distance and run count per calendar month, zero-filled."""
    columns = ["month_start", "distance_km", "runs"]
    if activities.empty:
        return pd.DataFrame(columns=columns)

    indexed = activities.set_index("date")
    grouped = indexed.resample("MS").agg(
        distance_km=("distance_km", "sum"),
        runs=("activity_id", "count"),
    )
    grouped = grouped.asfreq("MS", fill_value=0.0)
    return grouped.reset_index(names="month_start")[columns]


def weekly_streaks(weekly: pd.DataFrame) -> Streaks:
    """Current and longest run of consecutive weeks containing at least one run.

    ``current`` counts back from the most recent week in the data; a week with
    no runs ends the streak, which is why :func:`weekly_volume` zero-fills.
    """
    if weekly.empty:
        return Streaks(0, 0)

    active = (weekly["runs"] > 0).to_numpy()
    longest = current = 0
    for was_active in active:
        current = current + 1 if was_active else 0
        longest = max(longest, current)
    return Streaks(int(current), int(longest))


def compute_acwr(
    daily_load: pd.Series, acute_days: int = ACUTE_DAYS, chronic_days: int = CHRONIC_DAYS
) -> pd.DataFrame:
    """Return ``DataFrame[date, acute, chronic, acwr]`` from a daily load series.

    Both legs are rolling *means* of daily load.  The ratio is undefined until
    ``chronic_days`` of history exist and whenever the chronic leg is zero (a
    total layoff), so those days are ``NaN`` rather than a misleading infinity.
    """
    daily_load = daily_load.astype("float64")
    acute = daily_load.rolling(acute_days, min_periods=acute_days).mean()
    chronic = daily_load.rolling(chronic_days, min_periods=chronic_days).mean()
    ratio = acute / chronic.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(daily_load.index),
            "acute": acute.to_numpy(),
            "chronic": chronic.to_numpy(),
            "acwr": ratio.to_numpy(),
        }
    ).reset_index(drop=True)


def acwr_status(value: float | None) -> str:
    """Plain-English reading of an ACWR value."""
    if value is None or not np.isfinite(value):
        return "Not enough history"
    low, high = SWEET_SPOT
    if value < low:
        return "Detraining -- load is dropping off"
    if value <= high:
        return "Sweet spot -- sustainable build"
    if value < HIGH_RISK:
        return "Ramping up -- watch recovery"
    return "Spiking -- elevated injury risk"


def summary_kpis(activities: pd.DataFrame) -> dict[str, float]:
    """Headline totals for the Overview tab."""
    if activities.empty:
        return {"total_runs": 0, "total_distance_km": 0.0, "total_hours": 0.0,
                "avg_distance_km": 0.0, "weeks": 0}
    span_days = (activities["date"].max() - activities["date"].min()).days + 1
    return {
        "total_runs": int(len(activities)),
        "total_distance_km": float(activities["distance_km"].sum()),
        "total_hours": float(activities["moving_time_s"].sum() / 3600.0),
        "avg_distance_km": float(activities["distance_km"].mean()),
        "weeks": max(int(round(span_days / 7)), 1),
    }
