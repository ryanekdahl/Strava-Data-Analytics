"""Fitness (CTL), Fatigue (ATL) and Form (TSB) from a transparent load model.

Strava Premium's Fitness & Freshness chart is built on a proprietary "Relative
Effort" score.  This module rebuilds the same idea from first principles, and
every constant below is documented so a reader can audit the number rather than
trust it.

Step 1 -- intensity factor
    ``IF = threshold_pace / run_pace``

    Paces are in min/km, so dividing threshold pace by the run's pace gives a
    *speed* ratio: a run exactly at threshold scores 1.0, a faster run scores
    above 1.0 and an easy run below.  Threshold pace is estimated from the
    athlete's own data (see :func:`estimate_threshold_pace`), which keeps the
    scale personal rather than absolute.

Step 2 -- pace-based load (an rTSS-style score)
    ``load_pace = duration_hours * IF ** 2 * 100``

    One hour at threshold = 100 points, by construction.  The square is the
    standard TSS convention: physiological cost rises faster than linearly with
    intensity, so a 30-minute hard run outweighs a 30-minute jog by more than
    their pace ratio alone.

Step 3 -- optional heart-rate term (Banister TRIMP)
    ``TRIMP = duration_min * hrr * 0.64 * exp(1.92 * hrr)``
    where ``hrr = (avg_hr - hr_rest) / (hr_max - hr_rest)``.

    TRIMP is rescaled so that one hour at threshold heart-rate reserve (0.85)
    also equals 100 points, putting it on the same axis as the pace score.  When
    ``avg_hr`` is present the two are blended ``hr_weight``/``1 - hr_weight``;
    when it is missing the pace score is used alone.  HR captures effort that
    pace misses (heat, hills, fatigue), which is exactly why it is blended in
    rather than used on its own.

Step 4 -- exponentially weighted averages of daily load
    ``CTL`` (Fitness) = EWMA with a 42-day time constant,
    ``ATL`` (Fatigue) = EWMA with a 7-day time constant,
    ``TSB`` (Form)    = ``CTL - ATL`` shifted one day.

    Both use the impulse-response form ``x_t = x_{t-1} + (load_t - x_{t-1}) / N``
    (equivalently ``alpha = 1 / N``), the convention used by the sports-science
    literature, seeded at zero so fitness builds from nothing rather than
    inheriting the first recorded run.  Form is shifted a day because the form
    you take into today's run is the balance you *finished yesterday* with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CTL_DAYS = 42
ATL_DAYS = 7

DEFAULT_HR_REST = 60.0
DEFAULT_HR_MAX = 190.0
DEFAULT_HR_WEIGHT = 0.5

THRESHOLD_QUANTILE = 0.10  # fastest decile of qualifying runs
THRESHOLD_MIN_DISTANCE_KM = 5.0
FALLBACK_THRESHOLD_PACE = 5.0

# TRIMP for one hour at 85% heart-rate reserve; used to rescale TRIMP to the
# same "100 points per threshold hour" axis as the pace score.
_THRESHOLD_HRR = 0.85
_TRIMP_REFERENCE = 60.0 * _THRESHOLD_HRR * 0.64 * float(np.exp(1.92 * _THRESHOLD_HRR))


def estimate_threshold_pace(
    activities: pd.DataFrame,
    min_distance_km: float = THRESHOLD_MIN_DISTANCE_KM,
    quantile: float = THRESHOLD_QUANTILE,
) -> float:
    """Estimate threshold pace (min/km) as the fastest decile of real runs.

    Only runs of at least ``min_distance_km`` count, so a 2 km sprint cannot
    masquerade as sustainable threshold effort.  If there is not enough data
    the module falls back to :data:`FALLBACK_THRESHOLD_PACE`, which keeps the
    load scale defined for a brand-new athlete.
    """
    paces = activities.loc[
        activities["distance_km"] >= min_distance_km, "pace_min_per_km"
    ].dropna()
    if paces.empty:
        paces = activities["pace_min_per_km"].dropna()
    if paces.empty:
        return FALLBACK_THRESHOLD_PACE
    return float(paces.quantile(quantile))


def trimp(
    moving_time_s: pd.Series,
    avg_hr: pd.Series,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: float = DEFAULT_HR_MAX,
) -> pd.Series:
    """Banister TRIMP, rescaled to 100 points per hour at threshold HR."""
    hrr = ((avg_hr - hr_rest) / (hr_max - hr_rest)).clip(lower=0.0, upper=1.0)
    raw = (moving_time_s / 60.0) * hrr * 0.64 * np.exp(1.92 * hrr)
    return raw * (100.0 / _TRIMP_REFERENCE)


def compute_activity_load(
    activities: pd.DataFrame,
    threshold_pace: float | None = None,
    hr_weight: float = DEFAULT_HR_WEIGHT,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: float = DEFAULT_HR_MAX,
) -> pd.DataFrame:
    """Return ``activities`` with ``intensity_factor``, the two component
    scores and the blended ``load`` column appended.

    Runs without ``avg_hr`` fall back to the pace score alone, so a missing
    heart-rate strap never silently deflates a day's training load.
    """
    result = activities.copy()
    if result.empty:
        for column in ("intensity_factor", "load_pace", "load_hr", "load"):
            result[column] = pd.Series(dtype="float64")
        return result

    if threshold_pace is None:
        threshold_pace = estimate_threshold_pace(result)

    duration_hours = result["moving_time_s"] / 3600.0
    result["intensity_factor"] = threshold_pace / result["pace_min_per_km"]
    result["load_pace"] = duration_hours * result["intensity_factor"] ** 2 * 100.0
    result["load_hr"] = trimp(result["moving_time_s"], result["avg_hr"], hr_rest, hr_max)

    blended = hr_weight * result["load_hr"] + (1.0 - hr_weight) * result["load_pace"]
    result["load"] = blended.where(result["load_hr"].notna(), result["load_pace"])
    return result


def daily_load_series(
    activities_with_load: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Sum load per calendar day over a gap-free daily index (rest days = 0).

    Zero-filling matters: the EWMAs must decay on days off, and skipping them
    would make a week of rest look like a week of training.
    """
    if activities_with_load.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="date"), name="load")

    days = activities_with_load["date"].dt.normalize()
    totals = activities_with_load.groupby(days)["load"].sum()
    calendar = pd.date_range(
        start=start or totals.index.min(),
        end=end or totals.index.max(),
        freq="D",
    )
    series = totals.reindex(calendar, fill_value=0.0).astype("float64")
    series.index.name = "date"
    series.name = "load"
    return series


def _ewma(values: np.ndarray, days: int) -> np.ndarray:
    """Impulse-response EWMA ``x_t = x_{t-1} + (load_t - x_{t-1}) / days``.

    Seeded at **zero**, not at the first day's load.  Pandas' ``adjust=False``
    seeds an EWMA with the first observation, which would hand an athlete a
    full day's training load as their starting fitness; prepending a zero day
    makes fitness build from nothing, as it does in life.
    """
    seeded = pd.Series(np.concatenate([[0.0], values]), dtype="float64")
    return seeded.ewm(alpha=1.0 / days, adjust=False).mean().to_numpy()[1:]


def compute_fitness_freshness(
    daily_load: pd.Series, ctl_days: int = CTL_DAYS, atl_days: int = ATL_DAYS
) -> pd.DataFrame:
    """Return ``DataFrame[date, load, ctl, atl, tsb]`` from a daily load series.

    ``ctl``/``atl`` are EWMAs with ``alpha = 1 / days`` seeded at zero, so a
    constant daily load converges to that load from below.  ``tsb`` is
    ``ctl - atl`` shifted one day; day zero is 0 because an athlete with no
    recorded history carries neither fitness nor fatigue.
    """
    values = daily_load.astype("float64").to_numpy()
    ctl = _ewma(values, ctl_days)
    atl = _ewma(values, atl_days)
    tsb = pd.Series(ctl - atl).shift(1).fillna(0.0).to_numpy()

    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(daily_load.index),
            "load": values,
            "ctl": ctl,
            "atl": atl,
            "tsb": tsb,
        }
    ).reset_index(drop=True)


def form_label(tsb: float) -> str:
    """Plain-English reading of a Form (TSB) value, for the dashboard."""
    if tsb > 15:
        return "Very fresh (well rested, maybe undertrained)"
    if tsb > 5:
        return "Fresh -- good day to race"
    if tsb >= -10:
        return "Neutral -- steady training"
    if tsb >= -25:
        return "Fatigued -- productive overload"
    return "Very fatigued -- back off soon"
