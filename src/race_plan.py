"""Race planner: project Fitness, Fatigue and Form forward to a race date.

Everything else in the dashboard describes the past.  This module runs the same
CTL / ATL / TSB equations from :mod:`src.training_load` *forward*, over a plan of
weekly training load the athlete types in, so they can see what Form they would
carry onto the start line -- and adjust the plan before the race, not after.

The plan
    The days from the day after the last recorded session up to the day before
    the race are cut into 7-day blocks counted *backwards* from race day, so the
    last block is always the final week before the race and only the first block
    can be short.  Each block carries one number: its planned training load, in
    the same points as every run (100 per hour at threshold).

Spreading a week's load over its days
    A block's load is spread evenly over its days.  Where in the week the hard
    sessions fall moves Form by a few points on any one day; it barely moves the
    smoothed Fitness, and an even spread keeps the assumption visible rather
    than inventing a session schedule.

The template
    :func:`template_plan` fills the blocks from three numbers: a starting weekly
    load (by default the last four weeks' average), a weekly increase for the
    build, and a taper of one to three weeks at the end.  The taper cuts volume
    by roughly 40-50% over the final fortnight, the range a meta-analysis of
    tapering studies found to work best (Bosquet et al., 2007).

The projection
    History and plan are joined into one gap-free daily series and run through
    :func:`~src.training_load.compute_fitness_freshness`, so the projection
    continues the exact EWMAs behind the Fitness & Freshness chart rather than
    re-seeding them.  Race-day Form is read off the race date itself, which --
    with Form lagged one day -- is the balance finished the night before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training_load import ATL_DAYS, CTL_DAYS, compute_fitness_freshness

# Form in this band on race day reads as "fresh" in form_label; above it,
# fitness is being given away to rest.
RACE_READY_FORM = (5.0, 15.0)

# Share of the starting weekly load kept in each taper week, furthest first.
TAPER_SHAPES: dict[int, tuple[float, ...]] = {
    0: (),
    1: (0.6,),
    2: (0.75, 0.55),
    3: (0.85, 0.7, 0.55),
}

# A rise in Fitness of more than this many points in a week is the commonly
# quoted rule of thumb for a build that is too steep to absorb.
MAX_SAFE_RAMP = 8.0

RECENT_DAYS = 28


def recent_weekly_load(daily_load: pd.Series, days: int = RECENT_DAYS) -> float:
    """Average weekly load over the last ``days`` of the history."""
    if daily_load.empty:
        return 0.0
    return float(daily_load.iloc[-days:].sum() / min(days, len(daily_load)) * 7.0)


def plan_weeks(plan_start: pd.Timestamp, race_date: pd.Timestamp) -> pd.DataFrame:
    """Blocks of at most 7 days from ``plan_start`` up to the day before the race.

    Returns ``DataFrame[week_start, days, weeks_out]``, counted back from race
    day so the last row is the final week before the race (``weeks_out`` 1).
    Empty when the race is not after ``plan_start``.
    """
    plan_start, race_date = pd.Timestamp(plan_start).normalize(), pd.Timestamp(race_date).normalize()
    total = (race_date - plan_start).days
    if total <= 0:
        return pd.DataFrame({"week_start": pd.DatetimeIndex([]), "days": pd.Series(dtype="int64"),
                             "weeks_out": pd.Series(dtype="int64")})
    rows = []
    end = race_date
    weeks_out = 1
    while end > plan_start:
        start = max(end - pd.Timedelta(days=7), plan_start)
        rows.append({"week_start": start, "days": (end - start).days, "weeks_out": weeks_out})
        end, weeks_out = start, weeks_out + 1
    return pd.DataFrame(rows[::-1]).reset_index(drop=True)


def template_plan(weeks: pd.DataFrame, weekly_load: float, weekly_increase: float = 0.0,
                  taper_weeks: int = 2) -> pd.DataFrame:
    """``weeks`` with a ``load`` column: a steady build, then a taper.

    Build blocks grow by ``weekly_increase`` (a fraction, 0.05 = 5%) a week from
    ``weekly_load``; the last ``taper_weeks`` blocks keep the :data:`TAPER_SHAPES`
    share of the peak build week instead.  A short first block gets the share of
    its week's load that its days cover.
    """
    result = weeks.copy()
    n = len(result)
    shape = TAPER_SHAPES[taper_weeks][-n:] if n else ()
    build = n - len(shape)
    rates = [weekly_load * (1.0 + weekly_increase) ** i for i in range(build)]
    peak = rates[-1] if rates else weekly_load
    rates += [peak * share for share in shape]
    result["load"] = np.round(np.array(rates, dtype="float64") * result["days"] / 7.0)
    return result


def plan_daily_load(plan: pd.DataFrame) -> pd.Series:
    """Spread each block's ``load`` evenly over its days; one value per day."""
    if plan.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="date"), name="load")
    pieces = [
        pd.Series(float(row.load) / row.days,
                  index=pd.date_range(row.week_start, periods=row.days, freq="D"))
        for row in plan.itertuples()
    ]
    series = pd.concat(pieces).astype("float64").clip(lower=0.0)
    series.index.name = "date"
    series.name = "load"
    return series


def project_fitness(daily_load: pd.Series, plan: pd.DataFrame, race_date: pd.Timestamp,
                    ctl_days: int = CTL_DAYS, atl_days: int = ATL_DAYS) -> pd.DataFrame:
    """Fitness, Fatigue and Form through race day, history and plan together.

    Returns the :func:`compute_fitness_freshness` frame plus a ``projected``
    flag for the planned days.  Race day itself carries no load: its Form is
    what the plan leaves you with, whatever the race then costs.
    """
    race_date = pd.Timestamp(race_date).normalize()
    planned = plan_daily_load(plan)
    race_day = pd.Series([0.0], index=pd.DatetimeIndex([race_date], name="date"), name="load")
    combined = pd.concat([daily_load.astype("float64"), planned, race_day])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    calendar = pd.date_range(combined.index.min(), race_date, freq="D", name="date")
    combined = combined.reindex(calendar, fill_value=0.0)

    result = compute_fitness_freshness(combined, ctl_days, atl_days)
    last_recorded = daily_load.index.max() if not daily_load.empty else calendar[0] - pd.Timedelta(days=1)
    result["projected"] = result["date"] > last_recorded
    return result


def race_day_summary(projection: pd.DataFrame) -> dict[str, float]:
    """The numbers the planner reports: race-day values, change and the steepest week.

    ``ctl_change`` is race-day Fitness minus Fitness on the last recorded day;
    ``max_ramp`` is the largest rise in Fitness over any 7 projected days.
    """
    race = projection.iloc[-1]
    history = projection[~projection["projected"]]
    start_ctl = float(history["ctl"].iloc[-1]) if not history.empty else 0.0
    ctl = projection["ctl"].to_numpy()
    ramps = ctl[7:] - ctl[:-7]
    projected_ramps = ramps[projection["projected"].to_numpy()[7:]]
    return {
        "ctl": float(race["ctl"]),
        "atl": float(race["atl"]),
        "tsb": float(race["tsb"]),
        "ctl_change": float(race["ctl"]) - start_ctl,
        "max_ramp": float(projected_ramps.max()) if projected_ramps.size else 0.0,
        "min_tsb": float(projection.loc[projection["projected"], "tsb"].min()),
    }


def race_day_advice(summary: dict[str, float]) -> str:
    """One plain-English sentence on whether the plan lands in the race-ready band."""
    low, high = RACE_READY_FORM
    tsb = summary["tsb"]
    if tsb < low:
        return (f"Race-day Form of {tsb:+.0f} is below the {low:+.0f} to {high:+.0f} band -- "
                "you would start carrying fatigue. Cut the last week or two harder, or "
                "lengthen the taper.")
    if tsb > high:
        return (f"Race-day Form of {tsb:+.0f} is above the {low:+.0f} to {high:+.0f} band -- "
                "fresh, but resting longer than you need to costs Fitness. A shorter or "
                "gentler taper would keep more of it.")
    return (f"Race-day Form of {tsb:+.0f} sits in the {low:+.0f} to {high:+.0f} band: "
            "rested, without giving away much Fitness.")
