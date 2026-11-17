"""Near-PR alerts: how far your recent form is from each personal record.

Strava's Best Efforts only speak up once a record falls.  A runner eight
seconds off their 5K best hears nothing, though that is exactly when a PR
attempt is most motivating.  This module turns the best-effort history into
a per-distance answer to "could I break it now, and what would it take?".

For each named distance it compares the fastest effort of the last
``recent_days`` (counted back from the latest run, like the rest of the app)
with the all-time record:

``gap_pct``   how much slower the recent best is than the PR, in percent
``gap_s``     the same gap in seconds over the whole distance
``status``    ``New PR`` when the record itself was set in the window;
              ``Within reach`` up to ``WITHIN_REACH_PCT`` off;
              ``Getting close`` up to ``GETTING_CLOSE_PCT``;
              ``Off PR pace`` beyond that; ``No recent effort`` when the
              window holds no effort at that distance.
``target_pace_min_per_km``  the even pace that beats the PR by one second.

The input is the long best-effort frame from ``src/streams.py`` (true in-run
efforts from the GPS files), or :func:`whole_run_efforts` when an export has
no GPS files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.performance import RECENT_WINDOW_DAYS
from src.streams import NAMED_DISTANCES_M

WITHIN_REACH_PCT = 3.0
GETTING_CLOSE_PCT = 8.0
NEW_PR, WITHIN_REACH, GETTING_CLOSE, OFF_PACE, NO_RECENT = (
    "New PR", "Within reach", "Getting close", "Off PR pace", "No recent effort")
STATUS_ORDER = [NEW_PR, WITHIN_REACH, GETTING_CLOSE, OFF_PACE, NO_RECENT]

COLUMNS = ["distance", "distance_km", "pr_time_s", "pr_date", "pr_activity_id",
           "recent_time_s", "recent_date", "recent_activity_id", "gap_s", "gap_pct",
           "target_pace_min_per_km", "pr_age_days", "status"]


def whole_run_efforts(activities: pd.DataFrame) -> pd.DataFrame:
    """Best-effort rows from whole runs, for exports without GPS files.

    Each run counts at every named distance it covers, at its average pace --
    the same assumption as ``personal_records``.  Returns the long
    ``DataFrame[activity_id, date, distance_m, time_s]`` that
    :func:`near_pr_table` takes.
    """
    rows = []
    for run in activities.itertuples():
        run_m = run.distance_km * 1000.0
        if not run_m > 0 or not run.moving_time_s > 0:
            continue
        for meters in NAMED_DISTANCES_M.values():
            if meters <= run_m:
                rows.append((run.activity_id, run.date, meters,
                             run.moving_time_s * meters / run_m))
    return pd.DataFrame(rows, columns=["activity_id", "date", "distance_m", "time_s"])


def classify(gap_pct: float, pr_is_recent: bool) -> str:
    """The status word for one distance."""
    if pr_is_recent:
        return NEW_PR
    if not np.isfinite(gap_pct):
        return NO_RECENT
    if gap_pct <= WITHIN_REACH_PCT:
        return WITHIN_REACH
    if gap_pct <= GETTING_CLOSE_PCT:
        return GETTING_CLOSE
    return OFF_PACE


def near_pr_table(best: pd.DataFrame, as_of: pd.Timestamp | None = None,
                  recent_days: int = RECENT_WINDOW_DAYS) -> pd.DataFrame:
    """Per named distance: the PR, the recent best, the gap and a status.

    ``best`` is the long ``DataFrame[activity_id, date, distance_m, time_s]``.
    ``as_of`` defaults to the latest effort's date.  Distances never covered
    are left out; rows come in distance order.
    """
    if best.empty:
        return pd.DataFrame(columns=COLUMNS)
    as_of = best["date"].max() if as_of is None else as_of
    cutoff = as_of - pd.Timedelta(days=recent_days)
    rows = []
    for label, meters in NAMED_DISTANCES_M.items():
        at = best[np.isclose(best["distance_m"], meters)]
        if at.empty:
            continue
        pr = at.loc[at["time_s"].idxmin()]
        recent = at[at["date"] >= cutoff]
        top = recent.loc[recent["time_s"].idxmin()] if not recent.empty else None
        pr_is_recent = pr["date"] >= cutoff
        recent_s = float(top["time_s"]) if top is not None else np.nan
        pr_s = float(pr["time_s"])
        gap_s = recent_s - pr_s
        km = meters / 1000.0
        rows.append({
            "distance": label, "distance_km": km,
            "pr_time_s": pr_s, "pr_date": pr["date"], "pr_activity_id": int(pr["activity_id"]),
            "recent_time_s": recent_s,
            "recent_date": top["date"] if top is not None else pd.NaT,
            "recent_activity_id": int(top["activity_id"]) if top is not None else None,
            "gap_s": gap_s, "gap_pct": gap_s / pr_s * 100.0,
            "target_pace_min_per_km": (pr_s - 1.0) / 60.0 / km,
            "pr_age_days": int((as_of - pr["date"]).days),
            "status": classify(gap_s / pr_s * 100.0, bool(pr_is_recent)),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def chances(table: pd.DataFrame) -> pd.DataFrame:
    """The distances worth a PR attempt now (``Within reach``), closest first."""
    return (table[table["status"] == WITHIN_REACH]
            .sort_values("gap_pct").reset_index(drop=True))
